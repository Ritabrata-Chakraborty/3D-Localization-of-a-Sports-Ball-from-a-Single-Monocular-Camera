#!/usr/bin/env python3
"""
Camera Calibration System
Three modes:
1. calibrate - Extract camera parameters using PnLCalib
2. smooth - Smooth camera parameters with confidence-weighted refinement (stationary camera assumption)
3. visualize - Generate 3×3 grid visualization of camera parameters
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import sys
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add PnLCalib to path
pnlcalib_path = Path(__file__).parent.parent / "PnLCalib"
sys.path.insert(0, str(pnlcalib_path))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib, keypoint_world_coords_2D, keypoint_aux_world_coords_2D
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, \
    get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict
import torchvision.transforms as T
import torchvision.transforms.functional as f
from PIL import Image


class CameraCalibrator:
    """Camera calibration using PnLCalib"""
    
    def __init__(self, weights_kp, weights_line, device='cuda:0'):
        self.device = device
        
        # Load configs
        config_dir = Path(__file__).parent.parent / "PnLCalib" / "config"
        self.cfg = yaml.safe_load(open(config_dir / "hrnetv2_w48.yaml", 'r'))
        self.cfg_l = yaml.safe_load(open(config_dir / "hrnetv2_w48_l.yaml", 'r'))
        
        # Load models
        print(f"Loading keypoint model from {weights_kp}...")
        loaded_state = torch.load(weights_kp, map_location=device)
        self.model_kp = get_cls_net(self.cfg)
        self.model_kp.load_state_dict(loaded_state)
        self.model_kp.to(device)
        self.model_kp.eval()
        
        print(f"Loading line model from {weights_line}...")
        loaded_state_l = torch.load(weights_line, map_location=device)
        self.model_line = get_cls_net_l(self.cfg_l)
        self.model_line.load_state_dict(loaded_state_l)
        self.model_line.to(device)
        self.model_line.eval()
        
        self.transform = T.Resize((540, 960))
        
        print("Models loaded successfully!")
    
    def inference_frame(self, frame, cam, kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True):
        """
        Run PnLCalib inference on a single frame (matches inference.py logic).
        Returns: final_params_dict or None
        Takes an existing FramebyFrameCalib object.
        """
        # Convert to PIL and tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
        
        # Resize if needed (resize to 540x960)
        if frame_tensor.size()[-1] != 960:
            frame_tensor = self.transform(frame_tensor)
        
        frame_tensor = frame_tensor.to(self.device)
        b, c, h, w = frame_tensor.size()
        
        # Run inference
        with torch.no_grad():
            heatmaps = self.model_kp(frame_tensor)
            heatmaps_l = self.model_line(frame_tensor)
        
        # Extract keypoints and lines
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
        
        # Update camera and get parameters
        cam.update(kp_dict, lines_dict)
        final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)
        
        return final_params_dict
    
    def projection_from_cam_params(self, final_params_dict):
        """
        Build projection matrix from calibration result.
        Matches inference.py logic exactly.
        """
        if final_params_dict is None:
            return None
            
        cam_params = final_params_dict["cam_params"]
        x_focal_length = cam_params['x_focal_length']
        y_focal_length = cam_params['y_focal_length']
        principal_point = np.array(cam_params['principal_point'])
        position_meters = np.array(cam_params['position_meters'])
        rotation = np.array(cam_params['rotation_matrix'])

        It = np.eye(4)[:-1]
        It[:, -1] = -position_meters
        Q = np.array([[x_focal_length, 0, principal_point[0]],
                      [0, y_focal_length, principal_point[1]],
                      [0, 0, 1]])
        P = Q @ (rotation @ It)

        return P
    
    def project_2d_to_3d(self, u, v, P, z=0.0):
        """
        Project 2D image point to 3D world coordinates
        Assumes z is known (0 for ground, estimated for air)
        """
        # Create the system: P @ [x, y, z, 1]^T = lambda * [u, v, 1]^T
        # We have 3 equations and 3 unknowns (x, y, lambda)
        
        # P @ [x, y, z, 1]^T = [u, v, 1]^T * lambda
        # P[0,:] @ [x, y, z, 1]^T = u * lambda
        # P[1,:] @ [x, y, z, 1]^T = v * lambda
        # P[2,:] @ [x, y, z, 1]^T = lambda
        
        # Rearranging:
        # P[0,0]*x + P[0,1]*y + P[0,2]*z + P[0,3] = u * (P[2,0]*x + P[2,1]*y + P[2,2]*z + P[2,3])
        # P[1,0]*x + P[1,1]*y + P[1,2]*z + P[1,3] = v * (P[2,0]*x + P[2,1]*y + P[2,2]*z + P[2,3])
        
        # Move z to right side (known):
        # (P[0,0] - u*P[2,0])*x + (P[0,1] - u*P[2,1])*y = u*P[2,3] - P[0,3] + (u*P[2,2] - P[0,2])*z
        # (P[1,0] - v*P[2,0])*x + (P[1,1] - v*P[2,1])*y = v*P[2,3] - P[1,3] + (v*P[2,2] - P[1,2])*z
        
        A = np.array([
            [P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
            [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]]
        ])
        
        b = np.array([
            u * P[2, 3] - P[0, 3] + (u * P[2, 2] - P[0, 2]) * z,
            v * P[2, 3] - P[1, 3] + (v * P[2, 2] - P[1, 2]) * z
        ])
        
        try:
            xy = np.linalg.solve(A, b)
            return xy[0], xy[1], z
        except np.linalg.LinAlgError:
            return None, None, None
    
    def estimate_z_from_diameter(self, diameter_px, focal_avg, real_diameter=0.22):
        """
        Estimate z (height) from apparent diameter
        real_diameter: real ball diameter in meters (default 22cm)
        """
        if diameter_px <= 0:
            return 0.0
        
        # Similar triangles: z = (real_diameter * focal_length) / apparent_diameter
        z = (real_diameter * focal_avg) / diameter_px
        return z


class CamTracker3D:
    """Camera calibration and tracking"""
    
    def __init__(self, results_dir, weights_kp, weights_line, device='cuda:0'):
        self.results_dir = Path(results_dir)
        self.calibrator = CameraCalibrator(weights_kp, weights_line, device)
        
        # Output directories
        self.frames_3d_dir = self.results_dir / "annotated_frames_3d"
        self.frames_smoothed_dir = self.results_dir / "annotated_frames_smoothed"
    
    def calibrate(self, frames_dir):
        """
        Mode 1: Calibrate camera and extract camera intrinsics/extrinsics
        Stores ONLY camera calibration data, not ball positions.
        """
        # Load detections (only used for frame indexing, not ball data)
        # Try multiple locations: results_dir, stage1_detection subdir, or parent/stage1_detection
        possible_paths = [
            self.results_dir / "detections.csv",
            self.results_dir / "stage1_detection" / "detections.csv",
            self.results_dir.parent / "stage1_detection" / "detections.csv",
        ]
        
        detections_path = None
        for path in possible_paths:
            if path.exists():
                detections_path = path
                break
        
        if detections_path is None:
            raise FileNotFoundError(f"Detections not found in any of: {possible_paths}")
        
        df = pd.read_csv(detections_path)
        print(f"Loaded {len(df)} frames from {detections_path}")
        
        # Get frame files
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if not image_files:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        
        # Create output directory
        self.frames_3d_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each frame - extract camera calibration only
        results = []
        
        print(f"Calibrating camera and extracting camera parameters...")
        for idx, row in df.iterrows():
            frame_num = int(row['frame'])
            
            # Load frame
            if frame_num >= len(image_files):
                print(f"\nWarning: Frame {frame_num} out of range")
                continue
            
            img_path = image_files[frame_num]
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"\nWarning: Could not load {img_path}")
                continue
            
            # Get frame dimensions and create fresh FramebyFrameCalib per frame (like inference.py)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
            
            # Run calibration with fresh cam object
            final_params_dict = self.calibrator.inference_frame(frame, cam, kp_threshold=0.3434, 
                                                                line_threshold=0.7867, pnl_refine=True)
            
            if final_params_dict is None:
                print(f"\nWarning: Calibration failed for frame {frame_num}")
                continue
            
            # Extract camera parameters (intrinsics and extrinsics)
            cam_params = final_params_dict["cam_params"]
            
            # Compute reconstruction error directly using PnLCalib's reproj_err method
            # cam.reproj_err expects: object_points (3D world), image_points (2D image)
            if hasattr(cam, 'keypoints_dict') and cam.keypoints_dict:
                # Extract world and image points from keypoints_dict
                obj_pts = []
                img_pts = []
                
                for kp_id, kp_data in cam.keypoints_dict.items():
                    # Get world coordinates from PnLCalib's keypoint coordinates
                    try:
                        if kp_id <= 57:
                            if kp_id < 1 or kp_id > len(keypoint_world_coords_2D):
                                continue  # Skip invalid keypoint IDs
                            wp = keypoint_world_coords_2D[kp_id - 1]
                            z_world = -2.44 if kp_id in [12, 15, 16, 19] else 0.0
                        else:
                            aux_idx = kp_id - 1 - 57
                            if aux_idx < 0 or aux_idx >= len(keypoint_aux_world_coords_2D):
                                continue  # Skip invalid auxiliary keypoint IDs
                            wp = keypoint_aux_world_coords_2D[aux_idx]
                            z_world = 0.0
                        
                        # Convert to 3D world point (already centered in PnLCalib)
                        world_point = np.array([wp[0], wp[1], z_world])
                        obj_pts.append(world_point)
                        
                        # Get detected image point
                        img_pts.append(np.array([kp_data['x'], kp_data['y']]))
                    except (IndexError, KeyError, TypeError):
                        # Skip keypoints with indexing errors
                        continue
                
                # Use PnLCalib's built-in reproj_err method
                if obj_pts and img_pts:
                    try:
                        recon_error = cam.reproj_err(obj_pts, img_pts)
                        if recon_error is None:
                            recon_error = -1.0
                    except Exception as e:
                        # If reproj_err fails, mark as -1.0
                        recon_error = -1.0
                else:
                    recon_error = -1.0
            else:
                recon_error = -1.0
            
            # Store camera calibration data
            rotation_matrix = np.array(cam_params['rotation_matrix'])
            rotation_flat = rotation_matrix.flatten()
            
            result = {
                'frame': frame_num,
                # Camera intrinsics
                'fx': cam_params['x_focal_length'],
                'fy': cam_params['y_focal_length'],
                'cx': cam_params['principal_point'][0],
                'cy': cam_params['principal_point'][1],
                # Camera extrinsics - position in world coordinates
                'cam_x': cam_params['position_meters'][0],
                'cam_y': cam_params['position_meters'][1],
                'cam_z': cam_params['position_meters'][2],
                # Camera rotation matrix (3x3, row-major)
                'r11': rotation_flat[0], 'r12': rotation_flat[1], 'r13': rotation_flat[2],
                'r21': rotation_flat[3], 'r22': rotation_flat[4], 'r23': rotation_flat[5],
                'r31': rotation_flat[6], 'r32': rotation_flat[7], 'r33': rotation_flat[8],
                # Reconstruction error (validation metric)
                'recon_error': recon_error,
            }
            results.append(result)
            
            # Generate annotated frame (no ball data, only camera position visualization)
            self.annotate_frame_camera(frame, cam_params, 
                                      self.frames_3d_dir / f"frame_{frame_num:04d}.jpg")
        
        print(f"Calibration complete! Processed {len(results)} frames.")
        
        # Save to CSV (camera calibration only)
        df_calib = pd.DataFrame(results)
        csv_path = self.results_dir / "camera_calibration.csv"
        df_calib.to_csv(csv_path, index=False)
        print(f"Saved camera calibration to {csv_path}")
        
        return csv_path
    
    
    def annotate_frame_camera(self, frame, cam_params, output_path):
        """Generate annotated frame with camera position and field lines"""
        annotated = frame.copy()
        
        # Draw PnLCalib detected field lines
        self._draw_pnlcalib_lines_projected(annotated, cam_params)
        
        # Draw camera position at top left (in 3D world coordinates)
        # Handle both tuple and array formats
        pos_meters = cam_params['position_meters']
        if isinstance(pos_meters, np.ndarray):
            cam_x, cam_y, cam_z = pos_meters[0], pos_meters[1], pos_meters[2]
        else:
            cam_x, cam_y, cam_z = pos_meters[0], pos_meters[1], pos_meters[2]
        
        # Handle both formats: raw calibration and smoothed CSV
        if 'x_focal_length' in cam_params:
            fx = cam_params['x_focal_length']
            fy = cam_params['y_focal_length']
            focal_avg = (fx + fy) / 2
        else:
            focal_avg = cam_params['focal_length']
        
        pos_text = f"x={cam_x:.2f}m y={cam_y:.2f}m z={cam_z:.2f}m f={focal_avg:.1f}"
        
        # Black outline for readability
        cv2.putText(annotated, pos_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Red text for camera parameters
        cv2.putText(annotated, pos_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        
        cv2.imwrite(str(output_path), annotated)
    
    def annotate_frame(self, frame, detection, world_x, world_y, world_z, focal_avg, output_path, 
                      cam_params=None):
        """Generate annotated frame with ball detection and 3D position"""
        annotated = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        
        # Draw PnLCalib detected field lines using camera parameters
        if cam_params is not None:
            self._draw_pnlcalib_lines_projected(annotated, cam_params)
        
        # Draw bounding box (very thin)
        x, y, w, h = int(detection['x']), int(detection['y']), \
                     int(detection['w']), int(detection['h'])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Draw circle (very thin)
        cx, cy = int(detection['center_x']), int(detection['center_y'])
        radius = int(detection['diameter']) // 2
        cv2.circle(annotated, (cx, cy), radius, (255, 0, 0), 1)
        
        # Draw center dot (small)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
        
        # Draw air/ground label (thin text)
        label = f"{detection['air_ground'].upper()} {int(detection['diameter'])}px"
        label_y = max(y - 15, 35)
        cv2.putText(annotated, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Draw 3D ball position at top (RED text for ball info, with black outline)
        pos_text = f"x={world_x:.2f}m y={world_y:.2f}m z={world_z:.2f}m f={focal_avg:.1f}"
        
        # Black outline for readability
        cv2.putText(annotated, pos_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Red text for ball position
        cv2.putText(annotated, pos_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        
        cv2.imwrite(str(output_path), annotated)
    
    def _draw_pnlcalib_lines_projected(self, frame, cam_params, color=(0, 0, 0), thickness=2):
        """
        Draw PnLCalib detected field lines by projecting world coordinates to image.
        Uses exact logic from inference.py project() function.
        """
        # Standard field line coordinates (world coordinates) - exact copy from inference.py
        lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                        [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                        [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                        [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                        [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                        [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                        [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                        [[0., 37.66, 0.], [0., 37.66, -2.44]],
                        [[0., 30.34, 0.], [0., 30.34, -2.44]],
                        [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                        [[105., 30.34, 0.], [105., 30.34, -2.44]],
                        [[105., 37.66, 0.], [105., 37.66, -2.44]],
                        [[52.5, 0., 0.], [52.5, 68, 0.]],
                        [[0., 68., 0.], [105., 68., 0.]],
                        [[0., 0., 0.], [0., 68., 0.]],
                        [[105., 0., 0.], [105., 68., 0.]],
                        [[0., 0., 0.], [105., 0., 0.]],
                        [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                        [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                        [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                        [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                        [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                        [[99.5, 24.84, 0.], [105., 24.84, 0.]]]
        
        # Build projection matrix from camera parameters (exact logic from projection_from_cam_params)
        # Handle both formats: raw calibration (x_focal_length) and smoothed CSV (focal_length)
        if 'x_focal_length' in cam_params:
            x_focal_length = cam_params['x_focal_length']
            y_focal_length = cam_params['y_focal_length']
        else:
            # Fallback for smoothed data or other formats
            focal = cam_params.get('focal_length', 1000.0)
            x_focal_length = focal
            y_focal_length = focal
        
        # Handle principal_point as either tuple or array
        principal_point = cam_params['principal_point']
        if isinstance(principal_point, (tuple, list)):
            principal_point = np.array(principal_point)
        
        # Handle position_meters as either tuple or array
        position_meters = cam_params['position_meters']
        if isinstance(position_meters, (tuple, list)):
            position_meters = np.array(position_meters)
        
        rotation = np.array(cam_params['rotation_matrix'])

        It = np.eye(4)[:-1]
        It[:, -1] = -position_meters
        Q = np.array([[x_focal_length, 0, principal_point[0]],
                      [0, y_focal_length, principal_point[1]],
                      [0, 0, 1]])
        P = Q @ (rotation @ It)
        
        lines_drawn = 0
        
        # Project each field line - exact logic from inference.py project()
        for line in lines_coords:
            w1 = line[0]
            w2 = line[1]
            i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
            i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
            i1 /= i1[-1]
            i2 /= i2[-1]
            cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, thickness)
            lines_drawn += 1

        # Draw penalty area arcs and center circle - exact logic from inference.py project()
        r = 9.15
        pts1, pts2, pts3 = [], [], []
        
        # Left penalty arc
        base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(37, 143, 50):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts1.append([ipos[0], ipos[1]])

        # Right penalty arc
        base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
        for ang in np.linspace(217, 323, 200):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts2.append([ipos[0], ipos[1]])

        # Center circle
        base_pos = np.array([0, 0, 0., 0.])
        for ang in np.linspace(0, 360, 500):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            ipos /= ipos[-1]
            pts3.append([ipos[0], ipos[1]])

        XEllipse1 = np.array(pts1, np.int32)
        XEllipse2 = np.array(pts2, np.int32)
        XEllipse3 = np.array(pts3, np.int32)
        cv2.polylines(frame, [XEllipse1], False, color, thickness)
        cv2.polylines(frame, [XEllipse2], False, color, thickness)
        cv2.polylines(frame, [XEllipse3], False, color, thickness)
        
        # Field lines drawn
    
    def _draw_pnlcalib_lines(self, frame, lines_dict, h_frame, w_frame, color=(100, 200, 255), thickness=1):
        """Draw PnLCalib detected field lines on the frame"""
        if not lines_dict:
            return
        
        lines_drawn = 0
        # lines_dict is normalized to [0, 1] range, scale to frame size
        for key, points in lines_dict.items():
            if points is None:
                continue
            
            # Handle different point formats
            if isinstance(points, (list, tuple)):
                points_list = list(points)
            elif isinstance(points, np.ndarray):
                points_list = points.tolist()
            else:
                continue
            
            if len(points_list) < 2:
                continue
            
            # Convert normalized points to frame coordinates
            frame_points = []
            for pt in points_list:
                try:
                    if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                        x = int(pt[0] * w_frame)
                        y = int(pt[1] * h_frame)
                        # Ensure points are within frame bounds
                        x = max(0, min(x, w_frame - 1))
                        y = max(0, min(y, h_frame - 1))
                        frame_points.append((x, y))
                except (TypeError, ValueError):
                    continue
            
            # Draw line between consecutive points
            if len(frame_points) >= 2:
                for i in range(len(frame_points) - 1):
                    cv2.line(frame, frame_points[i], frame_points[i + 1], color, thickness)
                    lines_drawn += 1
        
        # Field lines drawn
    
    def smooth(self, frames_dir, method='method1'):
        """
        Mode 2: Smooth camera calibration parameters (stationary camera assumption)
        Uses reconstruction error as a confidence metric for weighted smoothing
        
        Args:
            frames_dir: Directory containing frames
            method: Smoothing method - 'method1' (global poly) or 'method2' (piecewise poly)
        """
        # Load camera calibration data
        csv_path = self.results_dir / "camera_calibration.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Camera calibration not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} camera calibration frames from {csv_path}")
        
        # Analyze reconstruction error as confidence metric
        print("Analyzing reconstruction error as confidence metric...")
        
        # Handle cases where recon_error might be -1.0 (invalid)
        valid_errors = df[df['recon_error'] > 0]['recon_error']
        if len(valid_errors) > 0:
            error_mean = valid_errors.mean()
            error_std = valid_errors.std()
            error_median = valid_errors.median()
            print(f"  Reconstruction Error Stats:")
            print(f"    Mean: {error_mean:.4f} px")
            print(f"    Std Dev: {error_std:.4f} px")
            print(f"    Median: {error_median:.4f} px")
            print(f"    Valid frames: {len(valid_errors)}/{len(df)}")
            
            # Calculate confidence weights: lower error = higher confidence
            # Use inverse exponential: confidence = exp(-error / error_mean)
            df['confidence'] = np.exp(-df['recon_error'].clip(lower=0) / (error_mean + 1e-6))
            df.loc[df['recon_error'] <= 0, 'confidence'] = 0.5  # Low confidence for invalid errors
        else:
            print(f"  Warning: No valid reconstruction errors found. Using uniform weights.")
            df['confidence'] = 1.0
        
        # Normalize confidence weights to [0.1, 1.0] range for better numerical stability
        conf_min = df['confidence'].min()
        conf_max = df['confidence'].max()
        if conf_max > conf_min:
            df['confidence'] = 0.1 + 0.9 * (df['confidence'] - conf_min) / (conf_max - conf_min)
        else:
            df['confidence'] = 0.5
        
        print(f"  Confidence range: [{df['confidence'].min():.3f}, {df['confidence'].max():.3f}]")
        
        # Apply smoothing with confidence weighting
        print("Smoothing camera parameters...")
        
        # ===== 1. CAMERA POSITION: Make constant (stationary camera) =====
        print("1. Camera Position (stationary camera - constant x, y, z):")
        high_conf_threshold = df['confidence'].quantile(0.75)  # Top 25% confidence
        high_conf_frames = df[df['confidence'] >= high_conf_threshold]
        
        if len(high_conf_frames) > 3:
            # Use weighted median from high-confidence frames
            weights = high_conf_frames['confidence'].values
            
            # Weighted median calculation
            def weighted_median(values, weights):
                sorted_idx = np.argsort(values)
                sorted_vals = values[sorted_idx]
                sorted_weights = weights[sorted_idx]
                cumsum = np.cumsum(sorted_weights)
                cutoff = cumsum[-1] / 2.0
                return sorted_vals[np.searchsorted(cumsum, cutoff)]
            
            cam_x_const = weighted_median(high_conf_frames['cam_x'].values, weights)
            cam_y_const = weighted_median(high_conf_frames['cam_y'].values, weights)
            cam_z_const = weighted_median(high_conf_frames['cam_z'].values, weights)
            
            print(f"  Constant position from {len(high_conf_frames)} high-confidence frames:")
            print(f"    cam_x: {cam_x_const:.4f}m")
            print(f"    cam_y: {cam_y_const:.4f}m")
            print(f"    cam_z: {cam_z_const:.4f}m")
            
            # Set all frames to this constant position
            df['cam_x'] = cam_x_const
            df['cam_y'] = cam_y_const
            df['cam_z'] = cam_z_const
            
            print(f"  ✓ All frames set to constant position")
        else:
            # Fallback: use simple median
            cam_x_const = df['cam_x'].median()
            cam_y_const = df['cam_y'].median()
            cam_z_const = df['cam_z'].median()
            df['cam_x'] = cam_x_const
            df['cam_y'] = cam_y_const
            df['cam_z'] = cam_z_const
            print(f"  Warning: Not enough high-confidence frames, using simple median")
        
        # ===== 2. ROTATION MATRIX: Smooth via quaternions =====
        print(f"2. Rotation Matrix (smooth changes via quaternions - method: {method}):")
        
        # Extract rotation matrices and convert to quaternions
        rotations = []
        for idx in range(len(df)):
            R_mat = np.array([
                [df.loc[idx, 'r11'], df.loc[idx, 'r12'], df.loc[idx, 'r13']],
                [df.loc[idx, 'r21'], df.loc[idx, 'r22'], df.loc[idx, 'r23']],
                [df.loc[idx, 'r31'], df.loc[idx, 'r32'], df.loc[idx, 'r33']]
            ])
            rotations.append(R.from_matrix(R_mat))
        
        # Convert to quaternions (w, x, y, z)
        quats = np.array([rot.as_quat() for rot in rotations])  # shape: (N, 4)
        
        # Ensure quaternion continuity (avoid sign flips)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        
        # Apply smoothing method
        if method == 'method1':
            print(f"  Smoothing quaternions with global 2nd degree polynomial...")
            n = len(df)
            t = np.linspace(0, 1, n)
            W = np.diag(df['confidence'].values)
            A = np.vstack([np.ones(n), t, t**2]).T
            
            quats_smooth = np.zeros_like(quats)
            for j in range(4):
                coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ quats[:, j])
                quats_smooth[:, j] = A @ coeffs
        
        elif method == 'method2':
            print(f"  Smoothing quaternions with piecewise 2nd degree polynomial...")
            n = len(df)
            
            # Find split point (use focal length as proxy for zoom behavior)
            split_idx = df['fx'].idxmax()
            min_frames = max(10, int(0.2 * n))
            if split_idx < min_frames:
                split_idx = min_frames
            elif split_idx > n - min_frames:
                split_idx = n - min_frames
            
            print(f"    Split at frame {split_idx} (min {min_frames} frames per segment)")
            
            # Left segment
            n_left = split_idx + 1
            t_left = np.linspace(0, 1, n_left)
            W_left = np.diag(df['confidence'][:n_left].values)
            A_left = np.vstack([np.ones(n_left), t_left, t_left**2]).T
            
            # Right segment
            n_right = n - split_idx
            t_right = np.linspace(0, 1, n_right)
            W_right = np.diag(df['confidence'][split_idx:].values)
            A_right = np.vstack([np.ones(n_right), t_right, t_right**2]).T
            
            quats_smooth = np.zeros_like(quats)
            for j in range(4):
                # Fit left
                coeffs_left = np.linalg.solve(A_left.T @ W_left @ A_left,
                                             A_left.T @ W_left @ quats[:n_left, j])
                left_vals = A_left @ coeffs_left
                
                # Fit right
                coeffs_right = np.linalg.solve(A_right.T @ W_right @ A_right,
                                              A_right.T @ W_right @ quats[split_idx:, j])
                right_vals = A_right @ coeffs_right
                
                # Combine with continuity at split
                quats_smooth[:, j] = np.concatenate([
                    left_vals[:-1],
                    [(left_vals[-1] + right_vals[0]) / 2],
                    right_vals[1:]
                ])
        
        # Normalize quaternions
        quats_smooth = quats_smooth / np.linalg.norm(quats_smooth, axis=1, keepdims=True)
        
        # Convert back to rotation matrices
        rotation_cols = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
        for idx in range(len(df)):
            rot_smooth = R.from_quat(quats_smooth[idx])
            R_mat_smooth = rot_smooth.as_matrix()
            R_flat = R_mat_smooth.flatten()
            for i, col in enumerate(rotation_cols):
                df.loc[idx, col] = R_flat[i]
        
        print(f"  ✓ Rotation matrices smoothed (automatically orthogonal)")
        
        # ===== 3. FOCAL LENGTH & INTRINSICS: Smooth changes =====
        print(f"3. Focal Length & Intrinsics (smooth changes - method: {method}):")
        
        # Principal point remains constant (use median)
        df['cx'] = df['cx'].median()
        df['cy'] = df['cy'].median()
        print(f"  Principal point (cx, cy): constant at ({df['cx'].iloc[0]:.2f}, {df['cy'].iloc[0]:.2f})")
        
        # Apply smoothing method to focal lengths
        if method == 'method1':
            print(f"  Smoothing focal lengths with global 2nd degree polynomial...")
            n = len(df)
            t = np.linspace(0, 1, n)
            W = np.diag(df['confidence'].values)
            A = np.vstack([np.ones(n), t, t**2]).T
            
            for col in ['fx', 'fy']:
                y = df[col].values
                coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
                df[col] = A @ coeffs
                print(f"    {col}: a0={coeffs[0]:.2f}, a1={coeffs[1]:.2f}, a2={coeffs[2]:.2f}")
            
            print(f"  ✓ Global polynomial fit complete")
        
        elif method == 'method2':
            print(f"  Smoothing focal lengths with piecewise 2nd degree polynomial...")
            n = len(df)
            
            # Use same split point as quaternions
            split_idx = df['fx'].idxmax()
            min_frames = max(10, int(0.2 * n))
            if split_idx < min_frames:
                split_idx = min_frames
            elif split_idx > n - min_frames:
                split_idx = n - min_frames
            
            print(f"    Split at frame {split_idx}")
            
            # Left segment
            n_left = split_idx + 1
            t_left = np.linspace(0, 1, n_left)
            W_left = np.diag(df['confidence'][:n_left].values)
            A_left = np.vstack([np.ones(n_left), t_left, t_left**2]).T
            
            # Right segment
            n_right = n - split_idx
            t_right = np.linspace(0, 1, n_right)
            W_right = np.diag(df['confidence'][split_idx:].values)
            A_right = np.vstack([np.ones(n_right), t_right, t_right**2]).T
            
            for col in ['fx', 'fy']:
                y_left = df[col][:n_left].values
                y_right = df[col][split_idx:].values
                
                # Fit left
                coeffs_left = np.linalg.solve(A_left.T @ W_left @ A_left,
                                             A_left.T @ W_left @ y_left)
                left_vals = A_left @ coeffs_left
                
                # Fit right
                coeffs_right = np.linalg.solve(A_right.T @ W_right @ A_right,
                                              A_right.T @ W_right @ y_right)
                right_vals = A_right @ coeffs_right
                
                # Combine with continuity at split
                combined = np.concatenate([
                    left_vals[:-1],
                    [(left_vals[-1] + right_vals[0]) / 2],
                    right_vals[1:]
                ])
                df[col] = combined
                
                print(f"    {col} left: a0={coeffs_left[0]:.2f}, a1={coeffs_left[1]:.2f}, a2={coeffs_left[2]:.2f}")
                print(f"    {col} right: a0={coeffs_right[0]:.2f}, a1={coeffs_right[1]:.2f}, a2={coeffs_right[2]:.2f}")
            
            print(f"  ✓ Piecewise polynomial fit complete")
        
        print("✓ Smoothing complete!")
        
        # Save smoothed data (including confidence for analysis)
        if method == 'method1':
            output_csv = self.results_dir / "camera_calibration_smoothed_method1.csv"
        elif method == 'method2':
            output_csv = self.results_dir / "camera_calibration_smoothed_method2.csv"
        else:
            output_csv = self.results_dir / "camera_calibration_smoothed.csv"
        
        df.to_csv(output_csv, index=False)
        print(f"Saved smoothed camera calibration to {output_csv}")
        
        # Save statistics for analysis
        stats_csv = self.results_dir / "camera_calibration_stats.csv"
        stats = pd.DataFrame({
            'metric': ['recon_error_mean', 'recon_error_std', 'recon_error_median',
                      'confidence_min', 'confidence_max', 'high_conf_frames',
                      'cam_x_constant', 'cam_y_constant', 'cam_z_constant',
                      'cam_x_std', 'cam_y_std', 'cam_z_std'],
            'value': [error_mean if len(valid_errors) > 0 else -1,
                     error_std if len(valid_errors) > 0 else -1,
                     error_median if len(valid_errors) > 0 else -1,
                     df['confidence'].min(),
                     df['confidence'].max(),
                     len(high_conf_frames) if len(high_conf_frames) > 3 else -1,
                     df['cam_x'].mean(),
                     df['cam_y'].mean(),
                     df['cam_z'].mean(),
                     df['cam_x'].std(),
                     df['cam_y'].std(),
                     df['cam_z'].std()]
        })
        stats.to_csv(stats_csv, index=False)
        print(f"Saved smoothing statistics to {stats_csv}")
        print(f"Position std deviations (should be ~0 for stationary camera):")
        print(f"  cam_x: {df['cam_x'].std():.6f}m")
        print(f"  cam_y: {df['cam_y'].std():.6f}m")
        print(f"  cam_z: {df['cam_z'].std():.6f}m")


class CameraVisualizer:
    """Visualize camera calibration parameters comparing raw and smoothed methods"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
        # Load raw calibration (required)
        raw_path = self.results_dir / "camera_calibration.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw camera calibration not found: {raw_path}")
        
        self.df_raw = pd.read_csv(raw_path)
        print(f"✓ Loaded raw calibration: {len(self.df_raw)} frames")
        
        # Load smoothed methods (optional)
        method1_path = self.results_dir / "camera_calibration_smoothed_method1.csv"
        method2_path = self.results_dir / "camera_calibration_smoothed_method2.csv"
        
        self.df_method1 = None
        self.df_method2 = None
        
        if method1_path.exists():
            self.df_method1 = pd.read_csv(method1_path)
            print(f"✓ Loaded Method 1 (Global Polynomial): {len(self.df_method1)} frames")
        
        if method2_path.exists():
            self.df_method2 = pd.read_csv(method2_path)
            print(f"✓ Loaded Method 2 (Piecewise Polynomial): {len(self.df_method2)} frames")
    
    def _plot_parameter(self, ax, frames, param_name, title, ylabel, show_legend=True):
        """Helper to plot a single parameter with raw + smoothed methods"""
        # Plot raw (gray dots/line)
        ax.plot(frames, self.df_raw[param_name].values, 'o-', color='gray', 
               linewidth=0.5, markersize=3, alpha=0.4, label='Raw')
        
        # Plot Method 1 (Global Polynomial) if available
        if self.df_method1 is not None:
            ax.plot(frames, self.df_method1[param_name].values, '-', color='red',
                   linewidth=2.5, alpha=0.8, label='Method1 (Global)')
        
        # Plot Method 2 (Piecewise Polynomial) if available
        if self.df_method2 is not None:
            ax.plot(frames, self.df_method2[param_name].values, '-', color='green',
                   linewidth=2.5, alpha=0.8, label='Method2 (Piecewise)')
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        if show_legend:
            ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_calibration(self, output_path=None):
        """
        Generate 2×3 subplot visualization with overlaid original + smoothed data:
        Row 1, Col 1: Position (X, Y, Z) - combined
        Row 1, Col 2: Rotation (R11, R12, R13 | R21, R22, R23 | R31, R32, R33)
        Row 1, Col 3: Focal Length (fx)
        Row 2, Col 1: Principal Point (cx, cy)
        Row 2, Col 2: Stats summary
        Row 2, Col 3: Legend/Info
        """
        if output_path is None:
            output_path = self.results_dir / "camera_calibration_visualization.png"
        
        output_path = Path(output_path)
        
        # Extract data
        frames = self.df_raw['frame'].values
        
        # Create figure with 2×3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Camera Calibration: Original vs Smoothed Comparison', 
                     fontsize=16, fontweight='bold')
        
        # ========== ROW 1, COL 1: Position (X, Y, Z) ==========
        ax = axes[0, 0]
        # Plot raw
        ax.plot(frames, self.df_raw['cam_x'].values, 'o-', color='black', 
               linewidth=1, markersize=2, alpha=0.5, label='Raw (X)')
        ax.plot(frames, self.df_raw['cam_y'].values, 's-', color='black', 
               linewidth=1, markersize=2, alpha=0.5, label='Raw (Y)')
        ax.plot(frames, self.df_raw['cam_z'].values, '^-', color='black', 
               linewidth=1, markersize=2, alpha=0.5, label='Raw (Z)')
        
        # Plot method 1 if available
        if self.df_method1 is not None:
            ax.plot(frames, self.df_method1['cam_x'].values, '-', color='blue',
                   linewidth=2, alpha=0.7, label='M1 (X)')
            ax.plot(frames, self.df_method1['cam_y'].values, '-', color='green',
                   linewidth=2, alpha=0.7, label='M1 (Y)')
            ax.plot(frames, self.df_method1['cam_z'].values, '-', color='red',
                   linewidth=2, alpha=0.7, label='M1 (Z)')
        
        # Plot method 2 if available
        if self.df_method2 is not None:
            ax.plot(frames, self.df_method2['cam_x'].values, '--', color='blue',
                   linewidth=2, alpha=0.7, label='M2 (X)')
            ax.plot(frames, self.df_method2['cam_y'].values, '--', color='green',
                   linewidth=2, alpha=0.7, label='M2 (Y)')
            ax.plot(frames, self.df_method2['cam_z'].values, '--', color='red',
                   linewidth=2, alpha=0.7, label='M2 (Z)')
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Position (meters)', fontsize=11, fontweight='bold')
        ax.set_title('Position (X, Y, Z)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ========== ROW 1, COL 2: Rotation Matrix (R11-R33) ==========
        ax = axes[0, 1]
        rotation_cols = [('r11', 'black'), ('r12', 'blue'), ('r13', 'red'),
                        ('r21', 'black'), ('r22', 'blue'), ('r23', 'red'),
                        ('r31', 'black'), ('r32', 'blue'), ('r33', 'red')]
        
        # Plot all rotation components (raw + smoothed)
        for col, color in rotation_cols:
            ax.plot(frames, self.df_raw[col].values, 'o-', color=color, 
                   linewidth=0.5, markersize=1.5, alpha=0.3)
            
            if self.df_method1 is not None:
                ax.plot(frames, self.df_method1[col].values, '-', color=color,
                       linewidth=1.5, alpha=0.6)
            
            if self.df_method2 is not None:
                ax.plot(frames, self.df_method2[col].values, '--', color=color,
                       linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Rotation Matrix Value', fontsize=11, fontweight='bold')
        ax.set_title('Rotation Matrix (R11-R33)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # ========== ROW 1, COL 3: Focal Length (fx) ==========
        ax = axes[0, 2]
        ax.plot(frames, self.df_raw['fx'].values, 'o-', color='black',
               linewidth=1, markersize=2, alpha=0.5, label='Raw')
        
        if self.df_method1 is not None:
            ax.plot(frames, self.df_method1['fx'].values, '-', color='blue',
                   linewidth=2, alpha=0.8, label='Method 1')
        
        if self.df_method2 is not None:
            ax.plot(frames, self.df_method2['fx'].values, '--', color='green',
                   linewidth=2, alpha=0.8, label='Method 2')
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Focal Length (pixels)', fontsize=11, fontweight='bold')
        ax.set_title('Focal Length (fx)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # ========== ROW 2, COL 1: Principal Point (cx, cy) ==========
        ax = axes[1, 0]
        ax.plot(frames, self.df_raw['cx'].values, 'o-', color='black',
               linewidth=1, markersize=2, alpha=0.5, label='Raw (cx)')
        ax.plot(frames, self.df_raw['cy'].values, 's-', color='black',
               linewidth=1, markersize=2, alpha=0.5, label='Raw (cy)')
        
        if self.df_method1 is not None:
            ax.plot(frames, self.df_method1['cx'].values, '-', color='blue',
                   linewidth=2, alpha=0.8, label='M1 (cx)')
            ax.plot(frames, self.df_method1['cy'].values, '-', color='red',
                   linewidth=2, alpha=0.8, label='M1 (cy)')
        
        if self.df_method2 is not None:
            ax.plot(frames, self.df_method2['cx'].values, '--', color='blue',
                   linewidth=2, alpha=0.8, label='M2 (cx)')
            ax.plot(frames, self.df_method2['cy'].values, '--', color='red',
                   linewidth=2, alpha=0.8, label='M2 (cy)')
        
        ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax.set_ylabel('Principal Point (pixels)', fontsize=11, fontweight='bold')
        ax.set_title('Principal Point (cx, cy)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # ========== ROW 2, COL 2: Statistics ==========
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = "=== Statistics (Raw) ===\n\n"
        stats_text += f"Total Frames: {len(frames)}\n\n"
        stats_text += "Position (m):\n"
        stats_text += f"  X: {self.df_raw['cam_x'].mean():.4f} ± {self.df_raw['cam_x'].std():.4f}\n"
        stats_text += f"  Y: {self.df_raw['cam_y'].mean():.4f} ± {self.df_raw['cam_y'].std():.4f}\n"
        stats_text += f"  Z: {self.df_raw['cam_z'].mean():.4f} ± {self.df_raw['cam_z'].std():.4f}\n\n"
        stats_text += "Focal Length (px):\n"
        stats_text += f"  fx: {self.df_raw['fx'].mean():.2f} ± {self.df_raw['fx'].std():.2f}\n\n"
        stats_text += "Principal Point (px):\n"
        stats_text += f"  cx: {self.df_raw['cx'].mean():.2f} ± {self.df_raw['cx'].std():.2f}\n"
        stats_text += f"  cy: {self.df_raw['cy'].mean():.2f} ± {self.df_raw['cy'].std():.2f}\n"
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ========== ROW 2, COL 3: Legend/Info ==========
        ax = axes[1, 2]
        ax.axis('off')
        
        legend_text = "=== Legend ===\n\n"
        legend_text += "Raw Data:\n"
        legend_text += "  ● Markers with dots\n"
        legend_text += "  ○ Low opacity (0.3-0.5)\n\n"
        legend_text += "Method 1 (Global Poly):\n"
        legend_text += "  — Solid line (blue)\n"
        legend_text += "  ○ Higher opacity (0.6-0.8)\n\n"
        legend_text += "Method 2 (Piecewise):\n"
        legend_text += "  – – Dashed line (green)\n"
        legend_text += "  ○ Higher opacity (0.6-0.8)\n\n"
        
        loaded_methods = []
        if self.df_method1 is not None:
            loaded_methods.append("✓ Method 1 Loaded")
        if self.df_method2 is not None:
            loaded_methods.append("✓ Method 2 Loaded")
        
        legend_text += "=== Loaded Data ===\n"
        legend_text += "✓ Raw Data\n"
        for method in loaded_methods:
            legend_text += f"{method}\n"
        
        ax.text(0.1, 0.95, legend_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration System")
    parser.add_argument("--mode", type=str, required=True, choices=['calibrate', 'smooth', 'visualize'],
                       help="Mode: calibrate, smooth, or visualize")
    parser.add_argument("--frames", type=str, default="downloads/frames/V2_1",
                       help="Input frames directory")
    parser.add_argument("--results", type=str, default="results/V2_1",
                       help="Results directory")
    parser.add_argument("--weights_kp", type=str, default="PnLCalib/SV_kp",
                       help="Path to keypoint model weights")
    parser.add_argument("--weights_line", type=str, default="PnLCalib/SV_lines",
                       help="Path to line model weights")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for inference (cuda:0 or cpu)")
    parser.add_argument("--smooth-method", type=str, default="method1",
                       choices=['method1', 'method2'],
                       help="Smoothing method: method1 (global poly, default) or method2 (piecewise poly)")
    
    args = parser.parse_args()
    
    try:
        tracker = CamTracker3D(args.results, args.weights_kp, args.weights_line, args.device)
        
        if args.mode == 'calibrate':
            print("="*60)
            print("CAMERA CALIBRATION MODE")
            print("="*60)
            
            csv_path = tracker.calibrate(args.frames)
            
            print("\n" + "="*60)
            print("CALIBRATION COMPLETE!")
            print("="*60)
            print(f"Results: {csv_path}")
            print(f"Annotated frames: {tracker.frames_3d_dir}")
            print("\nNext step: Run smooth mode")
            print(f"  python3 calibrate_and_track.py --mode smooth --frames {args.frames} --results {args.results}")
        
        elif args.mode == 'smooth':
            print("="*60)
            print(f"SMOOTHING MODE - Method: {args.smooth_method}")
            print("="*60)
            
            tracker.smooth(args.frames, method=args.smooth_method)
            
            print("\n" + "="*60)
            print("SMOOTHING COMPLETE!")
            print("="*60)
            if args.smooth_method == 'method1':
                print(f"Updated: {args.results}/camera_calibration_smoothed_method1.csv")
            elif args.smooth_method == 'method2':
                print(f"Updated: {args.results}/camera_calibration_smoothed_method2.csv")
        
        elif args.mode == 'visualize':
            print("="*60)
            print("VISUALIZATION MODE")
            print("="*60)
            
            # Initialize visualizer
            visualizer = CameraVisualizer(args.results)
            
            # Generate plots
            visualizer.plot_calibration()
            
            print("\n" + "="*60)
            print("VISUALIZATION COMPLETE!")
            print("="*60)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

