#!/usr/bin/env python3
"""
Stage 5: Finalization System
Generates annotated video frames with comprehensive overlays and creates final visualizations
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import sys

# Add PnLCalib to path
pnlcalib_path = Path(__file__).parent.parent / "PnLCalib"
sys.path.insert(0, str(pnlcalib_path))


class Finalizer:
    """Finalization system for generating annotated frames and comprehensive visualizations"""
    
    def __init__(self, frames_dir, detection_csv, calibration_csv, calibration_raw_csv,
                 ball_3d_csv, ball_3d_fitted_csv, output_dir, weights_kp=None, weights_line=None):
        """
        Initialize Finalizer
        
        Args:
            frames_dir: Directory containing original video frames
            detection_csv: Path to smoothed detection CSV
            calibration_csv: Path to smoothed calibration CSV
            calibration_raw_csv: Path to raw calibration CSV
            ball_3d_csv: Path to raw 3D ball positions CSV
            ball_3d_fitted_csv: Path to fitted 3D ball positions CSV
            output_dir: Output directory for finalization results
            weights_kp: Path to PnLCalib keypoint weights (optional)
            weights_line: Path to PnLCalib line weights (optional)
        """
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir = self.output_dir / "annotated_frames"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.detection_df = pd.read_csv(detection_csv)
        print(f"  ✓ Loaded {len(self.detection_df)} detections")
        
        self.calibration_df = pd.read_csv(calibration_csv)
        print(f"  ✓ Loaded {len(self.calibration_df)} calibration frames (smoothed)")
        
        self.calibration_raw_df = pd.read_csv(calibration_raw_csv)
        print(f"  ✓ Loaded {len(self.calibration_raw_df)} calibration frames (raw)")
        
        self.ball_3d_df = pd.read_csv(ball_3d_csv)
        print(f"  ✓ Loaded {len(self.ball_3d_df)} 3D ball positions (raw)")
        
        self.ball_3d_fitted_df = pd.read_csv(ball_3d_fitted_csv)
        print(f"  ✓ Loaded {len(self.ball_3d_fitted_df)} 3D ball positions (fitted)")
        
        # Get frame files
        self.image_files = sorted(self.frames_dir.glob("*.jpg")) + \
                          sorted(self.frames_dir.glob("*.png")) + \
                          sorted(self.frames_dir.glob("*.jpeg"))
        
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {frames_dir}")
        
        print(f"  ✓ Found {len(self.image_files)} frames")
    
    @staticmethod
    def rotation_matrix_to_euler(R_mat):
        """
        Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees
        
        Args:
            R_mat: 3x3 rotation matrix
            
        Returns:
            (roll, pitch, yaw) in degrees
        """
        rot = R.from_matrix(R_mat)
        euler = rot.as_euler('xyz', degrees=True)
        return euler[0], euler[1], euler[2]  # roll, pitch, yaw
    
    def draw_pnlcalib_lines(self, frame, cam_params, color=(0, 0, 0), thickness=1):
        """
        Draw PnLCalib field lines on frame
        Adapted from calibrate.py _draw_pnlcalib_lines_projected()
        
        Args:
            frame: Image frame
            cam_params: Camera parameters dict with keys: fx, fy, cx, cy, cam_x, cam_y, cam_z, r11-r33
            color: Line color (default: black)
            thickness: Line thickness (default: 1)
        """
        # Standard field line coordinates (world coordinates)
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
        
        # Build projection matrix
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        
        position_meters = np.array([cam_params['cam_x'], cam_params['cam_y'], cam_params['cam_z']])
        rotation = np.array([[cam_params['r11'], cam_params['r12'], cam_params['r13']],
                            [cam_params['r21'], cam_params['r22'], cam_params['r23']],
                            [cam_params['r31'], cam_params['r32'], cam_params['r33']]])
        
        It = np.eye(4)[:-1]
        It[:, -1] = -position_meters
        Q = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        P = Q @ (rotation @ It)
        
        # Project and draw lines
        for line in lines_coords:
            w1 = line[0]
            w2 = line[1]
            i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
            i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
            
            if i1[-1] > 0 and i2[-1] > 0:  # Check if points are in front of camera
                i1 /= i1[-1]
                i2 /= i2[-1]
                cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, thickness)
        
        # Draw penalty area arcs and center circle
        r = 9.15
        
        # Left penalty arc
        base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
        pts1 = []
        for ang in np.linspace(37, 143, 50):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            if ipos[-1] > 0:
                ipos /= ipos[-1]
                pts1.append([ipos[0], ipos[1]])
        
        # Right penalty arc
        base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
        pts2 = []
        for ang in np.linspace(217, 323, 50):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            if ipos[-1] > 0:
                ipos /= ipos[-1]
                pts2.append([ipos[0], ipos[1]])
        
        # Center circle
        base_pos = np.array([0, 0, 0., 0.])
        pts3 = []
        for ang in np.linspace(0, 360, 100):
            ang_rad = np.deg2rad(ang)
            pos = base_pos + np.array([r*np.sin(ang_rad), r*np.cos(ang_rad), 0., 1.])
            ipos = P @ pos
            if ipos[-1] > 0:
                ipos /= ipos[-1]
                pts3.append([ipos[0], ipos[1]])
        
        if pts1:
            cv2.polylines(frame, [np.array(pts1, np.int32)], False, color, thickness)
        if pts2:
            cv2.polylines(frame, [np.array(pts2, np.int32)], False, color, thickness)
        if pts3:
            cv2.polylines(frame, [np.array(pts3, np.int32)], False, color, thickness)
    
    def annotate_frame(self, frame_idx):
        """
        Annotate a single frame with all overlays
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Annotated frame or None if frame not found
        """
        if frame_idx >= len(self.image_files):
            return None
        
        # Load frame
        img_path = self.image_files[frame_idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            return None
        
        # Get detection data
        det_row = self.detection_df[self.detection_df['frame'] == frame_idx]
        if len(det_row) == 0:
            return frame  # Return unmodified frame if no detection
        det_row = det_row.iloc[0]
        
        # Get calibration data (smoothed for camera position display)
        cal_row = self.calibration_df[self.calibration_df['frame'] == frame_idx]
        if len(cal_row) == 0:
            return frame  # Return unmodified frame if no calibration
        cal_row = cal_row.iloc[0]
        
        # Get RAW calibration data for PnLCalib line drawing (as done in Stage 2)
        cal_raw_row = self.calibration_raw_df[self.calibration_raw_df['frame'] == frame_idx]
        if len(cal_raw_row) == 0:
            return frame  # Return unmodified frame if no raw calibration
        cal_raw_row = cal_raw_row.iloc[0]
        
        # Get 3D ball position (try fitted first, then raw)
        ball_3d_row = self.ball_3d_fitted_df[self.ball_3d_fitted_df['frame'] == frame_idx]
        if len(ball_3d_row) == 0:
            ball_3d_row = self.ball_3d_df[self.ball_3d_df['frame'] == frame_idx]
        
        # Draw PnLCalib field lines using RAW calibration (not smoothed)
        # This matches the approach in Stage 2 (calibrate.py annotate_frame_camera)
        cam_params_raw = {
            'fx': cal_raw_row['fx'],
            'fy': cal_raw_row['fy'],
            'cx': cal_raw_row['cx'],
            'cy': cal_raw_row['cy'],
            'cam_x': cal_raw_row['cam_x'],
            'cam_y': cal_raw_row['cam_y'],
            'cam_z': cal_raw_row['cam_z'],
            'r11': cal_raw_row['r11'], 'r12': cal_raw_row['r12'], 'r13': cal_raw_row['r13'],
            'r21': cal_raw_row['r21'], 'r22': cal_raw_row['r22'], 'r23': cal_raw_row['r23'],
            'r31': cal_raw_row['r31'], 'r32': cal_raw_row['r32'], 'r33': cal_raw_row['r33']
        }
        self.draw_pnlcalib_lines(frame, cam_params_raw, color=(0, 0, 0), thickness=1)
        
        # Draw bbox and center if detection exists
        if pd.notna(det_row['x']) and pd.notna(det_row['center_x']):
            x, y = int(det_row['x']), int(det_row['y'])
            w, h = int(det_row['w']), int(det_row['h'])
            cx, cy = int(det_row['center_x']), int(det_row['center_y'])
            
            # Draw bbox (green, thin)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Draw center dot (red)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            
            # Text above bbox: AIR/GROUND {diameter}px
            air_ground = det_row['air_ground'].upper()
            diameter = det_row['diameter']
            label_above = f"{air_ground} {diameter:.1f}px"
            cv2.putText(frame, label_above, (x, max(y - 5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Text below bbox: 3D position
            if len(ball_3d_row) > 0:
                ball_3d = ball_3d_row.iloc[0]
                # Try fitted columns first, then raw
                if 'ball_x' in ball_3d:
                    bx, by, bz = ball_3d['ball_x'], ball_3d['ball_y'], ball_3d['ball_z']
                else:
                    bx, by, bz = ball_3d['x'], ball_3d['y'], ball_3d['z']
                
                label_below = f"3D: ({bx:.2f}, {by:.2f}, {bz:.2f})m"
                cv2.putText(frame, label_below, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Camera parameters at top left (use SMOOTHED calibration for display)
        R_mat = np.array([[cal_row['r11'], cal_row['r12'], cal_row['r13']],
                         [cal_row['r21'], cal_row['r22'], cal_row['r23']],
                         [cal_row['r31'], cal_row['r32'], cal_row['r33']]])
        roll, pitch, yaw = self.rotation_matrix_to_euler(R_mat)
        
        cam_text = f"Cam: ({cal_row['cam_x']:.1f}, {cal_row['cam_y']:.1f}, {cal_row['cam_z']:.1f})m"
        rot_text = f"Rot: ({roll:.1f}, {pitch:.1f}, {yaw:.1f})deg"
        
        cv2.putText(frame, cam_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, rot_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def generate_video(self):
        """Generate annotated video and save individual frames"""
        print("\nGenerating annotated frames and video...")
        
        # Get video properties from first frame
        first_frame = cv2.imread(str(self.image_files[0]))
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        video_path = self.output_dir / "final_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
        
        # Process all frames
        frames_processed = 0
        for frame_idx in range(len(self.image_files)):
            annotated = self.annotate_frame(frame_idx)
            
            if annotated is not None:
                # Save annotated frame
                output_path = self.annotated_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_path), annotated)
                
                # Write to video
                video_writer.write(annotated)
                frames_processed += 1
        
        video_writer.release()
        print(f"✓ Processed {frames_processed} frames")
        print(f"✓ Video saved to {video_path}")
        print(f"✓ Frames saved to {self.annotated_dir}")
    
    def create_final_csv(self):
        """Create final combined CSV with all data"""
        print("\nCreating final CSV...")
        
        # Merge all dataframes
        final_data = []
        
        for frame_idx in range(len(self.image_files)):
            # Get detection
            det_row = self.detection_df[self.detection_df['frame'] == frame_idx]
            if len(det_row) == 0:
                continue
            det_row = det_row.iloc[0]
            
            # Get calibration
            cal_row = self.calibration_df[self.calibration_df['frame'] == frame_idx]
            if len(cal_row) == 0:
                continue
            cal_row = cal_row.iloc[0]
            
            # Get 3D ball positions
            ball_raw = self.ball_3d_df[self.ball_3d_df['frame'] == frame_idx]
            ball_fitted = self.ball_3d_fitted_df[self.ball_3d_fitted_df['frame'] == frame_idx]
            
            # Build row
            row = {
                'frame': frame_idx,
                'center_x': det_row.get('center_x'),
                'center_y': det_row.get('center_y'),
                'diameter': det_row.get('diameter'),
                'air_ground': det_row.get('air_ground'),
            }
            
            # Add raw 3D positions
            if len(ball_raw) > 0:
                ball_raw = ball_raw.iloc[0]
                row['ball_x_raw'] = ball_raw.get('x')
                row['ball_y_raw'] = ball_raw.get('y')
                row['ball_z_raw'] = ball_raw.get('z')
            else:
                row['ball_x_raw'] = None
                row['ball_y_raw'] = None
                row['ball_z_raw'] = None
            
            # Add fitted 3D positions
            if len(ball_fitted) > 0:
                ball_fitted = ball_fitted.iloc[0]
                row['ball_x_fitted'] = ball_fitted.get('ball_x')
                row['ball_y_fitted'] = ball_fitted.get('ball_y')
                row['ball_z_fitted'] = ball_fitted.get('ball_z')
            else:
                row['ball_x_fitted'] = None
                row['ball_y_fitted'] = None
                row['ball_z_fitted'] = None
            
            # Add camera parameters
            R_mat = np.array([[cal_row['r11'], cal_row['r12'], cal_row['r13']],
                             [cal_row['r21'], cal_row['r22'], cal_row['r23']],
                             [cal_row['r31'], cal_row['r32'], cal_row['r33']]])
            roll, pitch, yaw = self.rotation_matrix_to_euler(R_mat)
            
            row['cam_x'] = cal_row['cam_x']
            row['cam_y'] = cal_row['cam_y']
            row['cam_z'] = cal_row['cam_z']
            row['roll'] = roll
            row['pitch'] = pitch
            row['yaw'] = yaw
            
            final_data.append(row)
        
        # Create DataFrame and save
        df_final = pd.DataFrame(final_data)
        csv_path = self.output_dir / "final_positions.csv"
        df_final.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"✓ Final CSV saved to {csv_path}")
        print(f"  Total rows: {len(df_final)}")
    
    def plot_3d_trajectories(self):
        """Create 3D plot with ball and camera trajectories with enhanced styling"""
        print("\nGenerating 3D trajectory plot...")
        
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Ball trajectory - raw (blue scatter with size based on air/ground)
        ball_raw = self.ball_3d_df.dropna(subset=['ball_x', 'ball_y', 'ball_z'])
        if len(ball_raw) > 0:
            # Color by air/ground status
            colors = ['#FF6B6B' if ag == 'air' else '#4169E1' 
                     for ag in ball_raw['air_ground']]
            sizes = [50 if ag == 'air' else 30 for ag in ball_raw['air_ground']]
            
            for i, (idx, row) in enumerate(ball_raw.iterrows()):
                ax.scatter(row['ball_x'], row['ball_y'], -row['ball_z'],
                          c=colors[i], s=sizes[i], alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add scatter legend
            ax.scatter([], [], c='#FF6B6B', s=50, alpha=0.7, edgecolors='black', 
                      linewidth=0.5, label='Ball (raw - air)')
            ax.scatter([], [], c='#4169E1', s=30, alpha=0.7, edgecolors='black', 
                      linewidth=0.5, label='Ball (raw - ground)')
        
        # Ball trajectory - fitted (thick colored line)
        fitted_col_x = None
        fitted_col_y = None
        fitted_col_z = None
        fitted_method = None
        
        # Detect which fitting method was used
        if 'polynomial_x' in self.ball_3d_fitted_df.columns:
            fitted_col_x, fitted_col_y, fitted_col_z = 'polynomial_x', 'polynomial_y', 'polynomial_z'
            fitted_method = 'Polynomial'
        elif 'bezier_x' in self.ball_3d_fitted_df.columns:
            fitted_col_x, fitted_col_y, fitted_col_z = 'bezier_x', 'bezier_y', 'bezier_z'
            fitted_method = 'Bezier'
        elif 'exponential_x' in self.ball_3d_fitted_df.columns:
            fitted_col_x, fitted_col_y, fitted_col_z = 'exponential_x', 'exponential_y', 'exponential_z'
            fitted_method = 'Exponential'
        elif 'mu_s_x' in self.ball_3d_fitted_df.columns:
            fitted_col_x, fitted_col_y, fitted_col_z = 'mu_s_x', 'mu_s_y', 'mu_s_z'
            fitted_method = 'Physics (μs)'
        elif 'ball_x' in self.ball_3d_fitted_df.columns:
            fitted_col_x, fitted_col_y, fitted_col_z = 'ball_x', 'ball_y', 'ball_z'
            fitted_method = 'Raw'
        
        if fitted_col_x is not None:
            ball_fitted = self.ball_3d_fitted_df.dropna(subset=[fitted_col_x, fitted_col_y, fitted_col_z])
            if len(ball_fitted) > 0:
                ax.plot(ball_fitted[fitted_col_x], ball_fitted[fitted_col_y], -ball_fitted[fitted_col_z],
                       'r-', linewidth=3.5, label=f'Ball (fitted: {fitted_method})', alpha=0.9)
                
                # Add trajectory start/end markers
                ax.scatter(ball_fitted[fitted_col_x].iloc[0], ball_fitted[fitted_col_y].iloc[0], 
                          -ball_fitted[fitted_col_z].iloc[0], c='green', s=150, marker='o', 
                          edgecolors='darkgreen', linewidth=1.5, alpha=0.9, zorder=5, label='Start')
                ax.scatter(ball_fitted[fitted_col_x].iloc[-1], ball_fitted[fitted_col_y].iloc[-1], 
                          -ball_fitted[fitted_col_z].iloc[-1], c='red', s=150, marker='s', 
                          edgecolors='darkred', linewidth=1.5, alpha=0.9, zorder=5, label='End')
        
        # Camera position - raw (gray scatter)
        cam_raw = self.calibration_raw_df.dropna(subset=['cam_x', 'cam_y', 'cam_z'])
        if len(cam_raw) > 0:
            ax.scatter(cam_raw['cam_x'], cam_raw['cam_y'], -cam_raw['cam_z'],
                      c='#808080', alpha=0.4, s=40, marker='^', edgecolors='black', 
                      linewidth=0.5, label='Camera (raw)')
        
        # Camera position - smoothed (thick green line)
        cam_smooth = self.calibration_df.dropna(subset=['cam_x', 'cam_y', 'cam_z'])
        if len(cam_smooth) > 0:
            ax.plot(cam_smooth['cam_x'], cam_smooth['cam_y'], -cam_smooth['cam_z'],
                   'g-', linewidth=3, marker='o', markersize=4, label='Camera (smoothed)', alpha=0.8)
        
        # Field boundaries (soccer field outline in light gray)
        field_x = [-52.5, 52.5, 52.5, -52.5, -52.5]
        field_y = [-34, -34, 34, 34, -34]
        field_z = [0, 0, 0, 0, 0]
        ax.plot(field_x, field_y, field_z, color='#CCCCCC', linewidth=2, alpha=0.6, linestyle='--')
        
        # Field center line
        ax.plot([0, 0], [-34, 34], [0, 0], color='#DDDDDD', linewidth=1, alpha=0.4, linestyle=':')
        
        # Set labels and title with better formatting
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title('3D Ball and Camera Trajectories', fontsize=18, fontweight='bold', pad=20)
        
        # Set view angle
        ax.view_init(elev=25, azim=45)
        
        # Improve grid and axis appearance
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Set pane colors to white with transparency
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        
        # Legend with better placement and styling
        legend = ax.legend(fontsize=11, loc='upper left', framealpha=0.95, 
                          edgecolor='black', fancybox=True, shadow=True)
        legend.get_frame().set_linewidth(1.5)
        
        # Add text box with statistics
        n_ball_raw = len(ball_raw)
        n_ball_fitted = len(ball_fitted) if fitted_col_x is not None else 0
        n_cam = len(cam_smooth)
        
        stats_text = f'Ball Detections: {n_ball_raw}\nBall Fitted Points: {n_ball_fitted}\nCamera Frames: {n_cam}'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot with high quality
        plot_path = self.output_dir / "trajectories_3d.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ 3D plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Finalization System")
    parser.add_argument("--frames", type=str, required=True,
                       help="Directory containing original video frames")
    parser.add_argument("--detection-csv", type=str, required=True,
                       help="Path to smoothed detection CSV")
    parser.add_argument("--calibration-csv", type=str, required=True,
                       help="Path to smoothed calibration CSV")
    parser.add_argument("--calibration-raw-csv", type=str, required=True,
                       help="Path to raw calibration CSV")
    parser.add_argument("--ball-3d-csv", type=str, required=True,
                       help="Path to raw 3D ball positions CSV")
    parser.add_argument("--ball-3d-fitted-csv", type=str, required=True,
                       help="Path to fitted 3D ball positions CSV")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for finalization results")
    parser.add_argument("--weights-kp", type=str, default=None,
                       help="Path to PnLCalib keypoint weights (optional)")
    parser.add_argument("--weights-line", type=str, default=None,
                       help="Path to PnLCalib line weights (optional)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("STAGE 5: FINALIZATION")
    print("="*60)
    
    # Initialize finalizer
    finalizer = Finalizer(
        frames_dir=args.frames,
        detection_csv=args.detection_csv,
        calibration_csv=args.calibration_csv,
        calibration_raw_csv=args.calibration_raw_csv,
        ball_3d_csv=args.ball_3d_csv,
        ball_3d_fitted_csv=args.ball_3d_fitted_csv,
        output_dir=args.output,
        weights_kp=args.weights_kp,
        weights_line=args.weights_line
    )
    
    # Generate outputs
    finalizer.generate_video()
    finalizer.create_final_csv()
    finalizer.plot_3d_trajectories()
    
    print("\n" + "="*60)
    print("✓ STAGE 5 COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output}")
    print(f"  - final_video.mp4")
    print(f"  - final_positions.csv")
    print(f"  - trajectories_3d.png")
    print(f"  - annotated_frames/")


if __name__ == "__main__":
    main()

