#!/usr/bin/env python3
"""
Simple Ball Detection System
Four modes:
1. detect - Run YOLO + Hough Circle detection on frames
2. correct - Manually correct bounding boxes and re-detect circles
3. smooth - Smooth diameter with single peak constraint
4. visualize - Generate 4-subplot visualization of ball detection
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import sys
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt


class BallDetector:
    """Simple ball detector using YOLO + Hough Circle"""
    
    def __init__(self, yolo_model_path, output_dir):
        self.yolo_model_path = yolo_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir = self.output_dir / "annotated_frames"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model
        print(f"Loading YOLO model from {yolo_model_path}...")
        self.model = YOLO(yolo_model_path)
        print("Model loaded successfully!")
        
        # Detection results
        self.detections = []
        
    @staticmethod
    def detect_hough_circle(image_crop):
        """
        Detect ball in cropped image using Hough Circle Transform
        Returns: (center_x, center_y, diameter) or (None, None, None)
        diameter is the width of the detected circle/ball
        """
        if image_crop.size == 0:
            return None, None, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take the first (strongest) circle
            x, y, r = circles[0, 0]
            return int(x), int(y), int(2 * r)  # Return diameter
        
        return None, None, None
    
    @staticmethod
    def detect_all_hough_circles(image_crop):
        """
        Detect ALL possible circles in cropped image
        Returns: list of (center_x, center_y, diameter) tuples
        Used in correction mode to allow switching between circle candidates
        """
        if image_crop.size == 0 or image_crop.shape[0] < 10 or image_crop.shape[1] < 10:
            return []
        
        h, w = image_crop.shape[:2]
        max_radius = min(h, w) // 2
        min_radius = 2
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        circles_list = []
        
        try:
            # Apply bilateral filter and adaptive thresholding
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                if radius < min_radius or radius > max_radius:
                    continue
                
                if circularity > 0.35:  # Lowered threshold
                    circles_list.append((int(cx), int(cy), int(radius * 2), circularity * area))
        
        except Exception as e:
            pass
        
        # Try Hough Circles with multiple parameter sets
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
            
            hough_params = [
                {'param1': 50, 'param2': 20, 'minRadius': max(2, max_radius//4), 'maxRadius': max_radius},
                {'param1': 50, 'param2': 18, 'minRadius': max(2, max_radius//5), 'maxRadius': max_radius},
                {'param1': 100, 'param2': 25, 'minRadius': max(2, max_radius//6), 'maxRadius': max_radius},
                {'param1': 30, 'param2': 15, 'minRadius': max(2, max_radius//8), 'maxRadius': max_radius},
                {'param1': 40, 'param2': 22, 'minRadius': max(2, max_radius//5), 'maxRadius': max_radius},
            ]
            
            for params in hough_params:
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=max(h, w) // 6,
                    **params
                )
                
                if circles is not None and len(circles[0]) > 0:
                    circles = np.uint16(np.around(circles))
                    # Take multiple circles from each parameter set
                    for x, y, radius in circles[0][:3]:  # Take up to 3 circles per param set
                        diameter = radius * 2
                        circles_list.append((int(x), int(y), int(diameter), 100))
        
        except Exception as e:
            pass
        
        # Try Canny edges with multiple thresholds
        try:
            canny_params = [(20, 80), (30, 100), (40, 120), (50, 150)]
            
            for low, high in canny_params:
                edges = cv2.Canny(gray, low, high)
                
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if len(contour) >= 5:
                        area = cv2.contourArea(contour)
                        if area < 10:
                            continue
                        
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        
                        if radius < min_radius or radius > max_radius:
                            continue
                        
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter == 0:
                            continue
                        
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity > 0.35:
                            circles_list.append((int(cx), int(cy), int(radius * 2), circularity * area))
        
        except Exception as e:
            pass
        
        # Sort by score (descending) and remove duplicates
        circles_list.sort(key=lambda c: c[3], reverse=True)
        
        # Remove duplicates (circles too close to each other)
        unique_circles = []
        for cx, cy, d, score in circles_list:
            is_duplicate = False
            for ucx, ucy, ud, _ in unique_circles:
                dist = np.sqrt((cx - ucx)**2 + (cy - ucy)**2)
                if dist < 6 and abs(d - ud) < 4:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append((cx, cy, d, score))
        
        return [(c[0], c[1], c[2]) for c in unique_circles]
    
    def process_frames(self, frames_dir):
        """
        Process all frames in directory with YOLO + Hough Circle
        """
        frames_path = Path(frames_dir)
        
        # Get all image files sorted by name
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if not image_files:
            print(f"No image files found in {frames_dir}")
            return False
        
        print(f"Processing {len(image_files)} frames...")
        
        self.detections = []
        
        for idx, img_path in enumerate(image_files):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"\nWarning: Could not read {img_path}")
                # Add null entry for unreadable frame
                self.detections.append({
                    'frame': idx,
                    'x': None,
                    'y': None,
                    'w': None,
                    'h': None,
                    'center_x': None,
                    'center_y': None,
                    'diameter': None,
                    'air_ground': 'unknown'
                })
                continue
            
            # Run YOLO detection
            results = self.model(image, verbose=False)
            
            # Extract bounding box (assume first detection is the ball)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                # Crop to bbox
                crop = image[y1:y2, x1:x2]
                
                # Detect circle in crop
                cx_crop, cy_crop, diameter = self.detect_hough_circle(crop)
                
                # Convert to full image coordinates
                if cx_crop is not None:
                    center_x = x1 + cx_crop
                    center_y = y1 + cy_crop
                else:
                    # Fallback to bbox center
                    center_x = x1 + w // 2
                    center_y = y1 + h // 2
                    diameter = min(w, h)
                
                # Store detection (air_ground will be set later)
                self.detections.append({
                    'frame': idx,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'center_x': center_x,
                    'center_y': center_y,
                    'diameter': diameter if diameter is not None else min(w, h),
                    'air_ground': 'air'  # Default, will be updated
                })
            else:
                print(f"\nWarning: No ball detected in frame {idx}")
                # Add null entry for undetected frame
                self.detections.append({
                    'frame': idx,
                    'x': None,
                    'y': None,
                    'w': None,
                    'h': None,
                    'center_x': None,
                    'center_y': None,
                    'diameter': None,
                    'air_ground': 'unknown'
                })
        
        detected_count = sum(1 for d in self.detections if d['x'] is not None)
        undetected_count = len(self.detections) - detected_count
        print(f"Detection complete! Total frames: {len(self.detections)}, "
              f"with ball: {detected_count}, without ball: {undetected_count}")
        return True
    
    def set_air_ground_labels(self, airborne_frame):
        """
        Set air/ground labels based on airborne frame number
        Frames 0 to airborne_frame-1 are 'ground', rest are 'air'
        """
        for detection in self.detections:
            if detection['frame'] < airborne_frame:
                detection['air_ground'] = 'ground'
            else:
                detection['air_ground'] = 'air'
        
        print(f"Labels set: frames 0-{airborne_frame-1} = GROUND, {airborne_frame}+ = AIR")
    
    def save_detections(self):
        """Save detections to CSV"""
        csv_path = self.output_dir / "detections.csv"
        df = pd.DataFrame(self.detections)
        df.to_csv(csv_path, index=False)
        print(f"Detections saved to {csv_path}")
        return csv_path
    
    def generate_annotated_images(self, frames_dir):
        """
        Generate annotated images with bbox, circle, center, and labels
        For frames with detections: draws bbox, circle, and labels
        For frames with NO detections: saves frame with "NO DETECTION" label
        Note: Converts float values to int for cv2 rendering
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        annotated_count = 0
        no_detection_count = 0
        error_count = 0
        
        for detection in self.detections:
            frame_idx = detection['frame']
            
            if frame_idx >= len(image_files):
                error_count += 1
                continue
            
            img_path = image_files[frame_idx]
            image = cv2.imread(str(img_path))
            if image is None:
                error_count += 1
                continue
            
            # Check if this frame has a detection or is a "no detection" frame
            if detection['x'] is None or detection['center_x'] is None:
                # No detection found in this frame - draw "NO DETECTION" label
                cv2.putText(image, "NO DETECTION", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                cv2.putText(image, f"Frame {frame_idx} ({detection['air_ground'].upper()})", 
                           (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                no_detection_count += 1
            else:
                # Detection found - draw bbox, circle, and labels
                # Convert float values to int for cv2 (from CSV they may be float)
                x = int(detection['x'])
                y = int(detection['y'])
                w = int(detection['w'])
                h = int(detection['h'])
                cx = int(detection['center_x'])
                cy = int(detection['center_y'])
                diameter = float(detection['diameter'])  # Keep as float for display
                radius = int(diameter / 2)  # Convert to int for cv2
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw circle
                cv2.circle(image, (cx, cy), radius, (255, 0, 0), 2)
                
                # Draw center dot
                cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
                
                # Draw label (show float diameter value)
                label = f"{detection['air_ground'].upper()} {diameter:.1f}px"
                label_y = max(y - 10, 20)
                cv2.putText(image, label, (x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                annotated_count += 1
            
            # Save annotated image (for both detection and no-detection frames)
            output_path = self.annotated_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), image)
        
        print(f"Annotated images saved to {self.annotated_dir}")
        print(f"  Frames with detections: {annotated_count}")
        print(f"  Frames with no detections: {no_detection_count}")
        print(f"  Frames with errors: {error_count}")


class BallCorrector:
    """Interactive bbox correction tool with circle switching"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "detections.csv"
        self.annotated_dir = self.results_dir / "annotated_frames"
        
        # Load existing detections
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No detections found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        self.detections = self.df.to_dict('records')
        
        # UI state
        self.current_frame_idx = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Circle switching state
        self.current_circles = []  # Available circles for current frame
        self.current_circle_idx = 0  # Index of currently selected circle
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing bounding box"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def correct_bbox(self, frame_idx, frames_dir, new_bbox):
        """
        Correct bbox for a frame and re-run Hough Circle detection
        new_bbox: (x1, y1, x2, y2)
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if frame_idx >= len(image_files):
            return False
        
        img_path = image_files[frame_idx]
        image = cv2.imread(str(img_path))
        if image is None:
            return False
        
        # Extract new bbox
        x1, y1, x2, y2 = new_bbox
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        
        # Crop and detect circle
        crop = image[y:y+h, x:x+w]
        cx_crop, cy_crop, diameter = BallDetector.detect_hough_circle(crop)
        
        # Update detection
        if cx_crop is not None:
            center_x = x + cx_crop
            center_y = y + cy_crop
        else:
            center_x = x + w // 2
            center_y = y + h // 2
            diameter = min(w, h)
        
        # Find and update detection
        for detection in self.detections:
            if detection['frame'] == frame_idx:
                detection['x'] = x
                detection['y'] = y
                detection['w'] = w
                detection['h'] = h
                detection['center_x'] = center_x
                detection['center_y'] = center_y
                detection['diameter'] = diameter if diameter is not None else min(w, h)
                break
        
        return True
    
    def detect_circles_in_bbox(self, frames_dir, frame_idx):
        """
        Detect all possible circles in the current bbox
        Sets self.current_circles and resets circle index to 0
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if frame_idx >= len(image_files):
            self.current_circles = []
            self.current_circle_idx = 0
            return
        
        # Get current detection bbox
        current_detection = None
        for det in self.detections:
            if det['frame'] == frame_idx:
                current_detection = det
                break
        
        if current_detection is None:
            self.current_circles = []
            self.current_circle_idx = 0
            return
        
        # Load image and crop
        img_path = image_files[frame_idx]
        image = cv2.imread(str(img_path))
        if image is None:
            self.current_circles = []
            self.current_circle_idx = 0
            return
        
        x, y, w, h = current_detection['x'], current_detection['y'], \
                     current_detection['w'], current_detection['h']
        
        # Check if detection has valid bbox (not None or NaN for frames without detection)
        import math
        if x is None or y is None or w is None or h is None:
            self.current_circles = []
            self.current_circle_idx = 0
            print(f"Skipping frame {frame_idx}: no valid detection bbox (None)")
            return
        
        # Also check for NaN values
        try:
            if math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h):
                self.current_circles = []
                self.current_circle_idx = 0
                print(f"Skipping frame {frame_idx}: no valid detection bbox (NaN)")
                return
        except (TypeError, ValueError):
            pass
        
        # Convert to integers for slicing
        x, y, w, h = int(x), int(y), int(w), int(h)
        crop = image[y:y+h, x:x+w]
        
        # Detect all circles
        circles = BallDetector.detect_all_hough_circles(crop)
        
        # Store circles with their offset coordinates
        self.current_circles = []
        for cx, cy, diameter in circles:
            # Convert to full image coordinates
            full_cx = x + cx
            full_cy = y + cy
            self.current_circles.append({
                'cx': full_cx,
                'cy': full_cy,
                'diameter': diameter
            })
        
        self.current_circle_idx = 0
    
    def switch_to_next_circle(self):
        """Switch to the next circle option"""
        if len(self.current_circles) > 0:
            self.current_circle_idx = (self.current_circle_idx + 1) % len(self.current_circles)
            return True
        return False
    
    def apply_current_circle(self, frame_idx):
        """Apply the currently selected circle to the detection"""
        if len(self.current_circles) == 0:
            return False
        
        circle = self.current_circles[self.current_circle_idx]
        
        # Update detection
        for detection in self.detections:
            if detection['frame'] == frame_idx:
                detection['center_x'] = circle['cx']
                detection['center_y'] = circle['cy']
                detection['diameter'] = circle['diameter']
                break
        
        return True
    
    def run_correction_session(self, frames_dir):
        """
        Interactive correction session
        Keys:
        - 'n': next frame
        - 'p': previous frame
        - 'f': fix/correct current frame
        - 's': switch to next circle option
        - 'q': quit and save
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if not image_files:
            print(f"No image files found in {frames_dir}")
            return False
        
        window_name = "Ball Correction (n=next, p=prev, f=fix, s=switch circle, q=quit)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        correction_mode = False
        corrections_made = 0
        
        while True:
            # Get current detection
            current_detection = None
            for det in self.detections:
                if det['frame'] == self.current_frame_idx:
                    current_detection = det
                    break
            
            if current_detection is None:
                print(f"No detection for frame {self.current_frame_idx}")
                self.current_frame_idx = (self.current_frame_idx + 1) % len(self.detections)
                continue
            
            # Detect circles for the current frame bbox (if not already done)
            if len(self.current_circles) == 0 and not correction_mode:
                self.detect_circles_in_bbox(frames_dir, self.current_frame_idx)
            
            # Load image
            img_path = image_files[current_detection['frame']]
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load {img_path}")
                break
            
            # Create display image
            display = image.copy()
            
            if correction_mode:
                # Draw current bbox being drawn
                if self.start_point and self.end_point:
                    cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 255), 2)
                    cv2.putText(display, "Draw new bbox (release to apply)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Draw current detection (skip if no detection)
                x, y, w, h = current_detection['x'], current_detection['y'], \
                            current_detection['w'], current_detection['h']
                
                # Check if detection has valid bbox (not None or NaN for frames without detection)
                import math
                is_valid = True
                if x is None or y is None or w is None or h is None:
                    is_valid = False
                else:
                    try:
                        if math.isnan(float(x)) or math.isnan(float(y)) or math.isnan(float(w)) or math.isnan(float(h)):
                            is_valid = False
                    except (TypeError, ValueError):
                        pass
                
                if is_valid:
                    # Convert to integers for drawing
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw current circle
                    cx, cy = current_detection['center_x'], current_detection['center_y']
                    if cx is not None and cy is not None:
                        try:
                            if not (math.isnan(float(cx)) or math.isnan(float(cy))):
                                cx, cy = int(cx), int(cy)
                                diameter = current_detection['diameter']
                                if diameter is not None and not math.isnan(float(diameter)):
                                    radius = int(diameter) // 2
                                    cv2.circle(display, (cx, cy), radius, (255, 0, 0), 2)
                                    cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
                        except (TypeError, ValueError):
                            pass
                    
                    # Draw other available circles in cyan
                    if len(self.current_circles) > 1:
                        for idx, circle in enumerate(self.current_circles):
                            if idx != self.current_circle_idx:
                                r = circle['diameter'] // 2
                                cv2.circle(display, (circle['cx'], circle['cy']), r, (255, 255, 0), 1)
                    
                    # Draw label
                    diameter = current_detection['diameter']
                    if diameter is not None and not math.isnan(float(diameter)):
                        label = f"{current_detection['air_ground'].upper()} {int(diameter)}px"
                        cv2.putText(display, label, (x, max(y - 10, 20)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # No detection - draw "NO DETECTION" label
                    cv2.putText(display, "NO DETECTION", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                    cv2.putText(display, f"Frame {self.current_frame_idx} ({current_detection['air_ground'].upper()})",
                               (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                
                # Frame info with circle count
                info = f"Frame {self.current_frame_idx}/{len(self.detections)-1} | Press 'f' to fix | Circles: {self.current_circle_idx+1}/{len(self.current_circles)}"
                cv2.putText(display, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n') and not correction_mode:
                self.current_frame_idx = (self.current_frame_idx + 1) % len(self.detections)
                self.current_circles = []  # Reset circles for new frame
                self.current_circle_idx = 0
            elif key == ord('p') and not correction_mode:
                self.current_frame_idx = (self.current_frame_idx - 1) % len(self.detections)
                self.current_circles = []  # Reset circles for new frame
                self.current_circle_idx = 0
            elif key == ord('s') and not correction_mode:
                # Switch to next circle
                if self.switch_to_next_circle():
                    self.apply_current_circle(self.current_frame_idx)
                    print(f"Switched to circle {self.current_circle_idx + 1}/{len(self.current_circles)}")
            elif key == ord('f') and not correction_mode:
                correction_mode = True
                self.start_point = None
                self.end_point = None
                print(f"Correction mode: Draw new bbox for frame {self.current_frame_idx}")
            
            # Check if bbox drawing is complete
            if correction_mode and not self.drawing and self.start_point and self.end_point:
                # Apply correction
                new_bbox = (self.start_point[0], self.start_point[1],
                           self.end_point[0], self.end_point[1])
                
                if self.correct_bbox(self.current_frame_idx, frames_dir, new_bbox):
                    print(f"Corrected frame {self.current_frame_idx}")
                    corrections_made += 1
                    self.current_circles = []  # Reset circles after bbox correction
                    self.current_circle_idx = 0
                else:
                    print(f"Failed to correct frame {self.current_frame_idx}")
                
                correction_mode = False
                self.start_point = None
                self.end_point = None
        
        cv2.destroyAllWindows()
        
        if corrections_made > 0:
            print(f"\n{corrections_made} corrections made.")
            
            # Save updated detections
            df = pd.DataFrame(self.detections)
            df.to_csv(self.csv_path, index=False)
            print(f"Updated detections saved to {self.csv_path}")
            
            return True
        else:
            print("\nNo corrections made.")
            return False


class DiameterSmoother:
    """Smooth diameter with single peak constraint using piecewise quadratic or cubic spline"""
    
    def __init__(self, results_dir, calibration_csv=None):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "detections.csv"
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No detections found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Load focal length for zoom-aware smoothing
        self.focal_length = None
        if calibration_csv:
            self._load_focal_length(calibration_csv)
    
    def _load_focal_length(self, calibration_csv):
        """Load focal length from calibration CSV for zoom detection"""
        try:
            calib_df = pd.read_csv(calibration_csv)
            if 'fy' in calib_df.columns:
                # Match frames with detections
                frames = self.df['frame'].values
                self.focal_length = np.zeros(len(frames))
                
                for i, frame in enumerate(frames):
                    if frame < len(calib_df):
                        self.focal_length[i] = calib_df.loc[frame, 'fy']
                    else:
                        self.focal_length[i] = calib_df['fy'].iloc[-1]
                
                print(f"  ✓ Loaded focal length data (range: [{self.focal_length.min():.1f}, {self.focal_length.max():.1f}])")
            else:
                print("  ⚠ No 'fy' column in calibration CSV")
        except Exception as e:
            print(f"  ⚠ Could not load focal length: {e}")
    
    def quadratic(self, x, a, b, c):
        """Quadratic function: a*x^2 + b*x + c"""
        return a * x**2 + b * x + c
    
    def method_9_zoom_aware_piecewise_quadratic(self, peak_frame):
        """
        Method 9: Zoom-aware piecewise quadratic (2-layer: base + zoom)
        
        Args:
            peak_frame: Frame number where diameter is at physical peak (no zoom)
        
        Returns:
            Smoothed diameter array
        """
        frames = self.df['frame'].values
        diameter = self.df['diameter'].values.astype(float)
        
        # Find peak frame index
        peak_idx = np.where(frames == peak_frame)[0]
        if len(peak_idx) == 0:
            raise ValueError(f"Peak frame {peak_frame} not found in detections")
        peak_idx = peak_idx[0]
        
        # Layer 1: Remove zoom effect (if available)
        if self.focal_length is not None:
            focal_ratio = self.focal_length / self.focal_length[peak_idx]
            diameter_no_zoom = diameter / focal_ratio
        else:
            diameter_no_zoom = diameter.copy()
        
        # Fit piecewise quadratics
        base_curve = np.zeros_like(diameter_no_zoom)
        
        # Left side (before and including peak)
        if peak_idx > 2:
            frames_left = frames[:peak_idx+1]
            diameter_left = diameter_no_zoom[:peak_idx+1]
            
            try:
                popt_left, _ = curve_fit(self.quadratic, frames_left, diameter_left, maxfev=10000)
                base_curve[:peak_idx+1] = self.quadratic(frames_left, *popt_left)
            except (RuntimeError, ValueError) as e:
                # Curve fitting failed, use raw data
                base_curve[:peak_idx+1] = diameter_left
        else:
            base_curve[:peak_idx+1] = diameter_no_zoom[:peak_idx+1]
        
        # Right side (after and including peak)
        if peak_idx < len(frames) - 3:
            frames_right = frames[peak_idx:]
            diameter_right = diameter_no_zoom[peak_idx:]
            
            try:
                popt_right, _ = curve_fit(self.quadratic, frames_right, diameter_right, maxfev=10000)
                base_curve[peak_idx:] = self.quadratic(frames_right, *popt_right)
            except (RuntimeError, ValueError) as e:
                # Curve fitting failed, use raw data
                base_curve[peak_idx:] = diameter_right
        else:
            base_curve[peak_idx:] = diameter_no_zoom[peak_idx:]
        
        # Apply light Gaussian smoothing
        base_curve = gaussian_filter1d(base_curve, sigma=1.5)
        
        # Enforce strict monotonicity
        for i in range(1, peak_idx+1):
            if base_curve[i] < base_curve[i-1]:
                base_curve[i] = base_curve[i-1]
        
        for i in range(peak_idx+1, len(base_curve)):
            if base_curve[i] > base_curve[i-1]:
                base_curve[i] = base_curve[i-1]
        
        # Layer 2: Apply zoom back (if available)
        if self.focal_length is not None:
            smoothed = base_curve * focal_ratio
            smoothed = gaussian_filter1d(smoothed, sigma=0.5)
        else:
            smoothed = base_curve
        
        return smoothed
    
    def method_13_zoom_aware_piecewise_spline(self, peak_frame):
        """
        Method 13: Zoom-aware piecewise cubic spline (2-layer: 2 splines + zoom)
        
        Args:
            peak_frame: Frame number where diameter is at physical peak (no zoom)
        
        Returns:
            Smoothed diameter array
        """
        frames = self.df['frame'].values
        diameter = self.df['diameter'].values.astype(float)
        
        # Find peak frame index
        peak_idx = np.where(frames == peak_frame)[0]
        if len(peak_idx) == 0:
            raise ValueError(f"Peak frame {peak_frame} not found in detections")
        peak_idx = peak_idx[0]
        
        # Layer 1: Remove zoom effect (if available)
        if self.focal_length is not None:
            focal_ratio = self.focal_length / self.focal_length[peak_idx]
            diameter_no_zoom = diameter / focal_ratio
        else:
            diameter_no_zoom = diameter.copy()
        
        # Fit TWO separate cubic splines (left and right of peak)
        base_curve = np.zeros_like(diameter_no_zoom)
        
        # Left spline (before peak, including peak)
        if peak_idx > 0:
            frames_left = frames[:peak_idx+1]
            diameter_left = diameter_no_zoom[:peak_idx+1]
            s_left = len(frames_left) * 8  # Heavy smoothing
            spline_left = UnivariateSpline(frames_left, diameter_left, s=s_left, k=min(3, len(frames_left)-1))
            base_curve[:peak_idx+1] = spline_left(frames_left)
        
        # Right spline (after peak, including peak)
        if peak_idx < len(frames) - 1:
            frames_right = frames[peak_idx:]
            diameter_right = diameter_no_zoom[peak_idx:]
            s_right = len(frames_right) * 8  # Heavy smoothing
            spline_right = UnivariateSpline(frames_right, diameter_right, s=s_right, k=min(3, len(frames_right)-1))
            base_curve[peak_idx:] = spline_right(frames_right)
        
        # Smooth the transition at peak
        base_curve = gaussian_filter1d(base_curve, sigma=1.0)
        
        # Layer 2: Apply zoom back (if available)
        if self.focal_length is not None:
            smoothed = base_curve * focal_ratio
            smoothed = gaussian_filter1d(smoothed, sigma=0.5)
        else:
            smoothed = base_curve
        
        return smoothed
    
    def method_15_zoom_exponential_bidirectional(self, peak_frame):
        """
        Method 15: Zoom-aware bidirectional exponential (3-layer)
        
        Args:
            peak_frame: Frame number where diameter is at physical peak (no zoom)
        
        Returns:
            Smoothed diameter array
        """
        frames = self.df['frame'].values
        diameter = self.df['diameter'].values.astype(float)
        
        # Find peak frame index
        peak_idx = np.where(frames == peak_frame)[0]
        if len(peak_idx) == 0:
            raise ValueError(f"Peak frame {peak_frame} not found in detections")
        peak_idx = peak_idx[0]
        
        # Layer 1: Remove zoom effect (if available)
        if self.focal_length is not None:
            focal_ratio = self.focal_length / self.focal_length[peak_idx]
            diameter_no_zoom = diameter / focal_ratio
        else:
            diameter_no_zoom = diameter.copy()
        
        # Layer 2: Bidirectional exponential smoothing
        alpha = 0.08  # Very smooth
        forward = np.zeros_like(diameter_no_zoom)
        forward[0] = diameter_no_zoom[0]
        for i in range(1, len(diameter_no_zoom)):
            forward[i] = alpha * diameter_no_zoom[i] + (1 - alpha) * forward[i-1]
        
        backward = np.zeros_like(forward)
        backward[-1] = forward[-1]
        for i in range(len(forward) - 2, -1, -1):
            backward[i] = alpha * forward[i] + (1 - alpha) * backward[i+1]
        
        base_curve = (forward + backward) / 2.0
        
        # Layer 3: Apply zoom back (if available)
        if self.focal_length is not None:
            smoothed = base_curve * focal_ratio
            smoothed = gaussian_filter1d(smoothed, sigma=0.5)
        else:
            smoothed = base_curve
        
        return smoothed
    
    def smooth_center_positions(self, method='method1'):
        """
        Smooth center_x and center_y positions using Savitzky-Golay filter
        
        Args:
            method: 'method1', 'method2', or 'method3' (determines window size)
        
        Returns:
            (smoothed_x, smoothed_y) arrays
        """
        center_x = self.df['center_x'].values.astype(float)
        center_y = self.df['center_y'].values.astype(float)
        
        # Use appropriate window size based on method
        if method == 'method1':
            window = 11
            polyorder = 2
        elif method == 'method2':
            window = 21
            polyorder = 3
        else:  # method3
            window = 31
            polyorder = 3
        
        # Ensure window is odd and not larger than data
        window = min(window, len(center_x) if len(center_x) % 2 == 1 else len(center_x) - 1)
        if window % 2 == 0:
            window -= 1
        window = max(window, 5)  # Minimum window of 5
        
        # Apply Savitzky-Golay filter
        try:
            from scipy.signal import savgol_filter
            smoothed_x = savgol_filter(center_x, window, polyorder)
            smoothed_y = savgol_filter(center_y, window, polyorder)
        except (ValueError, ImportError) as e:
            # Fallback to Gaussian smoothing if savgol fails
            smoothed_x = gaussian_filter1d(center_x, sigma=2.0)
            smoothed_y = gaussian_filter1d(center_y, sigma=2.0)
        
        return smoothed_x, smoothed_y
    
    def apply_smoothing(self, peak_frame, method='method1', smooth_type='diameter'):
        """
        Apply smoothing and save results
        
        Args:
            peak_frame: Frame number where diameter is at physical peak
            method: 'method1' (piecewise quadratic) or 'method2' (piecewise spline) or 'method3' (exponential)
            smooth_type: 'diameter', 'center', or 'both'
        """
        print(f"\nSmoothing {smooth_type} with peak at frame {peak_frame} using {method}...")
        
        # Store original values
        original_diameter = self.df['diameter'].values.copy()
        original_center_x = self.df['center_x'].values.copy()
        original_center_y = self.df['center_y'].values.copy()
        
        # Smooth diameter if requested
        if smooth_type in ['diameter', 'both']:
            if method == 'method1':
                print("  Using Method 1: Zoom-Aware Piecewise Quadratic")
                smoothed_diameter = self.method_9_zoom_aware_piecewise_quadratic(peak_frame)
            elif method == 'method2':
                print("  Using Method 2: Zoom-Aware Piecewise Cubic Spline")
                smoothed_diameter = self.method_13_zoom_aware_piecewise_spline(peak_frame)
            elif method == 'method3':
                print("  Using Method 3: Zoom-Aware Bidirectional Exponential")
                smoothed_diameter = self.method_15_zoom_exponential_bidirectional(peak_frame)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'method1', 'method2', or 'method3'")
            
            self.df['diameter'] = smoothed_diameter
        
        # Smooth center positions if requested
        if smooth_type in ['center', 'both']:
            print(f"  Smoothing center positions with method {method}...")
            smoothed_x, smoothed_y = self.smooth_center_positions(method)
            self.df['center_x'] = smoothed_x
            self.df['center_y'] = smoothed_y
        
        # Save updated detections with descriptive filename (always as _diameter)
        smoothed_csv_path = self.csv_path.parent / f"detections_smoothed_{method}_diameter.csv"
        self.df.to_csv(smoothed_csv_path, index=False, float_format='%.2f')
        print(f"✓ Saved smoothed detections to {smoothed_csv_path} (float precision)")
        print(f"  Original detections.csv NOT modified")
        
        # Print statistics
        if smooth_type in ['diameter', 'both']:
            print(f"Diameter Statistics:")
            print(f"  Original range: [{original_diameter.min():.2f}, {original_diameter.max():.2f}] pixels")
            print(f"  Smoothed range: [{self.df['diameter'].min():.2f}, {self.df['diameter'].max():.2f}] pixels")
            print(f"  Mean change: {np.abs(self.df['diameter'].values - original_diameter).mean():.2f} pixels")
        
        if smooth_type in ['center', 'both']:
            print(f"Center Position Statistics:")
            print(f"  X range: [{original_center_x.min():.2f}, {original_center_x.max():.2f}] → [{self.df['center_x'].min():.2f}, {self.df['center_x'].max():.2f}]")
            print(f"  Y range: [{original_center_y.min():.2f}, {original_center_y.max():.2f}] → [{self.df['center_y'].min():.2f}, {self.df['center_y'].max():.2f}]")
        
        return True


class BallVisualizer:
    """Visualize ball detection with 4-subplot plot"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "detections.csv"
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No detections found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
    
    def plot_detections(self, output_path=None):
        """
        Generate 4-subplot visualization:
        - Top left: All diameter trajectories (original + 3 smoothed methods)
        - Top right: X position vs Frame
        - Bottom left: Y position vs Frame  
        - Bottom right: X vs Y scatter (top-left origin, xlim=1920, ylim=1080)
        """
        if output_path is None:
            output_path = self.results_dir / "ball_detection_visualization.png"
        
        output_path = Path(output_path)
        
        # Extract data from original
        frames = self.df['frame'].values
        diameter = self.df['diameter'].values
        center_x = self.df['center_x'].values
        center_y = self.df['center_y'].values
        air_ground = self.df['air_ground'].values
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ball Detection Analysis: Diameter Trajectories & Positions', fontsize=18, fontweight='bold')
        
        # Color code by air/ground
        colors = ['red' if ag == 'air' else 'blue' for ag in air_ground]
        
        # Find all available smoothed files
        smoothed_files = {}
        for method in ['method1', 'method2', 'method3']:
            # Look for files with this method (any smooth_type variant)
            for smooth_type in ['diameter', 'center', 'both']:
                csv_file = self.results_dir / f"detections_smoothed_{method}_{smooth_type}.csv"
                if csv_file.exists():
                    # Store using method name as key (prefer 'diameter' if available)
                    if method not in smoothed_files or smooth_type == 'diameter':
                        smoothed_files[method] = csv_file
                    break
        
        # Plot 1: All diameter trajectories (top left)
        ax = axes[0, 0]
        ax.scatter(frames, diameter, c=colors, alpha=0.6, s=30, label='Original', zorder=2)
        ax.plot(frames, diameter, 'k-', alpha=0.7, linewidth=2, label='Original', zorder=1)
        
        # Overlay smoothed diameters
        colors_smooth = {'method1': 'green', 'method2': 'orange', 'method3': 'purple'}
        line_styles = {'method1': '--', 'method2': '-.', 'method3': ':'}
        
        for method in ['method1', 'method2', 'method3']:
            if method in smoothed_files:
                try:
                    smoothed_df = pd.read_csv(smoothed_files[method])
                    if 'diameter' in smoothed_df.columns and len(smoothed_df) == len(frames):
                        smoothed_diam = smoothed_df['diameter'].values
                        ax.plot(frames, smoothed_diam, line_styles[method], alpha=0.8, linewidth=2.0,
                               color=colors_smooth[method], label=f'{method}', zorder=3)
                except Exception as e:
                    pass
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Diameter (pixels)')
        ax.set_title('Ball Diameter: All Trajectories')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Plot 2: X position vs Frame (top right)
        ax = axes[0, 1]
        ax.scatter(frames, center_x, c=colors, alpha=0.6, s=30)
        ax.plot(frames, center_x, 'k-', alpha=0.7, linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('X Position (pixels)')
        ax.set_title('Ball X Position vs Frame')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Y position vs Frame (bottom left)
        ax = axes[1, 0]
        ax.scatter(frames, center_y, c=colors, alpha=0.6, s=30)
        ax.plot(frames, center_y, 'k-', alpha=0.7, linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Ball Y Position vs Frame')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: X vs Y scatter (bottom right, top-left origin)
        ax = axes[1, 1]
        scatter4 = ax.scatter(center_x, center_y, c=frames, cmap='viridis', alpha=0.6, s=30)
        ax.plot(center_x, center_y, 'k-', alpha=0.3, linewidth=1)
        
        # Set origin at top-left with proper axes
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)  # Invert Y to have top-left origin
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Ball Position in Frame (Top-Left Origin)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for frame number
        cbar = plt.colorbar(scatter4, ax=ax)
        cbar.set_label('Frame')
        
        # Add legend for air/ground
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.6, label='Air'),
            Patch(facecolor='blue', alpha=0.6, label='Ground')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved visualization to {output_path}")
        
        # Print statistics
        print(f"=== Ball Detection Statistics ===")
        print(f"Total frames: {len(frames)} | Air: {np.sum(air_ground == 'air')} | Ground: {np.sum(air_ground == 'ground')}")
        print(f"Diameter: {diameter.mean():.2f}±{diameter.std():.2f} px (range: [{diameter.min():.2f}, {diameter.max():.2f}])")
        print(f"Position X: {center_x.mean():.2f} px (range: [{center_x.min():.2f}, {center_x.max():.2f}])")
        print(f"Position Y: {center_y.mean():.2f} px (range: [{center_y.min():.2f}, {center_y.max():.2f}])")
        
        if smoothed_files:
            print(f"Smoothed methods available: {', '.join(smoothed_files.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Simple Ball Detection System")
    parser.add_argument("--mode", type=str, required=True, choices=['detect', 'correct', 'smooth', 'visualize'],
                       help="Mode: detect, correct, smooth, or visualize")
    parser.add_argument("--frames", type=str, default="downloads/frames/V2_1",
                       help="Input frames directory")
    parser.add_argument("--output", type=str, default="results/V2_1",
                       help="Output directory")
    parser.add_argument("--yolo", type=str, default="YOLO/output/yolo_training/weights/best.pt",
                       help="Path to YOLO model weights")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to camera calibration CSV (for zoom-aware smoothing)")
    parser.add_argument("--smooth-method", type=int, default=1, 
                       choices=[1, 2, 3],
                       help="Smoothing method: 1=Piecewise Quadratic, 2=Piecewise Cubic Spline, 3=Bidirectional Exponential")
    parser.add_argument("--smooth-type", type=str, default="diameter",
                       choices=['diameter', 'center', 'both'],
                       help="What to smooth: diameter, center positions, or both (default: diameter)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'detect':
            print("="*60)
            print("BALL DETECTION MODE")
            print("="*60)
            
            # Initialize detector
            detector = BallDetector(args.yolo, args.output)
            
            # Process frames
            if not detector.process_frames(args.frames):
                print("Detection failed!")
                sys.exit(1)
            
            # Ask for airborne frame
            print("\n" + "="*60)
            airborne_frame = input("Enter frame number where ball goes airborne (0 for first frame): ")
            try:
                airborne_frame = int(airborne_frame)
            except ValueError:
                print("Invalid input, using 0 (all frames in air)")
                airborne_frame = 0
            
            # Set labels
            detector.set_air_ground_labels(airborne_frame)
            
            # Save detections
            detector.save_detections()
            
            # Generate annotated images
            detector.generate_annotated_images(args.frames)
            
            print("\n" + "="*60)
            print("DETECTION COMPLETE!")
            print("="*60)
            print(f"Results saved to: {args.output}")
            print(f"CSV: {args.output}/detections.csv")
            print(f"Annotated frames: {args.output}/annotated_frames/")
            print("\nNext step: Run correction mode if needed")
            print(f"  python detect_ball.py --mode correct --frames {args.frames} --output {args.output}")
        
        elif args.mode == 'correct':
            print("="*60)
            print("BBOX CORRECTION MODE")
            print("="*60)
            
            # Initialize corrector
            corrector = BallCorrector(args.output)
            
            # Run correction session
            corrections_made = corrector.run_correction_session(args.frames)
            
            # Always regenerate annotated images (even if no changes made)
            print("\nRegenerating annotated images...")
            detector = BallDetector(args.yolo, args.output)
            detector.detections = corrector.detections
            detector.generate_annotated_images(args.frames)
            
            # Update air/ground labels if requested
            if corrections_made:
                print("\n" + "="*60)
                airborne_frame = input("Enter frame number where ball goes airborne (0 for first frame): ")
                try:
                    airborne_frame = int(airborne_frame)
                except ValueError:
                    print("Invalid input, keeping existing labels")
                    airborne_frame = None
                
                if airborne_frame is not None:
                    detector.set_air_ground_labels(airborne_frame)
                    detector.save_detections()
            
            print("\n" + "="*60)
            print("CORRECTION COMPLETE!")
            print("="*60)
            print(f"Updated results saved to: {args.output}")
            print(f"Annotated frames regenerated: {args.output}/annotated_frames/")
        
        elif args.mode == 'smooth':
            print("="*60)
            print("DIAMETER SMOOTHING MODE")
            print("="*60)
            
            # Map integer choice to method name
            method_map = {1: 'method1', 2: 'method2', 3: 'method3'}
            method_name = method_map[args.smooth_method]
            
            # Initialize smoother with optional calibration for zoom-aware smoothing
            smoother = DiameterSmoother(args.output, calibration_csv=args.calibration)
            
            # Show current diameter statistics
            print(f"\nCurrent diameter statistics:")
            print(f"  Frames: {smoother.df['frame'].min()} to {smoother.df['frame'].max()}")
            print(f"  Diameter range: [{smoother.df['diameter'].min():.2f}, {smoother.df['diameter'].max():.2f}] pixels")
            print(f"  Mean diameter: {smoother.df['diameter'].mean():.2f} ± {smoother.df['diameter'].std():.2f} pixels")
            
            # Show available methods
            print("\n" + "="*60)
            print("SMOOTHING METHODS:")
            print("  1 - Zoom-Aware Piecewise Quadratic (strict monotonicity)")
            print("  2 - Zoom-Aware Piecewise Cubic Spline (maximum smoothness)")
            print("  3 - Zoom-Aware Bidirectional Exponential (very smooth, no lag)")
            print(f"\nSelected method: {args.smooth_method} ({method_name})")
            print(f"Smooth type: {args.smooth_type}")
            
            if smoother.focal_length is not None:
                print("✓ Zoom-aware smoothing enabled (using focal length data)")
            else:
                print("⚠ No calibration data - smoothing without zoom compensation")
            
            # Ask for peak frame (only if smoothing diameter)
            if args.smooth_type in ['diameter', 'both']:
                print("\n" + "="*60)
                print("Enter the frame number where diameter is at its PHYSICAL peak.")
                print("(This is the peak if there was NO camera zoom - the true ball-camera distance peak)")
                peak_frame = input("Peak frame number: ")
                try:
                    peak_frame = int(peak_frame)
                except ValueError:
                    print("Invalid input! Using middle frame as default.")
                    peak_frame = int(smoother.df['frame'].median())
            else:
                peak_frame = int(smoother.df['frame'].median())  # Not used but required by function
            
            # Apply smoothing with selected method
            smoother.apply_smoothing(peak_frame, method=method_name, smooth_type=args.smooth_type)
            
            # Regenerate annotated images with smoothed diameter
            print("\nRegenerating annotated images with smoothed diameter...")
            detector = BallDetector(args.yolo, args.output)
            # Load the smoothed CSV instead of original
            smoothed_csv_path = Path(args.output) / f"detections_smoothed_{method_name}_diameter.csv"
            smoothed_df = pd.read_csv(smoothed_csv_path)
            detector.detections = smoothed_df.to_dict('records')
            detector.generate_annotated_images(args.frames)
            
            print("\n" + "="*60)
            print("SMOOTHING COMPLETE!")
            print("="*60)
            print(f"Updated results saved to: {args.output}")
            print(f"Original CSV: {args.output}/detections.csv (NOT modified)")
            print(f"Smoothed CSV: {args.output}/detections_smoothed_{method_name}_diameter.csv")
            print(f"Annotated frames: {args.output}/annotated_frames/ (int for display)")
            print(f"\nMethod used: {args.smooth_method} ({method_name})")
            print(f"Smooth type: {args.smooth_type}")
        
        elif args.mode == 'visualize':
            print("="*60)
            print("VISUALIZATION MODE")
            print("="*60)
            
            # Initialize visualizer
            visualizer = BallVisualizer(args.output)
            
            # Generate plots
            visualizer.plot_detections()
            
            print("\n" + "="*60)
            print("VISUALIZATION COMPLETE!")
            print("="*60)
            print(f"Plot saved to: {args.output}/ball_detection_visualization.png")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

