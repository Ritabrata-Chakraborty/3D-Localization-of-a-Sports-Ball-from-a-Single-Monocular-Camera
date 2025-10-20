#!/usr/bin/env python3
"""
Simple Ball Detection System
Two modes:
1. detect - Run YOLO + Hough Circle detection on frames
2. correct - Manually correct bounding boxes and re-detect circles
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import sys


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
        Detect ball in cropped image using advanced edge detection and circle fitting
        Returns: (center_x, center_y, diameter) or (None, None, None)
        diameter is the width of the detected circle/ball
        """
        if image_crop.size == 0 or image_crop.shape[0] < 10 or image_crop.shape[1] < 10:
            return None, None, None
        
        h, w = image_crop.shape[:2]
        max_radius = min(h, w) // 2
        min_radius = 2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # Try multiple preprocessing techniques to find the ball
        best_circle = None
        best_score = 0
        
        # Method 1: Adaptive thresholding + contour fitting
        try:
            # Apply bilateral filter to preserve edges while reducing noise
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:  # Lower threshold
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Get minimum enclosing circle
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                # Check if circle is reasonable
                if radius < min_radius or radius > max_radius:
                    continue
                
                # Score based on circularity and size
                # Prefer circular objects (circularity close to 1.0)
                score = circularity * area
                
                if circularity > 0.4 and score > best_score:  # Lowered threshold from 0.5
                    best_score = score
                    best_circle = (int(cx), int(cy), int(radius * 2))
        
        except Exception as e:
            pass
        
        # Method 2: Hough Circle Transform with adaptive parameters
        if best_circle is None:
            try:
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
                
                # Try multiple parameter sets with more flexible radius ranges
                param_sets = [
                    {'param1': 50, 'param2': 20, 'minRadius': max(2, max_radius//4), 'maxRadius': max_radius},
                    {'param1': 50, 'param2': 18, 'minRadius': max(2, max_radius//5), 'maxRadius': max_radius},
                    {'param1': 100, 'param2': 25, 'minRadius': max(2, max_radius//5), 'maxRadius': max_radius},
                    {'param1': 30, 'param2': 15, 'minRadius': max(2, max_radius//6), 'maxRadius': max_radius},
                ]
                
                for params in param_sets:
                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist=max(h, w) // 6,
                        **params
                    )
                    
                    if circles is not None and len(circles[0]) > 0:
                        circles = np.uint16(np.around(circles))
                        # Take the first (most confident) circle
                        x, y, radius = circles[0][0]
                        diameter = radius * 2
                        best_circle = (int(x), int(y), int(diameter))
                        break
            
            except Exception as e:
                pass
        
        # Method 3: Canny edges + contour circle fitting
        if best_circle is None:
            try:
                # Use multiple Canny thresholds
                canny_params = [(20, 80), (30, 100), (40, 120), (50, 150)]
                
                for low, high in canny_params:
                    edges = cv2.Canny(gray, low, high)
                    
                    # Dilate edges to connect broken edges
                    kernel = np.ones((2, 2), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                    
                    # Find contours from edges
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if len(contour) >= 5:  # Need at least 5 points
                            area = cv2.contourArea(contour)
                            if area < 10:
                                continue
                            
                            # Fit minimum enclosing circle
                            (cx, cy), radius = cv2.minEnclosingCircle(contour)
                            
                            if radius < min_radius or radius > max_radius:
                                continue
                            
                            # Calculate how well the contour fits a circle
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter == 0:
                                continue
                            
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            score = circularity * area
                            
                            if circularity > 0.35 and score > best_score:  # Lowered threshold
                                best_score = score
                                best_circle = (int(cx), int(cy), int(radius * 2))
            
            except Exception as e:
                pass
        
        # Method 4: Fallback - find largest blob
        if best_circle is None:
            try:
                # Simple threshold to find the brightest/darkest regions
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Get the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > 10:
                        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                        if radius >= min_radius and radius <= max_radius:
                            best_circle = (int(cx), int(cy), int(radius * 2))
            
            except Exception as e:
                pass
        
        if best_circle is not None:
            return best_circle
        
        return None, None, None
    
    @staticmethod
    def detect_all_hough_circles(image_crop):
        """
        Detect all possible circles in cropped image
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
        
        # ALWAYS add inscribed circle of bbox as first option
        bbox_center_x = w // 2
        bbox_center_y = h // 2
        bbox_radius = min(h, w) // 2
        circles_list.append((bbox_center_x, bbox_center_y, bbox_radius * 2, 1000))  # High score for bbox inscribed circle
        
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
                {'param1': 60, 'param2': 25, 'minRadius': max(2, max_radius//7), 'maxRadius': max_radius},
                {'param1': 80, 'param2': 28, 'minRadius': max(2, max_radius//6), 'maxRadius': max_radius},
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
                    # Take multiple circles from each parameter set, not just the first
                    for x, y, radius in circles[0][:5]:  # Take up to 5 circles per param set
                        diameter = radius * 2
                        circles_list.append((int(x), int(y), int(diameter), 100 + np.random.randint(0, 50)))
        
        except Exception as e:
            pass
        
        # Try Canny edges with multiple thresholds
        try:
            canny_params = [(20, 80), (30, 100), (40, 120), (50, 150), (25, 90), (35, 110)]
            
            for low, high in canny_params:
                edges = cv2.Canny(gray, low, high)
                
                # Dilate edges to connect broken edges
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Find contours from edges
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
                        
                        if circularity > 0.35:  # Lowered threshold
                            circles_list.append((int(cx), int(cy), int(radius * 2), circularity * area))
        
        except Exception as e:
            pass
        
        # Try OTSU thresholding for largest blob
        try:
            _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contours (not just one)
                contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours_by_area[:3]:  # Take top 3 largest
                    area = cv2.contourArea(contour)
                    if area > 10:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        if radius >= min_radius and radius <= max_radius:
                            circles_list.append((int(cx), int(cy), int(radius * 2), area))
        
        except Exception as e:
            pass
        
        # Add circles at different radii from bbox center (radial sweep)
        try:
            center_x, center_y = w // 2, h // 2
            radius_options = [
                max_radius,
                max_radius * 3 // 4,
                max_radius // 2,
                max_radius // 3,
                max_radius * 2 // 3,
                max_radius * 3 // 5,
            ]
            
            for radius in radius_options:
                if radius >= min_radius:
                    circles_list.append((center_x, center_y, radius * 2, 50))
        
        except Exception as e:
            pass
        
        # Try inverted threshold
        try:
            _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contours_by_area = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours_by_area[:3]:  # Take top 3 largest
                    area = cv2.contourArea(contour)
                    if area > 10:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        if radius >= min_radius and radius <= max_radius:
                            circles_list.append((int(cx), int(cy), int(radius * 2), area))
        
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
                # Allow smaller distance threshold for more variety
                if dist < 6 and abs(d - ud) < 4:  # Close position AND close diameter
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append((cx, cy, d, score))
        
        return [(c[0], c[1], c[2]) for c in unique_circles]
        
    def select_closest_detection(self, boxes, prev_center):
        """
        Select the detection closest to the previous detection center
        boxes: list of [x1, y1, x2, y2]
        prev_center: (x, y) of previous detection center
        Returns: index of closest box
        """
        if prev_center is None or len(boxes) == 0:
            return 0  # Return first box if no previous detection
        
        min_dist = float('inf')
        best_idx = 0
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx
    
    def process_frames(self, frames_dir):
        """
        Process all frames in directory with YOLO + Hough Circle
        Uses NMS and closest detection selection for multiple detections
        """
        frames_path = Path(frames_dir)
        
        # Get all image files sorted by name
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        if not image_files:
            print(f"No image files found in {frames_dir}")
            return False
        
        print(f"\nProcessing {len(image_files)} frames...")
        
        self.detections = []
        prev_center = None  # Track previous detection center
        
        for idx, img_path in enumerate(image_files):
            print(f"Processing frame {idx}/{len(image_files)}: {img_path.name}", end='\r')
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"\nWarning: Could not read {img_path}")
                continue
            
            # Run YOLO detection with built-in NMS (iou parameter controls NMS threshold)
            results = self.model(image, verbose=False, conf=0.02, iou=0.3)
            
            # Extract detections (already filtered by YOLO's NMS)
            if len(results[0].boxes) > 0:
                boxes = []
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                    boxes.append([x1, y1, x2, y2])
                
                # Select detection closest to previous detection
                selected_idx = self.select_closest_detection(boxes, prev_center)
                selected_box = boxes[selected_idx]
                
                x1, y1, x2, y2 = map(int, selected_box)
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
                
                # Update previous center for next frame
                prev_center = (center_x, center_y)
                
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
        
        print(f"\nDetection complete! Found ball in {len(self.detections)} frames.")
        return True
    
    def smooth_detections(self, window_size=5, smooth_type='diameter'):
        """
        Smooth ball parameters across frames
        Uses a moving average filter to reduce jitter
        
        Args:
            window_size: number of frames to average (default: 5)
            smooth_type: what to smooth - 'diameter', 'position', or 'both' (default: 'diameter')
        """
        if len(self.detections) == 0:
            return
        
        # Validate smooth_type
        if smooth_type not in ['diameter', 'position', 'both']:
            print(f"Invalid smooth_type '{smooth_type}', using 'diameter'")
            smooth_type = 'diameter'
        
        print(f"\nSmoothing detections ({smooth_type}) with window size {window_size}...")
        
        # Extract parameters
        frames = [d['frame'] for d in self.detections]
        center_x = np.array([d['center_x'] for d in self.detections], dtype=np.float32)
        center_y = np.array([d['center_y'] for d in self.detections], dtype=np.float32)
        diameter = np.array([d['diameter'] for d in self.detections], dtype=np.float32)
        
        # Apply moving average smoothing
        def moving_average(data, window):
            """Apply moving average with edge handling"""
            smoothed = np.copy(data)
            half_window = window // 2
            
            for i in range(len(data)):
                # Determine window bounds
                start = max(0, i - half_window)
                end = min(len(data), i + half_window + 1)
                
                # Calculate average
                smoothed[i] = np.mean(data[start:end])
            
            return smoothed
        
        # Smooth parameters based on smooth_type
        smoothed_center_x = center_x.copy()
        smoothed_center_y = center_y.copy()
        smoothed_diameter = diameter.copy()
        
        if smooth_type in ['position', 'both']:
            smoothed_center_x = moving_average(center_x, window_size)
            smoothed_center_y = moving_average(center_y, window_size)
        
        if smooth_type in ['diameter', 'both']:
            smoothed_diameter = moving_average(diameter, window_size)
        
        # Update detections with smoothed values
        for i, detection in enumerate(self.detections):
            detection['center_x'] = int(round(smoothed_center_x[i]))
            detection['center_y'] = int(round(smoothed_center_y[i]))
            detection['diameter'] = int(round(smoothed_diameter[i]))
        
        print(f"Smoothing complete!")
    
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
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        print(f"\nGenerating annotated images...")
        
        for detection in self.detections:
            frame_idx = detection['frame']
            if frame_idx >= len(image_files):
                continue
            
            img_path = image_files[frame_idx]
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Draw bounding box (thinner line)
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Draw circle (thinner line)
            cx, cy = detection['center_x'], detection['center_y']
            radius = detection['diameter'] // 2
            cv2.circle(image, (cx, cy), radius, (255, 0, 0), 1)
            
            # Draw center dot (smaller)
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
            
            # Draw label (thinner text)
            label = f"{detection['air_ground'].upper()} {detection['diameter']}px"
            label_y = max(y - 10, 20)
            cv2.putText(image, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Save annotated image
            output_path = self.annotated_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), image)
            print(f"Saved {output_path.name}", end='\r')
        
        print(f"\nAnnotated images saved to {self.annotated_dir}")


class BallCorrector:
    """Interactive bbox correction tool"""
    
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
        self.temp_image = None
        
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
                'diameter': diameter,
                'crop_cx': cx,
                'crop_cy': cy
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
        - 'f': fix/correct current frame (draw bbox)
        - 's': switch to next circle option within current bbox
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
                # Draw current bbox being drawn (thinner)
                if self.start_point and self.end_point:
                    cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 255), 1)
                    cv2.putText(display, "Draw new bbox (release to apply)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # Draw current detection (thinner)
                x, y, w, h = current_detection['x'], current_detection['y'], \
                            current_detection['w'], current_detection['h']
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # Draw circle (thinner)
                cx, cy = current_detection['center_x'], current_detection['center_y']
                radius = current_detection['diameter'] // 2
                cv2.circle(display, (cx, cy), radius, (255, 0, 0), 1)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), -1)
                
                # Draw other available circles in cyan
                if len(self.current_circles) > 1:
                    for idx, circle in enumerate(self.current_circles):
                        if idx != self.current_circle_idx:
                            r = circle['diameter'] // 2
                            cv2.circle(display, (circle['cx'], circle['cy']), r, (255, 255, 0), 1)
                
                # Draw label (thinner text)
                label = f"{current_detection['air_ground'].upper()} {current_detection['diameter']}px"
                cv2.putText(display, label, (x, max(y - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Frame info with circle count
                info = f"Frame {self.current_frame_idx}/{len(self.detections)-1} | Press 'f' to fix | Circles: {self.current_circle_idx+1}/{len(self.current_circles)}"
                cv2.putText(display, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
        
        # Always save updated detections and regenerate images
        print(f"\n{corrections_made} corrections/switches made.")
        
        # Save updated detections
        df = pd.DataFrame(self.detections)
        df.to_csv(self.csv_path, index=False)
        print(f"Updated detections saved to {self.csv_path}")
        
        # Always regenerate annotated images based on latest CSV
        print("\nRegenerating annotated images from latest detections...")
        self.generate_annotated_images(frames_dir)
        print("Annotated images regenerated successfully!")
        
        return corrections_made > 0
    
    def generate_annotated_images(self, frames_dir):
        """
        Generate annotated images with bbox, circle, center, and labels (using thin lines)
        """
        frames_path = Path(frames_dir)
        image_files = sorted(frames_path.glob("*.jpg")) + \
                     sorted(frames_path.glob("*.png")) + \
                     sorted(frames_path.glob("*.jpeg"))
        
        annotated_dir = self.results_dir / "annotated_frames"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating annotated images...")
        
        for detection in self.detections:
            frame_idx = detection['frame']
            if frame_idx >= len(image_files):
                continue
            
            img_path = image_files[frame_idx]
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Draw bounding box (thin line)
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Draw circle (thin line)
            cx, cy = detection['center_x'], detection['center_y']
            radius = detection['diameter'] // 2
            cv2.circle(image, (cx, cy), radius, (255, 0, 0), 1)
            
            # Draw center dot (small)
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)
            
            # Draw label (thin text)
            label = f"{detection['air_ground'].upper()} {detection['diameter']}px"
            label_y = max(y - 10, 20)
            cv2.putText(image, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Save annotated image
            output_path = annotated_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(output_path), image)
            print(f"Saved {output_path.name}", end='\r')
        
        print(f"\nAnnotated images saved to {annotated_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple Ball Detection System")
    parser.add_argument("--mode", type=str, required=True, choices=['detect', 'correct', 'smooth'],
                       help="Mode: detect, correct, or smooth")
    parser.add_argument("--frames", type=str, default="downloads/frames/V2_1",
                       help="Input frames directory")
    parser.add_argument("--output", type=str, default="results/V2_1",
                       help="Output directory")
    parser.add_argument("--yolo", type=str, default="YOLO/output/yolo_training/weights/best.pt",
                       help="Path to YOLO model weights")
    parser.add_argument("--window", type=int, default=5,
                       help="Smoothing window size for smooth mode (default: 5)")
    parser.add_argument("--smooth-type", type=str, default='diameter', 
                       choices=['diameter', 'position', 'both'],
                       help="What to smooth in smooth mode: 'diameter', 'position', or 'both' (default: diameter)")
    
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
            print("\nNext steps:")
            print(f"  Smooth: python detect_ball.py --mode smooth --frames {args.frames} --output {args.output}")
            print(f"  Correct: python detect_ball.py --mode correct --frames {args.frames} --output {args.output}")
        
        elif args.mode == 'correct':
            print("="*60)
            print("BBOX CORRECTION MODE")
            print("="*60)
            
            # Initialize corrector
            corrector = BallCorrector(args.output)
            
            # Run correction session
            corrections_made = corrector.run_correction_session(args.frames)
            
            if corrections_made:
                # Ask for airborne frame again
                print("\n" + "="*60)
                airborne_frame = input("Enter frame number where ball goes airborne (0 for first frame): ")
                try:
                    airborne_frame = int(airborne_frame)
                except ValueError:
                    print("Invalid input, keeping existing labels")
                    airborne_frame = None
                
                if airborne_frame is not None:
                    # Update labels
                    detector = BallDetector(args.yolo, args.output)
                    detector.detections = corrector.detections
                    detector.set_air_ground_labels(airborne_frame)
                    detector.save_detections()
                    
                    # Regenerate annotated images
                    detector.generate_annotated_images(args.frames)
                
                print("\n" + "="*60)
                print("CORRECTION COMPLETE!")
                print("="*60)
                print(f"Updated results saved to: {args.output}")
                print("\nNext step:")
                print(f"  Smooth: python detect_ball.py --mode smooth --frames {args.frames} --output {args.output}")
            else:
                print("\nNo changes made.")
        
        elif args.mode == 'smooth':
            print("="*60)
            print("SMOOTHING MODE")
            print("="*60)
            
            # Load existing detections
            csv_path = Path(args.output) / "detections.csv"
            if not csv_path.exists():
                print(f"Error: No detections found at {csv_path}")
                print("Please run detection mode first!")
                sys.exit(1)
            
            # Initialize detector and load detections
            detector = BallDetector(args.yolo, args.output)
            df = pd.read_csv(csv_path)
            detector.detections = df.to_dict('records')
            
            print(f"Loaded {len(detector.detections)} detections from {csv_path}")
            
            # Apply smoothing
            detector.smooth_detections(window_size=args.window, smooth_type=args.smooth_type)
            
            # Save smoothed detections
            detector.save_detections()
            
            # Regenerate annotated images with smoothed data
            detector.generate_annotated_images(args.frames)
            
            print("\n" + "="*60)
            print("SMOOTHING COMPLETE!")
            print("="*60)
            print(f"Smoothed results saved to: {args.output}")
            print(f"Smoothing type: {args.smooth_type}")
            print(f"Window size: {args.window}")
            print(f"CSV: {args.output}/detections.csv")
            print(f"Annotated frames: {args.output}/annotated_frames/")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

