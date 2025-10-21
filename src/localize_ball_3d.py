#!/usr/bin/env python3
"""
3D Ball Localization Pipeline with Polynomial Fitting
Based on Van Zandycke et al. CVPRW 2022

Four-stage pipeline:
1. Geometric Localization: Diameter-based depth estimation (Van Zandycke formula)
2. Physics Constraints: Parabolic trajectories + ground plane enforcement
3. Temporal Continuity: ICP-inspired iterative refinement with velocity constraints
4. Polynomial Fitting: Degree-2 polynomials fitted to air frames only

Coordinate System (PnLCalib convention):
- Origin: Field center (midfield line)
- X-axis: Field length (±52.5m for 105m field)
- Y-axis: Field width (±34m for 68m field)
- Z-axis: NEGATIVE is UP (z=0 is ground, z<0 is airborne)
  
Visualization: Plots show inverted Z-axis (height) for intuitive viewing
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import json


class BallLocalizer3D:
    def __init__(self, ball_diameter=0.22, frame_rate=30.0, max_velocity=40.0):
        """
        Initialize 3D ball localizer
        
        Args:
            ball_diameter: Real ball diameter in meters (default: 0.22m for football)
            frame_rate: Video frame rate in fps (default: 30.0)
            max_velocity: Maximum realistic ball velocity in m/s (default: 40.0)
        """
        self.ball_diameter = ball_diameter
        self.ball_radius = ball_diameter / 2.0
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate  # Time between frames
        self.max_velocity = max_velocity
        self.results = []
    
    def load_data(self, calibration_csv, detections_csv):
        """Load camera calibration and detection data"""
        print(f"Loading calibration from {calibration_csv}...")
        self.calib_df = pd.read_csv(calibration_csv)
        print(f"  ✓ Loaded {len(self.calib_df)} calibration frames")
        
        print(f"Loading detections from {detections_csv}...")
        self.detect_df = pd.read_csv(detections_csv)
        print(f"  ✓ Loaded {len(self.detect_df)} detection frames")
        
        # Match frames
        common_frames = set(self.calib_df['frame']) & set(self.detect_df['frame'])
        print(f"  ✓ Found {len(common_frames)} common frames")
        
        return len(common_frames) > 0
    
    def compute_3d_position_vanzandycke(self, center_x, center_y, diameter, K, R, cam_pos, is_ground=False):
        """
        Compute 3D position using Van Zandycke et al. CVPRW 2022 formula
        
        Formula:
            b_o = R^T · (φ · b_c) / (e_c^+_y - e_c^-_y) + c_o
        
        Where:
            - b_c: center ray in camera coordinates = K^{-1} · [b_x, b_y, 1]^T
            - e_c^±: edge rays = K^{-1} · [b_x, b_y ± d/2, 1]^T
            - φ: real ball diameter
            - R: rotation matrix, c_o: camera position
        
        Args:
            center_x, center_y: Ball center in image (pixels)
            diameter: Ball diameter in image (pixels)
            K: Camera intrinsic matrix (3x3)
            R: Camera rotation matrix (3x3)
            cam_pos: Camera position in world coordinates (3,)
            is_ground: If True, constrain ball to ground plane (z = 0)
        
        Returns:
            ball_world: 3D position in world coordinates (3,) or None
            depth: Distance from camera or None
        """
        # Check for valid diameter
        if diameter < 1e-10:
            return None, None
        
        # Compute inverse of K
        K_inv = np.linalg.inv(K)
        
        # Ball center ray in camera coordinates: b_c = K^{-1} · [b_x, b_y, 1]^T
        p_center = np.array([center_x, center_y, 1.0])
        b_c = K_inv @ p_center
        
        if is_ground:
            # For ground balls, use ray-plane intersection with z=0 plane
            # This gives more accurate x,y positioning
            
            # Transform ray direction to world coordinates
            ray_camera = b_c / np.linalg.norm(b_c)  # Normalized ray in camera coords
            ray_world = R.T @ ray_camera  # Ray direction in world coords
            
            # Solve for t where: cam_pos[2] + t * ray_world[2] = 0
            if abs(ray_world[2]) < 1e-10:
                # Ray is parallel to ground plane, fall back to Van Zandycke method
                pass  # Will use Van Zandycke method below
            else:
                t = -cam_pos[2] / ray_world[2]
                
                if t > 0:  # Valid intersection in front of camera
                    # Compute intersection point
                    b_world = cam_pos + t * ray_world
                    
                    # Set z to be at ball center height when on ground
                    # Since z is negative up, ground contact means z = -ball_radius
                    b_world[2] = -self.ball_radius
                    
                    # Compute depth (distance from camera to ball)
                    depth = np.linalg.norm(b_world - cam_pos)
                    
                    return b_world, depth
        
        # Van Zandycke method (for air balls or fallback)
        # Edge rays: e_c^± = K^{-1} · [b_x, b_y ± d/2, 1]^T
        p_edge_top = np.array([center_x, center_y - diameter/2.0, 1.0])
        p_edge_bottom = np.array([center_x, center_y + diameter/2.0, 1.0])
        
        e_c_top = K_inv @ p_edge_top
        e_c_bottom = K_inv @ p_edge_bottom
        
        # Compute denominator: e_c^+_y - e_c^-_y (difference in y-components)
        # Note: top edge has smaller y-coordinate, so we use bottom - top
        delta_y = e_c_bottom[1] - e_c_top[1]
        
        if abs(delta_y) < 1e-10:
            return None, None
        
        # Van Zandycke formula: b_o = R^T · (φ · b_c) / delta_y + c_o
        b_camera = (self.ball_diameter * b_c) / delta_y
        b_world = R.T @ b_camera + cam_pos
        
        # For ground balls, enforce z constraint
        if is_ground:
            b_world[2] = -self.ball_radius
        
        # Compute depth
        depth = np.linalg.norm(b_camera)
        
        return b_world, depth
    
    def process_all_frames(self, apply_physics=False, apply_temporal=False):
        """
        Process all frames and compute 3D positions
        
        Args:
            apply_physics: Apply physics-based corrections (Stage 2)
            apply_temporal: Apply temporal continuity refinement (Stage 3)
        """
        print("\n" + "="*60)
        print("STAGE 1: GEOMETRIC LOCALIZATION (Van Zandycke Method)")
        print("="*60)
        
        self.results = []
        
        for _, det_row in self.detect_df.iterrows():
            frame = det_row['frame']
            
            # Find matching calibration frame
            calib_row = self.calib_df[self.calib_df['frame'] == frame]
            if len(calib_row) == 0:
                continue
            
            calib_row = calib_row.iloc[0]
            
            # Build camera intrinsic matrix
            fx, fy = calib_row['fx'], calib_row['fy']
            cx, cy = calib_row['cx'], calib_row['cy']
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
            
            # Build rotation matrix
            R = np.array([[calib_row['r11'], calib_row['r12'], calib_row['r13']],
                         [calib_row['r21'], calib_row['r22'], calib_row['r23']],
                         [calib_row['r31'], calib_row['r32'], calib_row['r33']]])
            
            # Camera position
            cam_pos = np.array([calib_row['cam_x'], calib_row['cam_y'], calib_row['cam_z']])
            
            # Ball detection
            center_x = det_row['center_x']
            center_y = det_row['center_y']
            diameter = det_row['diameter']
            air_ground = det_row.get('air_ground', 'unknown')
            is_ground = (air_ground == 'ground')
            
            # Compute 3D position using Van Zandycke method
            ball_world, depth = self.compute_3d_position_vanzandycke(
                center_x, center_y, diameter, K, R, cam_pos, is_ground=is_ground
            )
            
            if ball_world is not None:
                self.results.append({
                    'frame': frame,
                    'ball_x': ball_world[0],
                    'ball_y': ball_world[1],
                    'ball_z': ball_world[2],
                    'center_x': center_x,
                    'center_y': center_y,
                    'diameter': diameter,
                    'depth': depth,
                    'air_ground': air_ground
                })
        
        print(f"  ✓ Computed 3D positions for {len(self.results)} frames")
        
        # Stage 2: Physics-based corrections
        if apply_physics and len(self.results) > 0:
            print("\n" + "="*60)
            print("STAGE 2: PHYSICS-BASED CORRECTIONS")
            print("="*60)
            self.apply_physics_corrections()
        
        # Stage 3: Temporal continuity
        if apply_temporal and len(self.results) > 0:
            print("\n" + "="*60)
            print("STAGE 3: TEMPORAL CONTINUITY REFINEMENT")
            print("="*60)
            self.apply_temporal_continuity()
        
        return len(self.results) > 0
    
    def apply_physics_corrections(self):
        """
        STAGE 2: Apply physics-based corrections to the trajectory
        
        1. Enforce ground plane constraint (z = -ball_radius) for ground segments
        2. Fit parabolic trajectories for air segments (gravity effect)
        3. Smooth x,y coordinates with Savitzky-Golay filter
        """
        df = pd.DataFrame(self.results)
        
        # Identify trajectory segments (ground vs air)
        segments = self._identify_segments(df)
        print(f"  ✓ Identified {len(segments)} trajectory segments")
        
        corrected_results = []
        
        for seg_idx, (seg_type, start_idx, end_idx) in enumerate(segments):
            seg_df = df.iloc[start_idx:end_idx+1].copy()
            seg_len = len(seg_df)
            
            if seg_type == 'ground':
                print(f"    Segment {seg_idx+1}: Ground ({seg_len} frames)")
                corrected_seg = self._correct_ground_segment(seg_df)
            else:
                print(f"    Segment {seg_idx+1}: Air ({seg_len} frames)")
                corrected_seg = self._correct_air_segment(seg_df)
            
            corrected_results.extend(corrected_seg)
        
        # Update results
        self.results = corrected_results
        print(f"  ✓ Applied physics corrections to all segments")
    
    def _identify_segments(self, df):
        """Identify contiguous segments of ground/air/unknown"""
        segments = []
        if len(df) == 0:
            return segments
        
        current_type = df.iloc[0]['air_ground']
        start_idx = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['air_ground'] != current_type:
                segments.append((current_type, start_idx, i-1))
                current_type = df.iloc[i]['air_ground']
                start_idx = i
        
        # Add final segment
        segments.append((current_type, start_idx, len(df)-1))
        
        return segments
    
    def _correct_ground_segment(self, seg_df):
        """
        Correct ground segment:
        - Enforce z = -ball_radius (ball center at radius above ground)
        - Smooth x, y using Savitzky-Golay filter
        """
        corrected = []
        
        x = seg_df['ball_x'].values
        y = seg_df['ball_y'].values
        
        # Smooth x, y coordinates
        if len(x) >= 5:
            window = min(5, len(x) if len(x) % 2 == 1 else len(x) - 1)
            x_smooth = savgol_filter(x, window_length=window, polyorder=2)
            y_smooth = savgol_filter(y, window_length=window, polyorder=2)
        elif len(x) >= 3:
            x_smooth = savgol_filter(x, window_length=3, polyorder=1)
            y_smooth = savgol_filter(y, window_length=3, polyorder=1)
        else:
            x_smooth = x
            y_smooth = y
        
        # Build corrected results
        for i, (idx, row) in enumerate(seg_df.iterrows()):
            corrected.append({
                'frame': row['frame'],
                'ball_x': x_smooth[i],
                'ball_y': y_smooth[i],
                'ball_z': -self.ball_radius,  # Enforce ground constraint
                'center_x': row['center_x'],
                'center_y': row['center_y'],
                'diameter': row['diameter'],
                'depth': row['depth'],
                'air_ground': row['air_ground']
            })
        
        return corrected
    
    def _correct_air_segment(self, seg_df):
        """
        Correct air segment using physics:
        1. Fit parabolic trajectory in z (gravity: z(t) = z0 + vz*t - 0.5*g*t^2)
        2. Smooth trajectory in x, y (Savitzky-Golay filter)
        3. Ensure z doesn't go above ground (z <= -ball_radius)
        """
        if len(seg_df) < 3:
            # Too short to fit, return as-is with ground constraint
            results = seg_df.to_dict('records')
            for r in results:
                r['ball_z'] = min(r['ball_z'], -self.ball_radius)
            return results
        
        corrected = []
        
        # Extract positions
        x = seg_df['ball_x'].values
        y = seg_df['ball_y'].values
        z = seg_df['ball_z'].values
        
        # Time parameter (normalized)
        t = np.arange(len(x), dtype=float)
        
        try:
            # Fit parabolic trajectory for z: z(t) = a*t^2 + b*t + c
            z_coeffs = np.polyfit(t, z, deg=min(2, len(t)-1))
            z_fitted = np.polyval(z_coeffs, t)
            
            # Smooth with Savitzky-Golay and blend with parabolic fit
            if len(z) >= 5:
                window = min(5, len(z) if len(z) % 2 == 1 else len(z) - 1)
                z_smooth = savgol_filter(z, window_length=window, polyorder=2)
                # Blend: favor parabolic fit for longer segments
                alpha = 0.6 if len(z) >= 10 else 0.3
                z_corrected = alpha * z_fitted + (1 - alpha) * z_smooth
            else:
                z_corrected = z_fitted
            
            # Ensure z doesn't go above ground
            z_corrected = np.minimum(z_corrected, -self.ball_radius)
            
            # Smooth x, y
            if len(x) >= 5:
                window = min(5, len(x) if len(x) % 2 == 1 else len(x) - 1)
                x_corrected = savgol_filter(x, window_length=window, polyorder=2)
                y_corrected = savgol_filter(y, window_length=window, polyorder=2)
            elif len(x) >= 3:
                x_corrected = savgol_filter(x, window_length=3, polyorder=1)
                y_corrected = savgol_filter(y, window_length=3, polyorder=1)
            else:
                x_corrected = x
                y_corrected = y
            
        except Exception as e:
            print(f"      Warning: Physics correction failed: {e}")
            x_corrected = x
            y_corrected = y
            z_corrected = np.minimum(z, -self.ball_radius)
        
        # Build corrected results
        for i, (idx, row) in enumerate(seg_df.iterrows()):
            corrected.append({
                'frame': row['frame'],
                'ball_x': x_corrected[i],
                'ball_y': y_corrected[i],
                'ball_z': z_corrected[i],
                'center_x': row['center_x'],
                'center_y': row['center_y'],
                'diameter': row['diameter'],
                'depth': row['depth'],
                'air_ground': row['air_ground']
            })
        
        return corrected
    
    def apply_temporal_continuity(self):
        """
        STAGE 3: Apply temporal continuity refinement (ICP-inspired)
        
        Uses knowledge of frame rate (30 fps) to enforce:
        1. Velocity-constrained smoothing (forward + backward passes)
        2. Ground frames as spatial anchors
        3. Iterative refinement to align air frames with ground constraints
        """
        df = pd.DataFrame(self.results)
        n = len(df)
        
        if n < 3:
            print("  ⚠ Too few frames for temporal refinement")
            return
        
        # Extract positions and labels
        positions = df[['ball_x', 'ball_y', 'ball_z']].values.copy()
        is_ground = (df['air_ground'] == 'ground').values
        
        print(f"  Frame rate: {self.frame_rate} fps (dt = {self.dt:.4f}s)")
        print(f"  Max velocity constraint: {self.max_velocity} m/s")
        print(f"  Ground frames: {is_ground.sum()} / {n}")
        
        # Step 1: Velocity-constrained smoothing (forward pass)
        print("\n  Step 1: Forward pass (velocity-constrained smoothing)...")
        positions_fwd = positions.copy()
        for i in range(1, n):
            if is_ground[i]:
                continue  # Keep ground frames fixed
            
            # Predict based on previous velocity
            if i > 1 and not is_ground[i-1]:
                v_prev = (positions_fwd[i-1] - positions_fwd[i-2]) / self.dt
                predicted = positions_fwd[i-1] + v_prev * self.dt
            else:
                predicted = positions_fwd[i-1]
            
            # Blend prediction with observation
            alpha = 0.3  # Weight for observation
            blended = alpha * positions[i] + (1 - alpha) * predicted
            
            # Enforce velocity constraint
            delta = blended - positions_fwd[i-1]
            velocity = np.linalg.norm(delta) / self.dt
            if velocity > self.max_velocity:
                delta = delta * (self.max_velocity * self.dt / np.linalg.norm(delta))
            
            positions_fwd[i] = positions_fwd[i-1] + delta
        
        # Step 2: Velocity-constrained smoothing (backward pass)
        print("  Step 2: Backward pass (velocity-constrained smoothing)...")
        positions_bwd = positions_fwd.copy()
        for i in range(n - 2, -1, -1):
            if is_ground[i]:
                continue  # Keep ground frames fixed
            
            # Predict based on next velocity
            if i < n - 2 and not is_ground[i+1]:
                v_next = (positions_bwd[i+2] - positions_bwd[i+1]) / self.dt
                predicted = positions_bwd[i+1] - v_next * self.dt
            else:
                predicted = positions_bwd[i+1]
            
            # Blend with forward pass result
            alpha = 0.5
            blended = alpha * positions_fwd[i] + (1 - alpha) * predicted
            
            positions_bwd[i] = blended
        
        # Step 3: Iterative refinement (ICP-inspired)
        print("  Step 3: Iterative refinement (ICP-inspired)...")
        positions_refined = positions_bwd.copy()
        
        iterations = 5
        for iteration in range(iterations):
            energy_before = self._compute_energy(positions_refined, is_ground)
            
            for i in range(n):
                if is_ground[i]:
                    continue  # Ground frames are anchors
                
                # Smoothness constraint: average of neighbors
                if i > 0 and i < n - 1:
                    smooth_pos = 0.5 * (positions_refined[i-1] + positions_refined[i+1])
                elif i > 0:
                    smooth_pos = positions_refined[i-1]
                else:
                    smooth_pos = positions_refined[i+1]
                
                # Find nearest ground anchors
                ground_indices = np.where(is_ground)[0]
                if len(ground_indices) > 0:
                    # Find closest ground frame before and after
                    before = ground_indices[ground_indices < i]
                    after = ground_indices[ground_indices > i]
                    
                    if len(before) > 0 and len(after) > 0:
                        # Interpolate between ground anchors
                        idx_before = before[-1]
                        idx_after = after[0]
                        t = (i - idx_before) / (idx_after - idx_before)
                        anchor_pos = (1 - t) * positions_refined[idx_before] + t * positions_refined[idx_after]
                        
                        # Blend smoothness with anchor constraint
                        alpha = 0.7  # Weight for smoothness
                        beta = 0.3   # Weight for anchor
                        target_pos = alpha * smooth_pos + beta * anchor_pos
                    else:
                        target_pos = smooth_pos
                else:
                    target_pos = smooth_pos
                
                # Update with small step (damping)
                step_size = 0.3
                positions_refined[i] = positions_refined[i] + step_size * (target_pos - positions_refined[i])
            
            energy_after = self._compute_energy(positions_refined, is_ground)
            print(f"    Iteration {iteration+1}: Energy = {energy_after:.6f} (Δ = {energy_before - energy_after:.6f})")
        
        # Update results
        for i, result in enumerate(self.results):
            result['ball_x'] = positions_refined[i, 0]
            result['ball_y'] = positions_refined[i, 1]
            result['ball_z'] = positions_refined[i, 2]
        
        print(f"  ✓ Temporal continuity refinement complete")
    
    def _compute_energy(self, positions, is_ground):
        """
        Compute energy metric for trajectory quality
        
        Energy = sum of velocity changes + distance from ground anchors
        """
        n = len(positions)
        energy = 0.0
        
        # Velocity change penalty
        for i in range(1, n - 1):
            v1 = positions[i] - positions[i-1]
            v2 = positions[i+1] - positions[i]
            dv = v2 - v1
            energy += np.linalg.norm(dv) ** 2
        
        # Distance from ground anchors (for air frames)
        ground_indices = np.where(is_ground)[0]
        for i in range(n):
            if not is_ground[i] and len(ground_indices) > 0:
                # Distance to nearest ground frame
                distances = [np.linalg.norm(positions[i] - positions[j]) for j in ground_indices]
                min_dist = min(distances)
                energy += 0.1 * min_dist ** 2
        
        return energy
    
    def _fit_bezier(self, t, pos):
        """
        Fit cubic Bézier curve (4 control points) to trajectory
        
        Bézier curve: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        where t ∈ [0, 1] (normalized time)
        """
        from scipy.optimize import least_squares
        
        # Normalize t to [0, 1]
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)
        
        # Initial guess for control points (P0 is FIXED as anchor)
        P0 = pos[0]  # Start point (FIXED - anchor)
        P3 = pos[-1]  # End point
        P1 = pos[0] + (pos[-1] - pos[0]) * 0.33  # 1/3 along
        P2 = pos[0] + (pos[-1] - pos[0]) * 0.67  # 2/3 along
        
        def bezier_curve(params, t_norm, P0_fixed):
            P1, P2, P3 = params
            return ((1 - t_norm)**3 * P0_fixed + 
                    3 * (1 - t_norm)**2 * t_norm * P1 + 
                    3 * (1 - t_norm) * t_norm**2 * P2 + 
                    t_norm**3 * P3)
        
        def residuals(params):
            return pos - bezier_curve(params, t_norm, P0)
        
        # Optimize P1, P2, P3 (P0 is fixed as anchor)
        result = least_squares(residuals, [P1, P2, P3], method='lm')
        control_points = np.array([P0, result.x[0], result.x[1], result.x[2]])
        
        # Compute fitted curve
        fitted = bezier_curve(result.x, t_norm, P0)
        
        return fitted, control_points
    
    def _fit_exponential_skew(self, t, pos):
        """
        Fit exponential decay function: f(t) = A*exp(-k*t) + B*t + C
        
        This creates asymmetric curves with exponential decay component
        Constrained to pass through anchor point (t=0, pos[0])
        """
        from scipy.optimize import least_squares
        
        # Normalize t
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)
        
        # Anchor point constraint: at t_norm=0, f(0) = pos[0]
        # f(0) = A*exp(0) + B*0 + C = A + C
        # We need: A + C = pos[0]
        # Let's fix C and optimize A, k, B
        
        def exp_decay(params, t_norm, pos_0):
            A, k, B = params
            C = pos_0 - A  # Ensure f(0) = A + C = pos_0
            return A * np.exp(-k * t_norm) + B * t_norm + C
        
        def residuals(params):
            return pos - exp_decay(params, t_norm, pos[0])
        
        # Initial guess - make exponential component significant
        A_init = (pos.max() - pos.min()) * 0.3  # Significant exponential amplitude
        k_init = 2.0  # Moderate decay rate
        B_init = (pos[-1] - pos[0]) * 0.8  # Linear trend
        
        # Add bounds to ensure meaningful exponential behavior
        result = least_squares(residuals, [A_init, k_init, B_init], 
                              bounds=([-np.inf, 0.1, -np.inf], [np.inf, 10.0, np.inf]),
                              method='trf', max_nfev=2000)
        
        A, k, B = result.x
        C = pos[0] - A
        params = np.array([A, k, B, C])
        
        fitted = exp_decay(result.x, t_norm, pos[0])
        
        return fitted, params
    
    def _fit_mu_s_curve(self, t, pos):
        """
        Fit Mu-S curve (sigmoid-modulated parabola) for skewed trajectories
        
        f(t) = (At² + Bt + C) * (1 + D*tanh(E*(t - F)))
        
        Combines parabola with sigmoid to create asymmetry
        Constrained to pass through anchor point (t=0, pos[0])
        """
        from scipy.optimize import least_squares
        
        # Normalize t
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)
        
        # Anchor point constraint: at t_norm=0, f(0) = pos[0]
        # f(0) = (A*0 + B*0 + C) * (1 + D*tanh(E*(0 - F)))
        #      = C * (1 + D*tanh(-E*F))
        # For simplicity, we can constrain C such that the curve passes through pos[0]
        
        # First fit simple parabola to get initial estimates
        poly_coeffs = np.polyfit(t_norm, pos, deg=2)
        A_init, B_init, _ = poly_coeffs
        
        def mu_s_curve(params, t_norm, pos_0):
            A, B, D, E, F = params
            # Compute C such that f(0) = pos_0
            sigmoid_0 = 1 + D * np.tanh(-E * F)
            C = pos_0 / (sigmoid_0 + 1e-10)  # Avoid division by zero
            
            parabola = A * t_norm**2 + B * t_norm + C
            sigmoid_skew = 1 + D * np.tanh(E * (t_norm - F))
            return parabola * sigmoid_skew
        
        def residuals(params):
            return pos - mu_s_curve(params, t_norm, pos[0])
        
        # Initial guess: parabola with small skew
        D_init = 0.1  # Small skew factor
        E_init = 2.0  # Skew steepness
        F_init = 0.5  # Skew center (middle of trajectory)
        
        result = least_squares(residuals, [A_init, B_init, D_init, E_init, F_init],
                              method='lm', max_nfev=1000)
        
        # Compute C for the final parameters
        A, B, D, E, F = result.x
        sigmoid_0 = 1 + D * np.tanh(-E * F)
        C = pos[0] / (sigmoid_0 + 1e-10)
        params = np.array([A, B, C, D, E, F])
        
        fitted = mu_s_curve(result.x, t_norm, pos[0])
        
        return fitted, params
    
    def fit_polynomial_trajectory(self):
        """
        Fit polynomial equations with cross terms to last ground point + air frames
        
        This method:
        1. Uses last ground point as anchor
        2. Adds all air frames
        3. Fits equations with cross terms: X(t) + xy + yz + zx terms
        4. Computes fitted positions and metrics (R², RMSE)
        5. Stores results for JSON/CSV export
        
        Returns:
            dict: Polynomial fit results or None if insufficient data
        """
        print("\n" + "="*60)
        print("POLYNOMIAL TRAJECTORY FITTING (PARABOLIC)")
        print("="*60)
        
        df = pd.DataFrame(self.results)
        
        # Get air frames
        air_mask = df['air_ground'] == 'air'
        df_air = df[air_mask].copy()
        
        # Get last ground point as anchor
        ground_mask = df['air_ground'] == 'ground'
        df_ground = df[ground_mask]
        
        n_air = len(df_air)
        n_ground = len(df_ground)
        n_total = len(df)
        
        print(f"Total frames: {n_total}")
        print(f"Air frames: {n_air}")
        print(f"Ground frames: {n_ground}")
        
        if n_air < 3:
            print("  ⚠ Warning: Too few air frames for polynomial fitting")
            return None
        
        # Combine last ground point + air frames
        if n_ground > 0:
            last_ground = df_ground.iloc[[-1]]  # Get last ground point
            df_fit = pd.concat([last_ground, df_air], ignore_index=True)
            print(f"\n✓ Including last ground point as anchor: frame {last_ground['frame'].values[0]}")
        else:
            df_fit = df_air
            print(f"\n⚠ No ground points available")
        
        print(f"Total points for fitting: {len(df_fit)}")
        
        # Extract frame numbers and compute time (relative to anchor point)
        frames = df_fit['frame'].values
        t = (frames - frames[0]) / self.frame_rate  # Time relative to first point (anchor)
        
        print(f"Frame range: [{frames[0]}, {frames[-1]}]")
        print(f"Time range: [{t[0]:.3f}, {t[-1]:.3f}] seconds")
        print(f"Duration: {t[-1] - t[0]:.3f} seconds")
        
        # Extract positions
        x = df_fit['ball_x'].values
        y = df_fit['ball_y'].values
        z = df_fit['ball_z'].values
        
        print(f"\nPosition ranges:")
        print(f"  X: [{x.min():.3f}, {x.max():.3f}] m")
        print(f"  Y: [{y.min():.3f}, {y.max():.3f}] m")
        print(f"  Z: [{z.min():.3f}, {z.max():.3f}] m")
        
        # Compute all 4 fitting methods for comparison
        print(f"\nComputing all 4 trajectory fitting methods...")
        
        # Method 1: Simple Polynomial (degree-2, baseline) - constrained to pass through anchor
        print(f"\n1. Simple Polynomial (degree-2 parabola):")
        # For polynomial f(t) = at² + bt + c, constrain f(0) = pos[0]
        # At t=0: f(0) = c, so c = pos[0] (FIXED)
        # Fit only a and b using least squares
        from scipy.optimize import least_squares
        
        def poly_constrained(params, t, c_fixed):
            a, b = params
            return a * t**2 + b * t + c_fixed
        
        def fit_poly_constrained(t, pos):
            def residuals(params):
                return pos - poly_constrained(params, t, pos[0])
            
            # Initial guess from unconstrained fit
            poly_init = np.polyfit(t, pos, deg=2)
            a_init, b_init = poly_init[0], poly_init[1]
            
            result = least_squares(residuals, [a_init, b_init], method='lm')
            a, b = result.x
            c = pos[0]
            return np.array([a, b, c]), poly_constrained([a, b], t, c)
        
        x_poly_coeffs, x_poly = fit_poly_constrained(t, x)
        y_poly_coeffs, y_poly = fit_poly_constrained(t, y)
        z_poly_coeffs, z_poly = fit_poly_constrained(t, z)
        
        rmse_poly = np.sqrt(np.mean((x - x_poly)**2 + (y - y_poly)**2 + (z - z_poly)**2))
        r2_poly = 1 - np.sum((x - x_poly)**2 + (y - y_poly)**2 + (z - z_poly)**2) / \
                      np.sum((x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2)
        print(f"   RMSE: {rmse_poly:.4f} m, R²: {r2_poly:.6f}")
        print(f"   Anchor constraint: f(0) = ({x[0]:.3f}, {y[0]:.3f}, {z[0]:.3f})")
        
        # Method 2: Cubic Bézier (4 control points)
        print(f"\n2. Cubic Bézier (4 control points):")
        x_bezier, bezier_x_params = self._fit_bezier(t, x)
        y_bezier, bezier_y_params = self._fit_bezier(t, y)
        z_bezier, bezier_z_params = self._fit_bezier(t, z)
        rmse_bezier = np.sqrt(np.mean((x - x_bezier)**2 + (y - y_bezier)**2 + (z - z_bezier)**2))
        r2_bezier = 1 - np.sum((x - x_bezier)**2 + (y - y_bezier)**2 + (z - z_bezier)**2) / \
                        np.sum((x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2)
        print(f"   RMSE: {rmse_bezier:.4f} m, R²: {r2_bezier:.6f}")
        
        # Method 3: Exponential decay f(t) = A*exp(-k*t) + B*t + C
        print(f"\n3. Exponential decay f(t) = A*exp(-k*t) + B*t + C:")
        x_exp, exp_x_params = self._fit_exponential_skew(t, x)
        y_exp, exp_y_params = self._fit_exponential_skew(t, y)
        z_exp, exp_z_params = self._fit_exponential_skew(t, z)
        rmse_exp = np.sqrt(np.mean((x - x_exp)**2 + (y - y_exp)**2 + (z - z_exp)**2))
        r2_exp = 1 - np.sum((x - x_exp)**2 + (y - y_exp)**2 + (z - z_exp)**2) / \
                     np.sum((x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2)
        print(f"   RMSE: {rmse_exp:.4f} m, R²: {r2_exp:.6f}")
        
        # Method 4: Mu-S curve (sigmoid-based skewed parabola)
        print(f"\n4. Mu-S curve (sigmoid skew):")
        x_mus, mus_x_params = self._fit_mu_s_curve(t, x)
        y_mus, mus_y_params = self._fit_mu_s_curve(t, y)
        z_mus, mus_z_params = self._fit_mu_s_curve(t, z)
        rmse_mus = np.sqrt(np.mean((x - x_mus)**2 + (y - y_mus)**2 + (z - z_mus)**2))
        r2_mus = 1 - np.sum((x - x_mus)**2 + (y - y_mus)**2 + (z - z_mus)**2) / \
                     np.sum((x - x.mean())**2 + (y - y.mean())**2 + (z - z.mean())**2)
        print(f"   RMSE: {rmse_mus:.4f} m, R²: {r2_mus:.6f}")
        
        # Store all 4 methods for plotting and CSV output
        self.all_fitted_trajectories = {
            'polynomial': {
                'x': x_poly, 'y': y_poly, 'z': z_poly,
                'rmse': rmse_poly, 'r2': r2_poly,
                'params': {'X': x_poly_coeffs, 'Y': y_poly_coeffs, 'Z': z_poly_coeffs}
            },
            'bezier': {
                'x': x_bezier, 'y': y_bezier, 'z': z_bezier,
                'rmse': rmse_bezier, 'r2': r2_bezier,
                'params': {'X': bezier_x_params, 'Y': bezier_y_params, 'Z': bezier_z_params}
            },
            'exponential': {
                'x': x_exp, 'y': y_exp, 'z': z_exp,
                'rmse': rmse_exp, 'r2': r2_exp,
                'params': {'X': exp_x_params, 'Y': exp_y_params, 'Z': exp_z_params}
            },
            'mu_s': {
                'x': x_mus, 'y': y_mus, 'z': z_mus,
                'rmse': rmse_mus, 'r2': r2_mus,
                'params': {'X': mus_x_params, 'Y': mus_y_params, 'Z': mus_z_params}
            }
        }
        
        # Select best method
        methods_rmse = {
            'polynomial': rmse_poly,
            'bezier': rmse_bezier,
            'exponential': rmse_exp,
            'mu_s': rmse_mus
        }
        
        best_method = min(methods_rmse.keys(), key=lambda k: methods_rmse[k])
        best_rmse = methods_rmse[best_method]
        
        print(f"\n✓ Best method: {best_method.upper()} (RMSE: {best_rmse:.4f} m)")
        print(f"  All 4 methods will be saved to CSV and shown in plots")
        
        # Use best method for primary output
        x_fitted = self.all_fitted_trajectories[best_method]['x']
        y_fitted = self.all_fitted_trajectories[best_method]['y']
        z_fitted = self.all_fitted_trajectories[best_method]['z']
        x_params = self.all_fitted_trajectories[best_method]['params']['X']
        y_params = self.all_fitted_trajectories[best_method]['params']['Y']
        z_params = self.all_fitted_trajectories[best_method]['params']['Z']
        
        # Store method and parameters
        self.fitting_method = best_method
        self.fitting_params = {
            'X': x_params,
            'Y': y_params,
            'Z': z_params
        }
        
        # Create dummy coefficients for compatibility
        x_coeffs_full = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y_coeffs_full = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z_coeffs_full = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Approximate with polynomial for initial conditions
        x_coeffs = np.polyfit(t, x_fitted, deg=2)
        y_coeffs = np.polyfit(t, y_fitted, deg=2)
        z_coeffs = np.polyfit(t, z_fitted, deg=2)
        
        # Compute metrics for each coordinate
        def compute_metrics(observed, fitted, coord_name):
            residuals = observed - fitted
            rmse = np.sqrt(np.mean(residuals**2))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observed - np.mean(observed))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            max_error = np.max(np.abs(residuals))
            
            return {
                'rmse': float(rmse),
                'r_squared': float(r_squared),
                'max_error': float(max_error)
            }
        
        x_metrics = compute_metrics(x, x_fitted, 'X')
        y_metrics = compute_metrics(y, y_fitted, 'Y')
        z_metrics = compute_metrics(z, z_fitted, 'Z')
        
        # Overall metrics
        all_residuals = np.concatenate([x - x_fitted, y - y_fitted, z - z_fitted])
        overall_rmse = np.sqrt(np.mean(all_residuals**2))
        avg_r_squared = (x_metrics['r_squared'] + y_metrics['r_squared'] + z_metrics['r_squared']) / 3.0
        
        print(f"\n{'='*60}")
        print("PARABOLIC FIT RESULTS (PROJECTILE MOTION)")
        print(f"{'='*60}")
        
        print(f"\nX(t) = {x_coeffs[2]:.6f} + {x_coeffs[1]:.6f}*t + {x_coeffs[0]:.6f}*t²")
        print(f"  RMSE: {x_metrics['rmse']:.4f} m")
        print(f"  R²: {x_metrics['r_squared']:.6f}")
        print(f"  Max error: {x_metrics['max_error']:.4f} m")
        
        print(f"\nY(t) = {y_coeffs[2]:.6f} + {y_coeffs[1]:.6f}*t + {y_coeffs[0]:.6f}*t²")
        print(f"  RMSE: {y_metrics['rmse']:.4f} m")
        print(f"  R²: {y_metrics['r_squared']:.6f}")
        print(f"  Max error: {y_metrics['max_error']:.4f} m")
        
        print(f"\nZ(t) = {z_coeffs[2]:.6f} + {z_coeffs[1]:.6f}*t + {z_coeffs[0]:.6f}*t²")
        print(f"  RMSE: {z_metrics['rmse']:.4f} m")
        print(f"  R²: {z_metrics['r_squared']:.6f}")
        print(f"  Max error: {z_metrics['max_error']:.4f} m")
        
        print(f"\nOverall fit quality:")
        print(f"  Overall RMSE: {overall_rmse:.4f} m")
        print(f"  Average R²: {avg_r_squared:.6f}")
        
        # Compute initial conditions (at t=0, anchor point)
        x0, y0, z0 = x_coeffs[2], y_coeffs[2], z_coeffs[2]
        vx0, vy0, vz0 = x_coeffs[1], y_coeffs[1], z_coeffs[1]  # First derivative at t=0
        v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
        
        print(f"\nInitial conditions (at t=0, anchor point):")
        print(f"  Position: ({x0:.3f}, {y0:.3f}, {z0:.3f}) m")
        print(f"  Velocity: ({vx0:.3f}, {vy0:.3f}, {vz0:.3f}) m/s")
        print(f"  Speed: {v0:.3f} m/s")
        
        # Compute accelerations (second derivative, constant for degree 2)
        ax = 2 * x_coeffs[0]
        ay = 2 * y_coeffs[0]
        az = 2 * z_coeffs[0]
        
        print(f"\nAcceleration (parabolic motion):")
        print(f"  a_x: {ax:.3f} m/s² (horizontal drag/spin)")
        print(f"  a_y: {ay:.3f} m/s² (lateral drift)")
        print(f"  a_z: {az:.3f} m/s² (gravity + air resistance)")
        print(f"  Expected: a_z ≈ 9.81 m/s² (gravity)")
        
        print(f"{'='*60}\n")
        
        # Store results
        self.polynomial_fit = {
            'frames': frames,
            'time': t,
            'x_observed': x,
            'y_observed': y,
            'z_observed': z,
            'x_fitted': x_fitted,
            'y_fitted': y_fitted,
            'z_fitted': z_fitted,
            'anchor_frame': int(frames[0]),
            'n_ground_included': 1 if n_ground > 0 else 0,
            'coefficients': {
                'X': x_coeffs.tolist(),
                'Y': y_coeffs.tolist(),
                'Z': z_coeffs.tolist(),
                'X_cross': [float(x_coeffs_full[3]), float(x_coeffs_full[4])],  # [y, z]
                'Y_cross': [float(y_coeffs_full[3]), float(y_coeffs_full[4])],  # [x, z]
                'Z_cross': [float(z_coeffs_full[3]), float(z_coeffs_full[4])]   # [x, y]
            },
            'metrics': {
                'X': x_metrics,
                'Y': y_metrics,
                'Z': z_metrics,
                'overall_rmse': float(overall_rmse),
                'average_r_squared': float(avg_r_squared)
            },
            'initial_conditions': {
                'position': [float(x0), float(y0), float(z0)],
                'velocity': [float(vx0), float(vy0), float(vz0)],
                'speed': float(v0)
            },
            'acceleration': {
                'x': float(ax),
                'y': float(ay),
                'z': float(az)
            }
        }
        
        return self.polynomial_fit
    
    def save_polynomial_equations_json(self, output_json):
        """
        Save polynomial equations and metadata to JSON file
        
        Args:
            output_json: Path to output JSON file
        """
        if not hasattr(self, 'polynomial_fit') or self.polynomial_fit is None:
            print("  ⚠ No polynomial fit available to save")
            return
        
        fit = self.polynomial_fit
        coeffs = fit['coefficients']
        
        # Build JSON structure
        output_data = {
            'metadata': {
                'ball_diameter': float(self.ball_diameter),
                'frame_rate': float(self.frame_rate),
                'n_frames': int(len(fit['frames'])),
                'time_range': [float(fit['time'][0]), float(fit['time'][-1])],
                'frame_range': [int(fit['frames'][0]), int(fit['frames'][-1])],
                'anchor_frame': fit['anchor_frame'],
                'includes_ground_anchor': fit['n_ground_included'] > 0,
                'has_cross_terms': False,
                'model_type': 'skewed_trajectory',
                'fitting_method': self.fitting_method
            },
            'equations': {
                'X': {
                    'coefficients': coeffs['X'],
                    'equation': f"X(t) = {coeffs['X'][2]:.6f} + {coeffs['X'][1]:.6f}*t + {coeffs['X'][0]:.6f}*t²",
                    'r_squared': fit['metrics']['X']['r_squared'],
                    'rmse': fit['metrics']['X']['rmse'],
                    'max_error': fit['metrics']['X']['max_error'],
                    'physical_meaning': 'Horizontal position (field length)'
                },
                'Y': {
                    'coefficients': coeffs['Y'],
                    'equation': f"Y(t) = {coeffs['Y'][2]:.6f} + {coeffs['Y'][1]:.6f}*t + {coeffs['Y'][0]:.6f}*t²",
                    'r_squared': fit['metrics']['Y']['r_squared'],
                    'rmse': fit['metrics']['Y']['rmse'],
                    'max_error': fit['metrics']['Y']['max_error'],
                    'physical_meaning': 'Lateral position (field width)'
                },
                'Z': {
                    'coefficients': coeffs['Z'],
                    'equation': f"Z(t) = {coeffs['Z'][2]:.6f} + {coeffs['Z'][1]:.6f}*t + {coeffs['Z'][0]:.6f}*t²",
                    'r_squared': fit['metrics']['Z']['r_squared'],
                    'rmse': fit['metrics']['Z']['rmse'],
                    'max_error': fit['metrics']['Z']['max_error'],
                    'physical_meaning': 'Vertical position (height, negative Z is up)'
                }
            },
            'initial_conditions': fit['initial_conditions'],
            'acceleration': fit['acceleration'],
            'overall_metrics': {
                'overall_rmse': fit['metrics']['overall_rmse'],
                'average_r_squared': fit['metrics']['average_r_squared']
            },
            'fitting_parameters': {
                'best_method': self.fitting_method,
                'polynomial': {
                    'X_params': [float(p) for p in self.all_fitted_trajectories['polynomial']['params']['X']],
                    'Y_params': [float(p) for p in self.all_fitted_trajectories['polynomial']['params']['Y']],
                    'Z_params': [float(p) for p in self.all_fitted_trajectories['polynomial']['params']['Z']],
                    'rmse': float(self.all_fitted_trajectories['polynomial']['rmse']),
                    'r_squared': float(self.all_fitted_trajectories['polynomial']['r2'])
                },
                'bezier': {
                    'X_params': [float(p) for p in self.all_fitted_trajectories['bezier']['params']['X']],
                    'Y_params': [float(p) for p in self.all_fitted_trajectories['bezier']['params']['Y']],
                    'Z_params': [float(p) for p in self.all_fitted_trajectories['bezier']['params']['Z']],
                    'rmse': float(self.all_fitted_trajectories['bezier']['rmse']),
                    'r_squared': float(self.all_fitted_trajectories['bezier']['r2'])
                },
                'exponential': {
                    'X_params': [float(p) for p in self.all_fitted_trajectories['exponential']['params']['X']],
                    'Y_params': [float(p) for p in self.all_fitted_trajectories['exponential']['params']['Y']],
                    'Z_params': [float(p) for p in self.all_fitted_trajectories['exponential']['params']['Z']],
                    'rmse': float(self.all_fitted_trajectories['exponential']['rmse']),
                    'r_squared': float(self.all_fitted_trajectories['exponential']['r2'])
                },
                'mu_s': {
                    'X_params': [float(p) for p in self.all_fitted_trajectories['mu_s']['params']['X']],
                    'Y_params': [float(p) for p in self.all_fitted_trajectories['mu_s']['params']['Y']],
                    'Z_params': [float(p) for p in self.all_fitted_trajectories['mu_s']['params']['Z']],
                    'rmse': float(self.all_fitted_trajectories['mu_s']['rmse']),
                    'r_squared': float(self.all_fitted_trajectories['mu_s']['r2'])
                }
            }
        }
        
        # Save to JSON
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"  ✓ Saved polynomial equations to {output_json}")
    
    def save_fitted_trajectory_csv(self, output_csv):
        """
        Save fitted trajectories (all 3 methods) to CSV
        
        Args:
            output_csv: Path to output CSV file
        """
        if not hasattr(self, 'polynomial_fit') or self.polynomial_fit is None:
            print("  ⚠ No polynomial fit available to save")
            return
        
        fit = self.polynomial_fit
        
        # Determine air_ground labels
        # First frame is anchor (last ground point if available), rest are air
        if fit['n_ground_included'] > 0:
            air_ground_labels = ['ground'] + ['air'] * (len(fit['frames']) - 1)
        else:
            air_ground_labels = ['air'] * len(fit['frames'])
        
        # Create DataFrame with all 4 fitted trajectories
        df_fitted = pd.DataFrame({
            'frame': fit['frames'],
            'time': fit['time'],
            'air_ground': air_ground_labels,
            # Polynomial fit (baseline)
            'polynomial_x': self.all_fitted_trajectories['polynomial']['x'],
            'polynomial_y': self.all_fitted_trajectories['polynomial']['y'],
            'polynomial_z': self.all_fitted_trajectories['polynomial']['z'],
            # Bézier fit
            'bezier_x': self.all_fitted_trajectories['bezier']['x'],
            'bezier_y': self.all_fitted_trajectories['bezier']['y'],
            'bezier_z': self.all_fitted_trajectories['bezier']['z'],
            # Exponential fit
            'exponential_x': self.all_fitted_trajectories['exponential']['x'],
            'exponential_y': self.all_fitted_trajectories['exponential']['y'],
            'exponential_z': self.all_fitted_trajectories['exponential']['z'],
            # Mu-S fit
            'mu_s_x': self.all_fitted_trajectories['mu_s']['x'],
            'mu_s_y': self.all_fitted_trajectories['mu_s']['y'],
            'mu_s_z': self.all_fitted_trajectories['mu_s']['z'],
            # Best method (for backward compatibility)
            'ball_x': fit['x_fitted'],
            'ball_y': fit['y_fitted'],
            'ball_z': fit['z_fitted']
        })
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_fitted.to_csv(output_path, index=False, float_format='%.6f')
        
        print(f"  ✓ Saved all 4 fitted trajectories to {output_csv}")
        print(f"    - Columns: polynomial_*, bezier_*, exponential_*, mu_s_*, ball_* (best: {self.fitting_method})")
    
    def save_results(self, output_csv):
        """Save 3D positions to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"\n  ✓ Saved 3D positions to {output_csv}")
    
    def plot_results(self, output_dir):
        """Generate visualization plots"""
        if len(self.results) == 0:
            print("No results to plot!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        print("\nGenerating plots...")
        
        # 1. 3D Scatter Plot
        self._plot_3d_scatter(df, output_dir)
        
        # 2. 3D Trajectory
        self._plot_3d_trajectory(df, output_dir)
        
        # 3. Top View
        self._plot_top_view(df, output_dir)
        
        # 4. Side View (X-Z)
        self._plot_side_view(df, output_dir)
        
        # 5. Side View (Y-Z)
        self._plot_side_view_y_z(df, output_dir)
        
        # 6. Velocity Analysis
        self._plot_velocity_analysis(df, output_dir)
        
        print(f"  ✓ All plots saved to {output_dir}")
    
    def _plot_3d_scatter(self, df, output_dir):
        """3D scatter plot with color-coded frames"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Invert Z for display (show -Z as positive height)
        scatter = ax.scatter(df['ball_x'], df['ball_y'], -df['ball_z'],
                            c=df['frame'], cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title('3D Ball Trajectory (Scatter)', fontsize=14, fontweight='bold')
        
        # Set axis limits for standard football field (105m x 68m)
        ax.set_xlim(-52.5, 52.5)
        ax.set_ylim(-34, 34)
        max_height = max(5, -df['ball_z'].min() + 1)
        ax.set_zlim(0, max_height)
        
        # Add ground plane at height=0
        xx, yy = np.meshgrid(np.linspace(-52.5, 52.5, 10),
                            np.linspace(-34, 34, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')
        
        plt.colorbar(scatter, ax=ax, label='Frame', pad=0.1)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_3d_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ 3D scatter plot")

    def _plot_segments(self, ax, xs, ys, zs, labels, color_map, linewidth=2, alpha=0.8):
        """Plot contiguous segments colored by label"""
        try:
            xs_s = xs.reset_index(drop=True)
            ys_s = ys.reset_index(drop=True)
            labels_s = labels.reset_index(drop=True)
            zs_s = zs.reset_index(drop=True) if zs is not None else None
        except (AttributeError, TypeError):
            # Input is not a pandas Series, convert it
            xs_s = pd.Series(xs).reset_index(drop=True)
            ys_s = pd.Series(ys).reset_index(drop=True)
            labels_s = pd.Series(labels).reset_index(drop=True)
            zs_s = pd.Series(zs).reset_index(drop=True) if zs is not None else None

        n = len(xs_s)
        i = 0
        while i < n - 1:
            lbl = labels_s.iloc[i]
            j = i + 1
            while j < n and labels_s.iloc[j] == lbl:
                j += 1

            # Plot segment from i to j
            if zs_s is None:
                ax.plot(xs_s.iloc[i:j], ys_s.iloc[i:j], color=color_map.get(lbl, 'gray'),
                        linewidth=linewidth, alpha=alpha)
            else:
                ax.plot(xs_s.iloc[i:j], ys_s.iloc[i:j], zs_s.iloc[i:j], color=color_map.get(lbl, 'gray'),
                        linewidth=linewidth, alpha=alpha)

            i = j

        # Scatter points colored by label
        for lbl, col in color_map.items():
            mask = labels_s == lbl
            if mask.any():
                if zs_s is None:
                    ax.scatter(xs_s[mask], ys_s[mask], c=col, s=30, alpha=0.8, label=lbl.capitalize())
                else:
                    ax.scatter(xs_s[mask], ys_s[mask], zs_s[mask], c=col, s=30, alpha=0.8, label=lbl.capitalize())
    
    def _plot_3d_trajectory(self, df, output_dir):
        """3D trajectory line plot with all 4 fitting methods"""
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filter to air frames only
        df_air = df[df['air_ground'] == 'air'].copy()
        
        # Invert Z for display (show -Z as positive height)
        df_air['ball_z'] = -df_air['ball_z']
        
        # Plot observed trajectory (air frames only)
        ax.scatter(df_air['ball_x'], df_air['ball_y'], df_air['ball_z'],
                  c='gray', s=40, alpha=0.7, label='Observed', marker='o', zorder=1)
        
        # Plot all 4 fitted curves if available
        if hasattr(self, 'all_fitted_trajectories'):
            # Polynomial (magenta)
            poly = self.all_fitted_trajectories['polynomial']
            ax.plot(poly['x'], poly['y'], -np.array(poly['z']),
                   color='magenta', linewidth=2.5, alpha=0.9, linestyle='-.',
                   label=f"Polynomial (RMSE: {poly['rmse']:.3f}m)", zorder=3)
            
            # Bézier (red)
            bezier = self.all_fitted_trajectories['bezier']
            ax.plot(bezier['x'], bezier['y'], -np.array(bezier['z']),
                   color='red', linewidth=2.5, alpha=0.9, 
                   label=f"Bézier (RMSE: {bezier['rmse']:.3f}m)", zorder=5)
            
            # Exponential (blue)
            exp = self.all_fitted_trajectories['exponential']
            ax.plot(exp['x'], exp['y'], -np.array(exp['z']),
                   color='blue', linewidth=2.5, alpha=0.9, linestyle='--',
                   label=f"Exponential (RMSE: {exp['rmse']:.3f}m)", zorder=4)
            
            # Mu-S (green)
            mus = self.all_fitted_trajectories['mu_s']
            ax.plot(mus['x'], mus['y'], -np.array(mus['z']),
                   color='green', linewidth=2.5, alpha=0.9, linestyle=':',
                   label=f"Mu-S (RMSE: {mus['rmse']:.3f}m)", zorder=2)
        
        # Mark anchor point (first frame - last ground point)
        if hasattr(self, 'polynomial_fit') and self.polynomial_fit['n_ground_included'] > 0:
            anchor_x = self.polynomial_fit['x_fitted'][0]
            anchor_y = self.polynomial_fit['y_fitted'][0]
            anchor_z = -self.polynomial_fit['z_fitted'][0]  # Invert for display
            ax.scatter([anchor_x], [anchor_y], [anchor_z],
                      c='gold', s=300, marker='*', label='Anchor (Last Ground)', 
                      edgecolors='darkgoldenrod', linewidths=3, zorder=11)
        
        # Mark start and end of air trajectory
        ax.scatter([df_air['ball_x'].iloc[0]], [df_air['ball_y'].iloc[0]], [df_air['ball_z'].iloc[0]],
                  c='lime', s=250, marker='o', label='Start (Air)', edgecolors='darkgreen', linewidths=2.5, zorder=10)
        ax.scatter([df_air['ball_x'].iloc[-1]], [df_air['ball_y'].iloc[-1]], [df_air['ball_z'].iloc[-1]],
                  c='orange', s=250, marker='X', label='End', edgecolors='darkorange', linewidths=2.5, zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title('3D Ball Trajectory - All 4 Fitting Methods', fontsize=14, fontweight='bold')
        
        # Set axis limits for standard football field (105m x 68m)
        ax.set_xlim(-52.5, 52.5)
        ax.set_ylim(-34, 34)
        max_height = max(5, df_air['ball_z'].max() + 1)
        ax.set_zlim(0, max_height)
        
        # Add ground plane at height=0
        xx, yy = np.meshgrid(np.linspace(-52.5, 52.5, 10),
                            np.linspace(-34, 34, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='lightgray')
        
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9, markerscale=0.7)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_3d_trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ 3D trajectory plot (all 4 methods)")
    
    def _plot_top_view(self, df, output_dir):
        """Top view (X-Y plane) with all 4 fitting methods"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Filter to air frames only
        df_air = df[df['air_ground'] == 'air'].copy()
        
        # Plot observed trajectory (air frames only)
        ax.scatter(df_air['ball_x'], df_air['ball_y'],
                  c='gray', s=40, alpha=0.7, label='Observed', marker='o', zorder=1)
        
        # Plot all 4 fitted curves if available
        if hasattr(self, 'all_fitted_trajectories'):
            # Polynomial (magenta)
            poly = self.all_fitted_trajectories['polynomial']
            ax.plot(poly['x'], poly['y'],
                   color='magenta', linewidth=2.5, alpha=0.9, linestyle='-.',
                   label=f"Polynomial (RMSE: {poly['rmse']:.3f}m)", zorder=3)
            
            # Bézier (red)
            bezier = self.all_fitted_trajectories['bezier']
            ax.plot(bezier['x'], bezier['y'],
                   color='red', linewidth=2.5, alpha=0.9, 
                   label=f"Bézier (RMSE: {bezier['rmse']:.3f}m)", zorder=5)
            
            # Exponential (blue)
            exp = self.all_fitted_trajectories['exponential']
            ax.plot(exp['x'], exp['y'],
                   color='blue', linewidth=2.5, alpha=0.9, linestyle='--',
                   label=f"Exponential (RMSE: {exp['rmse']:.3f}m)", zorder=4)
            
            # Mu-S (green)
            mus = self.all_fitted_trajectories['mu_s']
            ax.plot(mus['x'], mus['y'],
                   color='green', linewidth=2.5, alpha=0.9, linestyle=':',
                   label=f"Mu-S (RMSE: {mus['rmse']:.3f}m)", zorder=2)
        
        # Mark anchor point (first frame - last ground point)
        if hasattr(self, 'polynomial_fit') and self.polynomial_fit['n_ground_included'] > 0:
            anchor_x = self.polynomial_fit['x_fitted'][0]
            anchor_y = self.polynomial_fit['y_fitted'][0]
            ax.scatter([anchor_x], [anchor_y],
                      c='gold', s=300, marker='*', label='Anchor (Last Ground)', 
                      edgecolors='darkgoldenrod', linewidths=3, zorder=11)
        
        # Mark start and end of air trajectory
        ax.scatter([df_air['ball_x'].iloc[0]], [df_air['ball_y'].iloc[0]],
                  c='lime', s=250, marker='o', label='Start (Air)', edgecolors='darkgreen', linewidths=2.5, zorder=10)
        ax.scatter([df_air['ball_x'].iloc[-1]], [df_air['ball_y'].iloc[-1]],
                  c='orange', s=250, marker='X', label='End', edgecolors='darkorange', linewidths=2.5, zorder=10)
        
        # Add center lines at midfield
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('X (m) - Field Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m) - Field Width', fontsize=12, fontweight='bold')
        ax.set_title('Ball Trajectory - Top View (All 4 Methods)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9, markerscale=0.7)
        
        # Set axis limits centered at field center (105m x 68m)
        ax.set_xlim(-52.5, 52.5)
        ax.set_ylim(-34, 34)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_top_view.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Top view plot (all 4 methods)")
    
    def _plot_side_view(self, df, output_dir):
        """Side view (X-Z plane)"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Filter to air frames only
        df_air = df[df['air_ground'] == 'air'].copy()
        
        # Invert Z for display (show -Z as positive height)
        df_air['ball_z'] = -df_air['ball_z']
        
        # Plot observed trajectory (air frames only)
        ax.scatter(df_air['ball_x'], df_air['ball_z'],
                  c='gray', s=40, alpha=0.7, label='Observed', marker='o', zorder=1)
        
        # Plot all 4 fitted curves if available
        if hasattr(self, 'all_fitted_trajectories'):
            # Polynomial (magenta)
            poly = self.all_fitted_trajectories['polynomial']
            ax.plot(poly['x'], -np.array(poly['z']),
                   color='magenta', linewidth=2.5, alpha=0.9, linestyle='-.',
                   label=f"Polynomial (RMSE: {poly['rmse']:.3f}m)", zorder=3)
            
            # Bézier (red)
            bezier = self.all_fitted_trajectories['bezier']
            ax.plot(bezier['x'], -np.array(bezier['z']),
                   color='red', linewidth=2.5, alpha=0.9, 
                   label=f"Bézier (RMSE: {bezier['rmse']:.3f}m)", zorder=5)
            
            # Exponential (blue)
            exp = self.all_fitted_trajectories['exponential']
            ax.plot(exp['x'], -np.array(exp['z']),
                   color='blue', linewidth=2.5, alpha=0.9, linestyle='--',
                   label=f"Exponential (RMSE: {exp['rmse']:.3f}m)", zorder=4)
            
            # Mu-S (green)
            mus = self.all_fitted_trajectories['mu_s']
            ax.plot(mus['x'], -np.array(mus['z']),
                   color='green', linewidth=2.5, alpha=0.9, linestyle=':',
                   label=f"Mu-S (RMSE: {mus['rmse']:.3f}m)", zorder=2)
        
        # Mark anchor point (first frame - last ground point)
        if hasattr(self, 'polynomial_fit') and self.polynomial_fit['n_ground_included'] > 0:
            anchor_x = self.polynomial_fit['x_fitted'][0]
            anchor_z = -self.polynomial_fit['z_fitted'][0]  # Invert for display
            ax.scatter([anchor_x], [anchor_z],
                      c='gold', s=300, marker='*', label='Anchor (Last Ground)', 
                      edgecolors='darkgoldenrod', linewidths=3, zorder=11)
        
        # Mark start and end of air trajectory
        ax.scatter([df_air['ball_x'].iloc[0]], [df_air['ball_z'].iloc[0]],
                  c='lime', s=200, marker='o', label='Start (Air)', edgecolors='darkgreen', linewidths=2.5, zorder=10)
        ax.scatter([df_air['ball_x'].iloc[-1]], [df_air['ball_z'].iloc[-1]],
                  c='orange', s=200, marker='X', label='End', edgecolors='darkorange', linewidths=2.5, zorder=10)
        
        # Ground line at height=0
        ax.axhline(y=0, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
        # Midfield line
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('X (m) - Field Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title('Ball Trajectory - Side View X-Z (All 4 Methods)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9, markerscale=0.7)
        
        # Set axis limits centered at field center (105m length)
        ax.set_xlim(-52.5, 52.5)
        max_height = max(5, df_air['ball_z'].max() + 1)
        ax.set_ylim(-0.5, max_height)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_side_view_x_z.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Side view plot X-Z (all 4 methods)")
    
    def _plot_side_view_y_z(self, df, output_dir):
        """Side view (Y-Z plane)"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter to air frames only
        df_air = df[df['air_ground'] == 'air'].copy()
        
        # Invert Z for display (show -Z as positive height)
        df_air['ball_z'] = -df_air['ball_z']
        
        # Plot observed trajectory (air frames only)
        ax.scatter(df_air['ball_y'], df_air['ball_z'],
                   c='gray', s=40, alpha=0.7, label='Observed', marker='o', zorder=1)
        
        # Plot all 4 fitted curves if available
        if hasattr(self, 'all_fitted_trajectories'):
            # Polynomial (magenta)
            poly = self.all_fitted_trajectories['polynomial']
            ax.plot(poly['y'], -np.array(poly['z']),
                   color='magenta', linewidth=2.5, alpha=0.9, linestyle='-.',
                   label=f"Polynomial (RMSE: {poly['rmse']:.3f}m)", zorder=3)
            
            # Bézier (red)
            bezier = self.all_fitted_trajectories['bezier']
            ax.plot(bezier['y'], -np.array(bezier['z']),
                   color='red', linewidth=2.5, alpha=0.9, 
                   label=f"Bézier (RMSE: {bezier['rmse']:.3f}m)", zorder=5)
            
            # Exponential (blue)
            exp = self.all_fitted_trajectories['exponential']
            ax.plot(exp['y'], -np.array(exp['z']),
                   color='blue', linewidth=2.5, alpha=0.9, linestyle='--',
                   label=f"Exponential (RMSE: {exp['rmse']:.3f}m)", zorder=4)
            
            # Mu-S (green)
            mus = self.all_fitted_trajectories['mu_s']
            ax.plot(mus['y'], -np.array(mus['z']),
                   color='green', linewidth=2.5, alpha=0.9, linestyle=':',
                   label=f"Mu-S (RMSE: {mus['rmse']:.3f}m)", zorder=2)
        
        # Mark anchor point (first frame - last ground point)
        if hasattr(self, 'polynomial_fit') and self.polynomial_fit['n_ground_included'] > 0:
            anchor_y = self.polynomial_fit['y_fitted'][0]
            anchor_z = -self.polynomial_fit['z_fitted'][0]  # Invert for display
            ax.scatter([anchor_y], [anchor_z],
                      c='gold', s=300, marker='*', label='Anchor (Last Ground)', 
                      edgecolors='darkgoldenrod', linewidths=3, zorder=11)
        
        # Mark start and end of air trajectory
        ax.scatter([df_air['ball_y'].iloc[0]], [df_air['ball_z'].iloc[0]],
                  c='lime', s=200, marker='o', label='Start (Air)', edgecolors='darkgreen', linewidths=2.5, zorder=10)
        ax.scatter([df_air['ball_y'].iloc[-1]], [df_air['ball_z'].iloc[-1]],
                  c='orange', s=200, marker='X', label='End', edgecolors='darkorange', linewidths=2.5, zorder=10)
        
        # Ground line at height=0
        ax.axhline(y=0, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
        # Midfield line
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Y (m) - Field Width', fontsize=12, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
        ax.set_title('Ball Trajectory - Side View Y-Z (All 4 Methods)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9, markerscale=0.7)
        
        # Set axis limits centered at field center (68m width)
        ax.set_xlim(-34, 34)
        max_height = max(5, df_air['ball_z'].max() + 1)
        ax.set_ylim(-0.5, max_height)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_side_view_y_z.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Side view plot Y-Z (all 4 methods)")
    
    def _plot_velocity_analysis(self, df, output_dir):
        """Plot velocity analysis for all 4 fitting methods"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
        
        # Filter to air frames only
        df_air = df[df['air_ground'] == 'air'].copy()
        
        # Compute velocities for observed data
        positions_obs = df_air[['ball_x', 'ball_y', 'ball_z']].values
        velocities_obs = np.zeros(len(positions_obs))
        
        for i in range(1, len(positions_obs)):
            delta = positions_obs[i] - positions_obs[i-1]
            velocities_obs[i] = np.linalg.norm(delta) / self.dt
        
        # Plot 1: Velocity magnitude for all 4 methods
        ax1.plot(df_air['frame'], velocities_obs, color='gray', linewidth=2, alpha=0.6, 
                label='Observed', linestyle=':', marker='o', markersize=4, zorder=1)
        
        # Compute and plot velocities for all 4 fitted curves
        if hasattr(self, 'all_fitted_trajectories') and hasattr(self, 'polynomial_fit'):
            # Get frames from the fitted data (includes anchor point)
            fit_frames = self.polynomial_fit['frames']
            
            # Polynomial (magenta)
            poly = self.all_fitted_trajectories['polynomial']
            pos_poly = np.column_stack([poly['x'], poly['y'], poly['z']])
            vel_poly = np.zeros(len(pos_poly))
            for i in range(1, len(pos_poly)):
                vel_poly[i] = np.linalg.norm(pos_poly[i] - pos_poly[i-1]) / self.dt
            ax1.plot(fit_frames, vel_poly, color='magenta', linewidth=2.5, alpha=0.9, 
                    linestyle='-.', label=f'Polynomial (RMSE: {poly["rmse"]:.3f}m)', zorder=3)
            
            # Bézier (red)
            bezier = self.all_fitted_trajectories['bezier']
            pos_bezier = np.column_stack([bezier['x'], bezier['y'], bezier['z']])
            vel_bezier = np.zeros(len(pos_bezier))
            for i in range(1, len(pos_bezier)):
                vel_bezier[i] = np.linalg.norm(pos_bezier[i] - pos_bezier[i-1]) / self.dt
            ax1.plot(fit_frames, vel_bezier, color='red', linewidth=2.5, alpha=0.9, 
                    label=f'Bézier (RMSE: {bezier["rmse"]:.3f}m)', zorder=5)
            
            # Exponential (blue)
            exp = self.all_fitted_trajectories['exponential']
            pos_exp = np.column_stack([exp['x'], exp['y'], exp['z']])
            vel_exp = np.zeros(len(pos_exp))
            for i in range(1, len(pos_exp)):
                vel_exp[i] = np.linalg.norm(pos_exp[i] - pos_exp[i-1]) / self.dt
            ax1.plot(fit_frames, vel_exp, color='blue', linewidth=2.5, alpha=0.9, 
                    linestyle='--', label=f'Exponential (RMSE: {exp["rmse"]:.3f}m)', zorder=4)
            
            # Mu-S (green)
            mus = self.all_fitted_trajectories['mu_s']
            pos_mus = np.column_stack([mus['x'], mus['y'], mus['z']])
            vel_mus = np.zeros(len(pos_mus))
            for i in range(1, len(pos_mus)):
                vel_mus[i] = np.linalg.norm(pos_mus[i] - pos_mus[i-1]) / self.dt
            ax1.plot(fit_frames, vel_mus, color='green', linewidth=2.5, alpha=0.9, 
                    linestyle=':', label=f'Mu-S (RMSE: {mus["rmse"]:.3f}m)', zorder=2)
        
        ax1.axhline(y=self.max_velocity, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Max Constraint ({self.max_velocity} m/s)', alpha=0.7)
        ax1.set_xlabel('Frame', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Ball Velocity Over Time (All 4 Methods)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=9, loc='best', framealpha=0.9, markerscale=0.7)
        
        # Plot 2: Speed comparison (scalar velocity)
        ax2.plot(df_air['frame'], velocities_obs, color='gray', linewidth=2, alpha=0.6, 
                label='Observed', linestyle=':', marker='o', markersize=4, zorder=1)
        
        if hasattr(self, 'all_fitted_trajectories') and hasattr(self, 'polynomial_fit'):
            ax2.plot(fit_frames, vel_poly, color='magenta', linewidth=2.5, alpha=0.9, 
                    linestyle='-.', label='Polynomial', zorder=3)
            ax2.plot(fit_frames, vel_bezier, color='red', linewidth=2.5, alpha=0.9, 
                    label='Bézier', zorder=5)
            ax2.plot(fit_frames, vel_exp, color='blue', linewidth=2.5, alpha=0.9, 
                    linestyle='--', label='Exponential', zorder=4)
            ax2.plot(fit_frames, vel_mus, color='green', linewidth=2.5, alpha=0.9, 
                    linestyle=':', label='Mu-S', zorder=2)
        
        ax2.set_xlabel('Frame', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
        ax2.set_title('Speed Profile Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=9, loc='best', framealpha=0.9, markerscale=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ball_velocity_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Velocity analysis plot (all 4 methods)")


def main():
    parser = argparse.ArgumentParser(
        description="3D Ball Localization Pipeline (Van Zandycke + Physics + Temporal Continuity)"
    )
    parser.add_argument("--calibration", type=str, required=True,
                       help="Path to camera calibration CSV")
    parser.add_argument("--detections", type=str, required=True,
                       help="Path to detections CSV")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV path for 3D positions")
    parser.add_argument("--ball-diameter", type=float, default=0.22,
                       help="Real ball diameter in meters (default: 0.22 for football)")
    parser.add_argument("--frame-rate", type=float, default=30.0,
                       help="Video frame rate in fps (default: 30.0)")
    parser.add_argument("--max-velocity", type=float, default=40.0,
                       help="Maximum realistic ball velocity in m/s (default: 40.0)")
    parser.add_argument("--apply-physics", action='store_true',
                       help="Apply physics-based trajectory corrections (Stage 2)")
    parser.add_argument("--apply-temporal", action='store_true',
                       help="Apply temporal continuity refinement (Stage 3)")
    parser.add_argument("--plot", action='store_true',
                       help="Generate visualization plots")
    parser.add_argument("--plot-output", type=str, default=None,
                       help="Directory for saving plots (default: same as output CSV)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D BALL LOCALIZATION PIPELINE")
    print("="*60)
    print(f"Method: Van Zandycke et al. CVPRW 2022")
    print(f"Ball diameter: {args.ball_diameter}m")
    print(f"Frame rate: {args.frame_rate} fps")
    print(f"Max velocity: {args.max_velocity} m/s")
    print(f"Physics corrections: {'YES' if args.apply_physics else 'NO'}")
    print(f"Temporal refinement: {'YES' if args.apply_temporal else 'NO'}")
    print()
    
    # Initialize localizer
    localizer = BallLocalizer3D(
        ball_diameter=args.ball_diameter,
        frame_rate=args.frame_rate,
        max_velocity=args.max_velocity
    )
    
    # Load data
    if not localizer.load_data(args.calibration, args.detections):
        print("Error: No common frames found!")
        return 1
    
    # Process all frames (includes all 3 stages if enabled)
    if not localizer.process_all_frames(
        apply_physics=args.apply_physics,
        apply_temporal=args.apply_temporal
    ):
        print("Error: No 3D positions computed!")
        return 1
    
    # Save results (all frames, air + ground)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    localizer.save_results(output_path)
    
    # Fit polynomial trajectory (air frames only)
    polynomial_fit = localizer.fit_polynomial_trajectory()
    
    if polynomial_fit is not None:
        # Save polynomial equations to JSON
        json_path = output_path.parent / 'trajectory_equations.json'
        localizer.save_polynomial_equations_json(json_path)
        
        # Save fitted trajectory to CSV (air frames only)
        fitted_csv_path = output_path.parent / 'ball_3d_positions_fitted.csv'
        localizer.save_fitted_trajectory_csv(fitted_csv_path)
    
    # Generate plots
    if args.plot:
        plot_dir = Path(args.plot_output) if args.plot_output else output_path.parent
        localizer.plot_results(plot_dir)
    
    print("\n" + "="*60)
    print("✓ 3D LOCALIZATION PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📊 OUTPUT FILES:")
    print(f"  • All frames (air + ground): {args.output}")
    if polynomial_fit is not None:
        print(f"  • Fitted trajectory (air only): {output_path.parent / 'ball_3d_positions_fitted.csv'}")
        print(f"  • Polynomial equations (JSON): {output_path.parent / 'trajectory_equations.json'}")
    if args.plot:
        print(f"\n📈 VISUALIZATIONS:")
        print(f"  • Plots directory: {plot_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
