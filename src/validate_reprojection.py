#!/usr/bin/env python3
"""
3D to 2D Reprojection Validation
Reprojects 3D ball positions back to 2D and compares with original detections
Computes reprojection error as validation metric
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle

class ReprojectionValidator:
    def __init__(self, method='best'):
        """
        Initialize reprojection validator
        
        Args:
            method: Which fitted trajectory to use for validation
                   'best' (default), 'polynomial', 'bezier', 'exponential', 'mu_s'
        """
        self.results = []
        self.method = method.lower()
        
        # Map method names to column prefixes
        self.method_map = {
            'best': 'ball',  # Uses ball_x, ball_y, ball_z (best method)
            'polynomial': 'polynomial',
            'bezier': 'bezier',
            'exponential': 'exponential',
            'mu_s': 'mu_s'
        }
        
        if self.method not in self.method_map:
            raise ValueError(f"Invalid method '{method}'. Choose from: {list(self.method_map.keys())}")
        
        self.column_prefix = self.method_map[self.method]
    
    def load_data(self, positions_csv, calibration_csv, detections_csv):
        """Load 3D positions, calibration, and original detections"""
        print("Loading data...")
        
        self.positions_df = pd.read_csv(positions_csv)
        print(f"  ✓ Loaded {len(self.positions_df)} 3D positions")
        
        # Check if selected method columns exist
        x_col = f'{self.column_prefix}_x'
        y_col = f'{self.column_prefix}_y'
        z_col = f'{self.column_prefix}_z'
        
        if x_col not in self.positions_df.columns:
            available_methods = []
            for method_name, prefix in self.method_map.items():
                if f'{prefix}_x' in self.positions_df.columns:
                    available_methods.append(method_name)
            
            raise ValueError(
                f"Method '{self.method}' not found in CSV. "
                f"Available methods: {available_methods}"
            )
        
        print(f"  ✓ Using method: {self.method.upper()} ({x_col}, {y_col}, {z_col})")
        
        self.calib_df = pd.read_csv(calibration_csv)
        print(f"  ✓ Loaded {len(self.calib_df)} calibration frames")
        
        self.detect_df = pd.read_csv(detections_csv)
        print(f"  ✓ Loaded {len(self.detect_df)} detections")
        
        return True
    
    def reproject_3d_to_2d(self, ball_3d, K, R, cam_pos):
        """
        Reproject 3D world position to 2D image coordinates
        
        Following the exact projection formula from calibrate.py:
        - It = [I | -cam_pos] (3x4 matrix)
        - P = K @ (R @ It)
        - p = P @ [x, y, z, 1]^T
        - Normalize: (p[0]/p[2], p[1]/p[2])
        
        Args:
            ball_3d: 3D position in world coordinates (3,) - already centered
            K: Camera intrinsic matrix (3x3)
            R: Camera rotation matrix (3x3)
            cam_pos: Camera position in world coordinates (3,) - already centered
        
        Returns:
            (x, y): 2D image coordinates in pixels
        """
        # Build It matrix: [I | -cam_pos] (3x4)
        It = np.eye(4)[:3, :]  # Take first 3 rows of 4x4 identity
        It[:, -1] = -cam_pos
        
        # Build projection matrix: P = K @ (R @ It)
        P = K @ (R @ It)
        
        # Build homogeneous world point [x, y, z, 1]
        ball_world_homo = np.array([ball_3d[0], ball_3d[1], ball_3d[2], 1.0])
        
        # Project to image coordinates
        p = P @ ball_world_homo
        
        # Normalize by depth (homogeneous coordinate)
        if abs(p[2]) < 1e-10:
            return None, None
        
        x = p[0] / p[2]
        y = p[1] / p[2]
        
        return x, y
    
    def compute_reprojection_errors(self):
        """
        Compute reprojection errors for all frames with weighted loss
        
        Uses reconstruction error from camera calibration as confidence weights:
        - High recon_error (poor calibration) → lower weight → less penalty
        - Low recon_error (good calibration) → higher weight → more penalty
        """
        print("Computing reprojection errors with weighted loss...")
        
        self.results = []
        errors = []
        recon_errors = []
        
        for _, pos_row in self.positions_df.iterrows():
            frame = pos_row['frame']
            
            # Get calibration for this frame
            calib_row = self.calib_df[self.calib_df['frame'] == frame]
            if len(calib_row) == 0:
                continue
            calib_row = calib_row.iloc[0]
            
            # Get original detection for this frame
            det_row = self.detect_df[self.detect_df['frame'] == frame]
            if len(det_row) == 0:
                continue
            det_row = det_row.iloc[0]
            
            # Get reconstruction error from calibration (confidence metric)
            recon_error = calib_row.get('recon_error', -1.0)
            if recon_error < 0:
                recon_error = 10.0  # Default high error for invalid frames
            
            # Build camera matrices
            fx, fy = calib_row['fx'], calib_row['fy']
            cx, cy = calib_row['cx'], calib_row['cy']
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
            
            R = np.array([[calib_row['r11'], calib_row['r12'], calib_row['r13']],
                         [calib_row['r21'], calib_row['r22'], calib_row['r23']],
                         [calib_row['r31'], calib_row['r32'], calib_row['r33']]])
            
            cam_pos = np.array([calib_row['cam_x'], calib_row['cam_y'], calib_row['cam_z']])
            
            # Get 3D position using selected method
            x_col = f'{self.column_prefix}_x'
            y_col = f'{self.column_prefix}_y'
            z_col = f'{self.column_prefix}_z'
            ball_3d = np.array([pos_row[x_col], pos_row[y_col], pos_row[z_col]])
            
            # Reproject to 2D
            x_reproj, y_reproj = self.reproject_3d_to_2d(ball_3d, K, R, cam_pos)
            
            if x_reproj is None:
                continue
            
            # Get original detection
            x_orig = det_row['center_x']
            y_orig = det_row['center_y']
            
            # Compute error
            error_x = x_reproj - x_orig
            error_y = y_reproj - y_orig
            error_euclidean = np.sqrt(error_x**2 + error_y**2)
            
            errors.append(error_euclidean)
            recon_errors.append(recon_error)
            
            self.results.append({
                'frame': frame,
                'x_original': x_orig,
                'y_original': y_orig,
                'x_reprojected': x_reproj,
                'y_reprojected': y_reproj,
                'error_x': error_x,
                'error_y': error_y,
                'error_euclidean': error_euclidean,
                'recon_error': recon_error,  # Store calibration reconstruction error
                'air_ground': pos_row.get('air_ground', 'unknown'),
                'diameter': det_row.get('diameter', 0)
            })
        
        print(f"  ✓ Computed reprojection errors for {len(self.results)} frames")
        
        # Compute statistics (both unweighted and weighted)
        if len(errors) > 0:
            errors = np.array(errors)
            recon_errors = np.array(recon_errors)
            
            # Compute confidence weights from reconstruction errors
            # Weight = exp(-recon_error / mean_recon_error)
            # High recon_error → low weight (less penalty)
            # Low recon_error → high weight (more penalty)
            mean_recon = np.mean(recon_errors)
            weights = np.exp(-recon_errors / (mean_recon + 1e-6))
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            
            # Unweighted statistics
            self.stats = {
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'rmse': np.sqrt(np.mean(errors**2)),
                # Weighted statistics
                'weighted_mean_error': np.sum(weights * errors),
                'weighted_rmse': np.sqrt(np.sum(weights * errors**2)),
                'mean_recon_error': mean_recon,
                'median_recon_error': np.median(recon_errors)
            }
            
            print("="*60)
            print("REPROJECTION ERROR STATISTICS")
            print("="*60)
            print("Unweighted Metrics:")
            print(f"  Mean: {self.stats['mean_error']:.2f} px | Median: {self.stats['median_error']:.2f} px | RMSE: {self.stats['rmse']:.2f} px")
            print(f"  Range: [{self.stats['min_error']:.2f}, {self.stats['max_error']:.2f}] px | Std: {self.stats['std_error']:.2f} px")
            
            print("Weighted Metrics (calibration-aware):")
            print(f"  Weighted Mean: {self.stats['weighted_mean_error']:.2f} px | Weighted RMSE: {self.stats['weighted_rmse']:.2f} px")
            
            print("Calibration Quality:")
            print(f"  Mean Recon Error: {self.stats['mean_recon_error']:.4f} px | Median: {self.stats['median_recon_error']:.4f} px")
            print("="*60)
            
            # Compute per-segment statistics
            df = pd.DataFrame(self.results)
            for label in ['ground', 'air']:
                mask = df['air_ground'] == label
                if mask.any():
                    segment_errors = df[mask]['error_euclidean'].values
                    print(f"{label.upper()}: Mean={np.mean(segment_errors):.2f} px | Median={np.median(segment_errors):.2f} px | RMSE={np.sqrt(np.mean(segment_errors**2)):.2f} px")
        
        return len(self.results) > 0
    
    def save_results(self, output_csv):
        """Save reprojection results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"✓ Saved reprojection results to {output_csv}")
    
    def plot_results(self, output_dir, video_frame_path=None):
        """Generate visualization plots"""
        if len(self.results) == 0:
            print("No results to plot!")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        print("Generating visualization plots...")
        
        # 1. Reprojection error over frames
        self._plot_error_over_frames(df, output_dir)
        
        # 2. Error distribution
        self._plot_error_distribution(df, output_dir)
        
        # 3. 2D comparison scatter plot
        self._plot_2d_comparison(df, output_dir)
        
        # 4. Error vector field
        self._plot_error_vectors(df, output_dir)
        
        # 5. Weighted loss analysis (recon_error vs reprojection error)
        self._plot_weighted_loss_analysis(df, output_dir)
        
        # 6. Sample frame comparison (if video frame provided)
        if video_frame_path:
            self._plot_sample_frame_comparison(df, video_frame_path, output_dir)
        
        print(f"  ✓ All plots saved to {output_dir}")
    
    def _plot_error_over_frames(self, df, output_dir):
        """Plot reprojection error over frames"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        frames = df['frame'].values
        errors = df['error_euclidean'].values
        color_map = {'ground': 'saddlebrown', 'air': 'dodgerblue', 'unknown': 'gray'}
        
        # Euclidean error
        ax = axes[0]
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                ax.plot(frames[mask], errors[mask], 'o-', color=color_map[label],
                       label=label.capitalize(), alpha=0.7, markersize=4)
        
        ax.axhline(y=self.stats['mean_error'], color='red', linestyle='--',
                  label=f"Mean: {self.stats['mean_error']:.2f} px", linewidth=2)
        ax.axhline(y=self.stats['median_error'], color='orange', linestyle=':',
                  label=f"Median: {self.stats['median_error']:.2f} px", linewidth=2)
        
        ax.set_ylabel('Reprojection Error (pixels)', fontweight='bold', fontsize=12)
        ax.set_title('Reprojection Error Over Frames', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Error components (X and Y)
        ax = axes[1]
        ax.plot(frames, df['error_x'].values, 'o-', label='Error X', alpha=0.7, markersize=3)
        ax.plot(frames, df['error_y'].values, 's-', label='Error Y', alpha=0.7, markersize=3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        ax.set_xlabel('Frame', fontweight='bold', fontsize=12)
        ax.set_ylabel('Error Components (pixels)', fontweight='bold', fontsize=12)
        ax.set_title('Reprojection Error Components (X, Y)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_error_frames.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, df, output_dir):
        """Plot error distribution histogram"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        errors = df['error_euclidean'].values
        
        # Histogram
        ax = axes[0]
        ax.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=self.stats['mean_error'], color='red', linestyle='--',
                  label=f"Mean: {self.stats['mean_error']:.2f} px", linewidth=2)
        ax.axvline(x=self.stats['median_error'], color='orange', linestyle=':',
                  label=f"Median: {self.stats['median_error']:.2f} px", linewidth=2)
        
        ax.set_xlabel('Reprojection Error (pixels)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title('Reprojection Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Box plot by segment
        ax = axes[1]
        ground_errors = df[df['air_ground'] == 'ground']['error_euclidean'].values
        air_errors = df[df['air_ground'] == 'air']['error_euclidean'].values
        
        box_data = []
        labels = []
        if len(ground_errors) > 0:
            box_data.append(ground_errors)
            labels.append('Ground')
        if len(air_errors) > 0:
            box_data.append(air_errors)
            labels.append('Air')
        
        if len(box_data) > 0:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            colors = ['saddlebrown', 'dodgerblue']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Reprojection Error (pixels)', fontweight='bold', fontsize=12)
        ax.set_title('Error Distribution by Segment Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_2d_comparison(self, df, output_dir):
        """Plot 2D comparison of original vs reprojected positions"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        color_map = {'ground': 'saddlebrown', 'air': 'dodgerblue', 'unknown': 'gray'}
        
        # Scatter plot: Original vs Reprojected
        ax = axes[0]
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                # Original positions
                ax.scatter(df[mask]['x_original'], df[mask]['y_original'],
                          c=color_map[label], s=50, alpha=0.6, marker='o',
                          label=f'{label.capitalize()} (Original)', edgecolors='black', linewidths=1)
                
                # Reprojected positions
                ax.scatter(df[mask]['x_reprojected'], df[mask]['y_reprojected'],
                          c=color_map[label], s=50, alpha=0.6, marker='x',
                          label=f'{label.capitalize()} (Reprojected)', linewidths=2)
        
        ax.set_xlabel('X (pixels)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontweight='bold', fontsize=12)
        ax.set_title('Original vs Reprojected 2D Positions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)  # Inverted: origin at top-left
        
        # Error magnitude as color
        ax = axes[1]
        scatter = ax.scatter(df['x_original'], df['y_original'],
                           c=df['error_euclidean'], cmap='hot', s=100, alpha=0.7,
                           edgecolors='black', linewidths=1)
        
        # Draw error vectors
        for _, row in df.iterrows():
            ax.arrow(row['x_original'], row['y_original'],
                    row['error_x'], row['error_y'],
                    head_width=5, head_length=5, fc='blue', ec='blue',
                    alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('X (pixels)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontweight='bold', fontsize=12)
        ax.set_title('Reprojection Error Magnitude', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)  # Inverted: origin at top-left
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error (pixels)', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_2d_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_error_vectors(self, df, output_dir):
        """Plot error vectors in 2D image space"""
        fig, ax = plt.subplots(figsize=(16, 9))
        
        color_map = {'ground': 'saddlebrown', 'air': 'dodgerblue', 'unknown': 'gray'}
        
        # Plot original positions
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                ax.scatter(df[mask]['x_original'], df[mask]['y_original'],
                          c=color_map[label], s=80, alpha=0.6, marker='o',
                          label=f'{label.capitalize()}', edgecolors='black', linewidths=1.5)
        
        # Draw error vectors (scaled for visibility)
        scale_factor = 5.0  # Amplify errors for visibility
        for _, row in df.iterrows():
            ax.arrow(row['x_original'], row['y_original'],
                    row['error_x'] * scale_factor, row['error_y'] * scale_factor,
                    head_width=8, head_length=8, fc='red', ec='red',
                    alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('X (pixels)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontweight='bold', fontsize=12)
        ax.set_title(f'Reprojection Error Vectors (scaled {scale_factor}x for visibility)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)  # Inverted: origin at top-left
        
        # Add text annotation
        ax.text(0.02, 0.98, f"Mean Error: {self.stats['mean_error']:.2f} px\nRMSE: {self.stats['rmse']:.2f} px",
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_error_vectors.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_weighted_loss_analysis(self, df, output_dir):
        """
        Plot weighted loss analysis showing relationship between 
        calibration reconstruction error and reprojection error
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weighted Loss Analysis: Calibration Quality vs Reprojection Error', 
                     fontsize=16, fontweight='bold')
        
        frames = df['frame'].values
        recon_errors = df['recon_error'].values
        reproj_errors = df['error_euclidean'].values
        
        # Compute weights
        mean_recon = np.mean(recon_errors)
        weights = np.exp(-recon_errors / (mean_recon + 1e-6))
        weights = weights / np.sum(weights)  # Normalize
        
        # Compute weighted errors
        weighted_errors = weights * reproj_errors * len(reproj_errors)  # Scale back for visualization
        
        color_map = {'ground': 'saddlebrown', 'air': 'dodgerblue', 'unknown': 'gray'}
        
        # Plot 1: Reconstruction error over frames
        ax = axes[0, 0]
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                ax.plot(frames[mask], recon_errors[mask], 'o-', color=color_map[label],
                       label=label.capitalize(), alpha=0.7, markersize=4)
        
        ax.axhline(y=mean_recon, color='red', linestyle='--',
                  label=f"Mean: {mean_recon:.4f} px", linewidth=2)
        ax.set_xlabel('Frame', fontweight='bold', fontsize=11)
        ax.set_ylabel('Calibration Recon Error (pixels)', fontweight='bold', fontsize=11)
        ax.set_title('Camera Calibration Quality (from PnLCalib)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reprojection error vs Reconstruction error (scatter)
        ax = axes[0, 1]
        scatter = ax.scatter(recon_errors, reproj_errors, c=frames, cmap='viridis',
                           s=60, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        # Add trend line
        z = np.polyfit(recon_errors, reproj_errors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(recon_errors.min(), recon_errors.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Calibration Recon Error (pixels)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Reprojection Error (pixels)', fontweight='bold', fontsize=11)
        ax.set_title('Correlation: Calibration Quality vs Reprojection Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame', fontweight='bold')
        
        # Plot 3: Confidence weights over frames
        ax = axes[1, 0]
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                ax.plot(frames[mask], weights[mask] * len(weights), 'o-', color=color_map[label],
                       label=label.capitalize(), alpha=0.7, markersize=4)
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Uniform weight')
        ax.set_xlabel('Frame', fontweight='bold', fontsize=11)
        ax.set_ylabel('Confidence Weight (normalized)', fontweight='bold', fontsize=11)
        ax.set_title('Frame Confidence Weights (High = Good Calibration)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Weighted vs Unweighted errors
        ax = axes[1, 1]
        for label in ['ground', 'air']:
            mask = df['air_ground'] == label
            if mask.any():
                ax.plot(frames[mask], reproj_errors[mask], 'o-', color=color_map[label],
                       alpha=0.4, markersize=3, linewidth=1, label=f'{label.capitalize()} (unweighted)')
                ax.plot(frames[mask], weighted_errors[mask], 's-', color=color_map[label],
                       alpha=0.8, markersize=4, linewidth=2, label=f'{label.capitalize()} (weighted)')
        
        ax.axhline(y=self.stats['mean_error'], color='red', linestyle='--',
                  label=f"Unweighted Mean: {self.stats['mean_error']:.2f} px", linewidth=1.5, alpha=0.5)
        ax.axhline(y=self.stats['weighted_mean_error'], color='darkred', linestyle='-',
                  label=f"Weighted Mean: {self.stats['weighted_mean_error']:.2f} px", linewidth=2)
        
        ax.set_xlabel('Frame', fontweight='bold', fontsize=11)
        ax.set_ylabel('Error (pixels)', fontweight='bold', fontsize=11)
        ax.set_title('Weighted vs Unweighted Reprojection Errors', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        summary_text = (
            f"Weighted Loss Summary:\n"
            f"  Unweighted RMSE: {self.stats['rmse']:.2f} px\n"
            f"  Weighted RMSE:   {self.stats['weighted_rmse']:.2f} px\n"
            f"  Improvement:     {((self.stats['rmse'] - self.stats['weighted_rmse']) / self.stats['rmse'] * 100):.1f}%"
        )
        ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reprojection_weighted_loss_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_frame_comparison(self, df, video_frame_path, output_dir):
        """Plot sample frame with original and reprojected positions"""
        # This is a placeholder - would need actual video frames
        pass


def main():
    parser = argparse.ArgumentParser(description="3D to 2D Reprojection Validation")
    parser.add_argument("--positions", type=str, required=True,
                       help="Path to 3D positions CSV")
    parser.add_argument("--calibration", type=str, required=True,
                       help="Path to camera calibration CSV")
    parser.add_argument("--detections", type=str, required=True,
                       help="Path to original detections CSV")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV path for reprojection results")
    parser.add_argument("--method", type=str, default='best',
                       choices=['best', 'polynomial', 'bezier', 'exponential', 'mu_s'],
                       help="Trajectory fitting method to validate (default: best)")
    parser.add_argument("--plot", action='store_true',
                       help="Generate visualization plots")
    parser.add_argument("--plot-output", type=str, default=None,
                       help="Directory for saving plots")
    
    args = parser.parse_args()
    
    print("="*60)
    print("3D TO 2D REPROJECTION VALIDATION")
    print("="*60)
    print(f"Validating method: {args.method.upper()}")
    print()
    
    # Initialize validator
    validator = ReprojectionValidator(method=args.method)
    
    # Load data
    if not validator.load_data(args.positions, args.calibration, args.detections):
        print("Error: Could not load data!")
        return 1
    
    # Compute reprojection errors
    if not validator.compute_reprojection_errors():
        print("Error: Could not compute reprojection errors!")
        return 1
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validator.save_results(output_path)
    
    # Generate plots
    if args.plot:
        plot_dir = Path(args.plot_output) if args.plot_output else output_path.parent
        validator.plot_results(plot_dir)
    
    print("\n" + "="*60)
    print("✓ REPROJECTION VALIDATION COMPLETE!")
    print("="*60)
    print(f"Output CSV: {args.output}")
    if args.plot:
        print(f"Plots: {plot_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

