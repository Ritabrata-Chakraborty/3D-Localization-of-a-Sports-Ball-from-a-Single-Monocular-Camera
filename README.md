# Ball Tracking and 3D Localization Pipeline - Complete Documentation

![final_video](https://github.com/user-attachments/assets/c7d0168a-74ac-4f67-b1e4-506fd124938d)


## Table of Contents
- [Overview](#overview)
- [Visual Examples](#visual-examples)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Stage-by-Stage Guide](#stage-by-stage-guide)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [Technical Details](#technical-details)
- [References](#references)

---

## Overview

This is a complete computer vision pipeline for detecting, tracking, and localizing a ball in 3D from monocular video footage. The system uses state-of-the-art object detection (YOLO), camera calibration from field markings (PnLCalib), and physics-based trajectory fitting to achieve accurate 3D ball localization.

---

## Visual Examples

### Final Output: 3D Trajectory Visualization

![3D Trajectory Plot](results/V2_1/stage5_finalization/trajectories_3d.png)

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/Ritabrata-Chakraborty/3D-Localization-of-a-Sports-Ball-from-a-Single-Monocular-Camera.git

cd 3D-Localization-of-a-Sports-Ball-from-a-Single-Monocular-Camera
```

### Step 2: Install PnLCalib & Create Conda Environment

```bash
git clone https://github.com/mguti97/PnLCalib

cd PnLCalib

conda env create -f PnLCalib.yml

conda activate PnLCalib

# Download pre-trained weights (if not included)
# The weights should be placed in:
# - PnLCalib/SV_kp (keypoint detection model)
# - PnLCalib/SV_lines (line detection model)

cd ..
```

**Note**: PnLCalib is a critical dependency for camera calibration. Ensure the model weights are properly downloaded and placed in the correct directories.

### Step 3: Train YOLO on Football Dataset

```bash
cd YOLO

# Download Dataset
bash yolo.sh download

# Train YOLO
bash yolo.sh train

cd ..
```

### Step 4: Prepare Video Data

```bash
# Download Video
bash scripts/video.sh download

# Extract Frames
bash scripts/video.sh extract
```

---

## Pipeline Architecture

The pipeline consists of 5 sequential stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Ball Detection (2D)                 │
│  Input: Video frames                                            │
│  Output: Ball positions (x, y, diameter) + air/ground labels    │
│  Methods: YOLO + Hough Circle + Temporal smoothing (3 methods)  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Stage 2: Camera Calibration                    │
│  Input: Video frames                                            │
│  Output: Camera intrinsics + extrinsics per frame               │
│  Methods: PnLCalib + Quaternion smoothing (2 methods)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Stage 3: 3D Localization + Fitting                 │
│  Input: 2D detections + Camera calibration                      │
│  Output: 3D ball positions + Fitted trajectories                │
│  Methods: Van Zandycke + Physics + Polynomial fitting           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 4: Validation                          │
│  Input: 3D positions + Camera calibration + 2D detections       │
│  Output: Reprojection errors + Validation metrics               │
│  Methods: 2D reprojection + Weighted loss analysis              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Stage 5: Finalization                         │
│  Input: All previous stage outputs                              │
│  Output: Annotated video + 3D plots + Final CSV                 │
│  Methods: Frame annotation + Video compilation + Visualization  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Guide

### Stage 1: Camera Calibration

**Purpose**: Extract camera intrinsic and extrinsic parameters using field line detection.

**Commands**:
```bash
# Run calibration
./scripts/calibrate.sh calibrate

# Apply smoothing (choose method1 or method2)
./scripts/calibrate.sh smooth method1

# Generate visualization
./scripts/calibrate.sh visualize
```

**Smoothing Methods**:
- **Method 1**: Global 2nd degree polynomial - For stationary cameras (recommended)
- **Method 2**: Piecewise 2nd degree polynomial - For moving cameras

---

### Stage 2: Ball Detection (2D)

**Purpose**: Detect the ball in each video frame and extract 2D position, diameter, and air/ground state.

**Commands**:
```bash
# Run detection
./scripts/detect.sh detect

# Apply smoothing (choose method 1, 2, or 3)
./scripts/detect.sh smooth 2

# Generate visualization
./scripts/detect.sh visualize
```

**Smoothing Methods**:
- **Method 1**: Piecewise Quadratic - Strict monotonicity, good for simple trajectories
- **Method 2**: Piecewise Cubic Spline - Maximum smoothness, best for most cases (recommended)
- **Method 3**: Bidirectional Exponential - Very smooth, no lag, good for noisy data

---

### Stage 3: 3D Localization + Trajectory Fitting

**Purpose**: Convert 2D detections to 3D world coordinates and fit polynomial trajectories.

**Command**:
```bash
./scripts/localize_3d.sh
```

**Process**:
1. **Geometric Localization**: Van Zandycke diameter-based depth estimation
2. **Physics Corrections**: Parabolic trajectory fitting + ground plane constraints
3. **Temporal Continuity**: ICP-inspired iterative refinement
4. **Polynomial Fitting**: Degree-2 polynomials fitted to air frames only

**Trajectory Fitting Methods**:
- Polynomial (degree-2 parabola)
- Bézier (cubic with 4 control points)
- Exponential decay
- Mu-S curve (sigmoid-modulated)

---

### Stage 4: Validation

**Purpose**: Validate 3D trajectories by reprojecting to 2D and comparing with original detections.

**Command**:
```bash
# Validate specific method
./scripts/validate_reprojection.sh --method bezier
```
---

### Stage 5: Finalization

**Purpose**: Generate annotated video with comprehensive overlays and create final visualizations.

**Commands**:
```bash
# specify custom smoothing methods
./scripts/finalize.sh --detection-method 1 --calibration-method 2
```

### Coordinate System

**World Coordinates (PnLCalib convention)**:
- Origin: Field center (midfield line)
- X-axis: Along field length (-52.5 to +52.5 m for 105m field)
- Y-axis: Along field width (-34 to +34 m for 68m field)
- Z-axis: Vertical (negative = up, z=0 is ground, z<0 is airborne)
- Field: Standard football field (105m × 68m)

**Note**: Visualizations invert the Z-axis for intuitive viewing (positive height).

---

## Output Files

### Complete Directory Structure

```
results/V2_1/
├── stage1_detection/
│   ├── detections.csv                              # Raw detections
│   ├── detections_smoothed_method1_diameter.csv    # Smoothed (method 1)
│   ├── detections_smoothed_method2_diameter.csv    # Smoothed (method 2)
│   ├── detections_smoothed_method3_diameter.csv    # Smoothed (method 3)
│   ├── ball_detection_visualization.png            # Analysis plots
│   └── annotated_frames/                           # Annotated images
│
├── stage2_calibration/
│   ├── camera_calibration.csv                      # Raw calibration
│   ├── camera_calibration_smoothed_method1.csv     # Smoothed (method 1)
│   ├── camera_calibration_smoothed_method2.csv     # Smoothed (method 2)
│   ├── camera_calibration_stats.csv                # Statistics
│   ├── camera_calibration_visualization.png        # Parameter plots
│   └── annotated_frames_3d/                        # Field line visualizations
│
├── stage3_localization_3d/
│   ├── ball_3d_positions.csv                       # Raw 3D positions
│   ├── ball_3d_positions_fitted.csv                # Fitted trajectories
│   ├── trajectory_equations.json                   # Polynomial equations
│   ├── ball_3d_trajectory.png                      # 3D visualization
│   ├── ball_3d_scatter.png                         # 3D scatter plot
│   ├── ball_top_view.png                           # Top view (X-Y)
│   ├── ball_side_view_x_z.png                      # Side view (X-Z)
│   ├── ball_side_view_y_z.png                      # Side view (Y-Z)
│   └── ball_velocity_analysis.png                  # Velocity plots
│
├── stage4_validation/
│   ├── reprojection_validation_bezier.csv          # Validation results
│   ├── reprojection_error_frames.png               # Error over time
│   ├── reprojection_error_distribution.png         # Error histogram
│   ├── reprojection_2d_comparison.png              # Original vs reprojected
│   └── reprojection_weighted_loss_analysis.png     # Quality analysis
│
└── stage5_finalization/
    ├── final_video.mp4                             # Annotated video
    ├── trajectories_3d.png                         # 3D trajectory plot
    ├── final_positions.csv                         # Combined CSV
    └── annotated_frames/                           # Individual frames
```

---

## Future Work

1. **End-to-End Learning**
   - Direct 3D position estimation from images
   - Reduced calibration dependency

2. **Advanced Physics Models**
   - Wind effects and air resistance
   - Ball spin and Magnus force modeling

3. **Multi-camera Support**
   - Data fusion from multiple angles
   - Improved accuracy through triangulation

4. **Enhanced Detection**
   - More accurate diameter estimation
   - Blur detection

5. **Extended Sports Support**
   - Automatic field detection
   - Basketball, tennis, and other sports
   - Moving camera tracking

---

## License & Contact

**Author:** [Ritabrata Chakraborty](https://ritabrata-chakraborty.github.io/Portfolio/) \
**Date:** October 2025

For questions, issues, or contributions, please open an issue on the repository.
