"""Extract frames from video files"""

import os
import sys
import json
import subprocess
import cv2
import argparse


def get_video_metadata(video_path):
    """
    Get video metadata using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict with fps, width, height, duration
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        video_path
    ]
    
    try:
        out = subprocess.check_output(cmd).decode("utf-8")
        info = json.loads(out)
        
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                r = stream.get("r_frame_rate", "25/1")
                try:
                    fps = eval(r)
                except Exception:
                    fps = float(stream.get("avg_frame_rate", 25.0))
                
                return {
                    "fps": float(fps),
                    "width": int(stream.get("width", 0)),
                    "height": int(stream.get("height", 0)),
                    "duration": float(stream.get("duration", info.get("format", {}).get("duration", 0)))
                }
    except FileNotFoundError:
        print("✗ Error: ffprobe not found. Install ffmpeg.")
        return {}
    except Exception as e:
        print(f"✗ Error getting metadata: {e}")
        return {}
    
    return {}


def extract_frames(video_path, out_dir, every_n_frames=1, start_sec=0, end_sec=None, max_frames=None):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        out_dir: Output directory for frames
        every_n_frames: Extract every n-th frame (1 = all frames)
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = end of video)
        max_frames: Maximum number of frames to extract (None = all)
    """
    if not os.path.exists(video_path):
        print(f"✗ Error: Video file '{video_path}' does not exist")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Error: Unable to open video file '{video_path}'")
        return
    
    meta = get_video_metadata(video_path)
    fps = meta.get("fps", 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames
    
    print(f"Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {meta.get('width', 0)}x{meta.get('height', 0)}")
    print(f"  Duration: {meta.get('duration', 0):.2f}s")
    print(f"\nExtracting frames {start_frame} to {end_frame} (every {every_n_frames})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    idx = start_frame
    saved = 0
    
    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if ((idx - start_frame) % every_n_frames) == 0:
            fname = os.path.join(out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
            
            if saved % 100 == 0:
                print(f"  Extracted {saved} frames...")
            
            if max_frames and saved >= max_frames:
                break
        
        idx += 1
    
    cap.release()
    print(f"✓ Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--out_dir", type=str, default="frames",
                       help="Output directory for frames")
    parser.add_argument("--every_n_frames", type=int, default=1,
                       help="Extract every n-th frame (default: 1)")
    parser.add_argument("--start_sec", type=float, default=0,
                       help="Start time in seconds (default: 0)")
    parser.add_argument("--end_sec", type=float, default=None,
                       help="End time in seconds (default: end of video)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to extract")
    
    args = parser.parse_args()
    
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    out_dir = os.path.join(args.out_dir, video_name)
    
    extract_frames(
        args.video_path,
        out_dir,
        args.every_n_frames,
        args.start_sec,
        args.end_sec,
        args.max_frames
    )

