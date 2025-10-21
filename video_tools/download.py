"""Download videos from YouTube using yt-dlp"""

import os
import sys
import subprocess
import argparse


def download_video(url, out_dir="Videos", video_name=None):
    """
    Download video from URL using yt-dlp
    
    Args:
        url: Video URL (YouTube, etc.)
        out_dir: Output directory
        video_name: Custom video name (optional)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if video_name:
        out_template = os.path.join(out_dir, f"{video_name}.%(ext)s")
    else:
        out_template = os.path.join(out_dir, "%(title)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "-o", out_template,
        url
    ]
    
    print(f"Downloading: {url}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print(f"✓ Downloaded to {out_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        raise
    except FileNotFoundError:
        print("✗ Error: yt-dlp not found. Install with: pip install yt-dlp")
        raise


def download_from_file(links_file, out_dir="Videos"):
    """
    Download multiple videos from a text file
    
    Args:
        links_file: Path to text file with one URL per line
        out_dir: Output directory
    """
    if not os.path.exists(links_file):
        print(f"✗ Error: Links file '{links_file}' not found")
        return
    
    with open(links_file, "r") as f:
        links = f.readlines()
    
    links = [link.strip() for link in links if link.strip()]
    
    print(f"Found {len(links)} video(s) to download")
    
    for idx, link in enumerate(links):
        print(f"\n[{idx+1}/{len(links)}] Processing: {link}")
        video_name = f"video_{idx+1:03d}"
        
        try:
            download_video(link, out_dir, video_name)
        except Exception as e:
            print(f"✗ Failed to download video {idx+1}: {e}")
            continue
    
    print(f"\n✓ Download complete! Videos saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download videos from YouTube using yt-dlp")
    parser.add_argument("input", help="Video URL or path to text file with URLs")
    parser.add_argument("-o", "--output", default="Videos", help="Output directory (default: Videos)")
    parser.add_argument("-n", "--name", help="Custom video name (only for single URL)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        download_from_file(args.input, args.output)
    else:
        download_video(args.input, args.output, args.name)


if __name__ == "__main__":
    main()
