#!/bin/bash
# Simplified: only the important python3 commands

cmd="$1"

if [ "$cmd" = "download" ]; then
    python3 ./video_tools/download.py ./downloads/video_links.txt --output ./downloads/videos
elif [ "$cmd" = "extract" ]; then
    python3 ./video_tools/extract.py \
        ./downloads/videos/V2_h264.mp4 \
        --out_dir ./downloads/frames \
        --every_n_frames 1 \
        --start_sec 0
else
    echo "Usage: $0 {download|extract}"
    exit 1
fi
