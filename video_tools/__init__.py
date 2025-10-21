"""Video Tools - Download and extract frames from videos"""

from .download import download_video
from .extract import extract_frames, get_video_metadata

__version__ = "1.0.0"
__all__ = ['download_video', 'extract_frames', 'get_video_metadata']

