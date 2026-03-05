"""Reusable helpers extracted from lab_libero.ipynb."""

import cv2
import imageio
import numpy as np
import tempfile
from IPython.display import Video


def resize_with_pad(image, target_height, target_width):
    """Resize image to target dimensions with padding to preserve aspect ratio."""
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize while preserving aspect ratio.
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Symmetrically pad to target size.
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def convert_to_uint8(img):
    """Convert input image to uint8 in range [0, 255]."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _extract_frame(obs):
    """Extract a displayable image frame from an observation dictionary."""
    if isinstance(obs, dict):
        for key in ("agentview_image", "frontview_image", "robot0_eye_in_hand_image", "image"):
            if key in obs:
                return obs[key]
        for key, value in obs.items():
            if key.endswith("_image"):
                return value
    return None


def _normalize_frame(frame):
    """Normalize any numeric image-like frame into uint8 [0, 255]."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def make_video(frames):
    """Encode a list of frames into an embeddable MP4 video."""
    frames_norm = [_normalize_frame(f) for f in frames]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        mp4_path = tmp.name
    imageio.mimsave(mp4_path, frames_norm, fps=20, format="mp4")
    return Video(mp4_path, embed=True)
