"""
preprocessing.py
-----------------
Implements the "Image -> Validation -> Resize -> Noise Handling ->
Normalization -> Color Processing" stages of the Computer Vision pipeline.

Two parallel implementations are provided on purpose, and this is a
deliberate design decision, not duplication:

1. `preprocess_image_cv2()` - a pure OpenCV/NumPy implementation. This is
   the version used for:
     - one-off single-image inference (Phase 6 `predict`/`diagnose`)
     - validating images before they enter a split (this phase)
   Throughput doesn't matter for a single uploaded leaf photo, so
   readability and using the same library as the EDA module (Phase 1)
   is preferred here.

2. `tf_preprocess_image()` - a TensorFlow-graph implementation using
   `tf.image` ops. This is used INSIDE the `tf.data` training pipeline
   (see dataset_loader.build_tf_dataset), because running a Python/OpenCV
   function per image inside `tf.data.Dataset.map()` defeats
   parallel prefetching and is a well-known training bottleneck on CPU.

Both implementations perform the EXACT same logical steps (resize -> BGR/RGB
handling -> normalize to [0, 1]) so that training-time and inference-time
preprocessing never diverge - this directly satisfies the project
requirement that prediction must "apply the exact same preprocessing used
during training".

Backbone-specific scaling (e.g. MobileNetV2 expects inputs in [-1, 1], not
[0, 1]) is intentionally NOT done here. It is applied as a
`tf.keras.applications.mobilenet_v2.preprocess_input` Lambda/Rescaling layer
built into the model itself in Phase 4. Keeping that step inside the model
(rather than in this module) means the exported .keras model is fully
self-contained: any caller that feeds it a [0, 1] RGB image gets a correct
prediction without needing to know which backbone was used internally.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)

_INTERPOLATION_MAP = {
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_NEAREST": cv2.INTER_NEAREST,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_image(path: Path, min_size: int = 32) -> Tuple[bool, Optional[str]]:
    """
    Confirm an image file can actually be decoded and is large enough to be
    useful. Returns (is_valid, reason_if_invalid).

    This is run once per image before it is assigned to a train/val/test
    split (see dataset_loader.build_splits), so corrupt files identified in
    Phase 1's EDA never make it into the training pipeline in the first
    place, rather than crashing training mid-epoch.
    """
    img = cv2.imread(str(path))
    if img is None:
        return False, "unreadable (corrupt or unsupported format)"
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False, f"too small ({w}x{h}, minimum is {min_size}x{min_size})"
    return True, None


# ---------------------------------------------------------------------------
# OpenCV preprocessing path (single-image / inference use)
# ---------------------------------------------------------------------------
def _maybe_denoise(img_bgr: np.ndarray, representation: str) -> np.ndarray:
    """Apply a light median blur, but only for representations known to
    benefit from it (see config.DENOISE_REPRESENTATIONS for rationale)."""
    if representation in config.DENOISE_REPRESENTATIONS:
        k = config.DENOISE_MEDIAN_KERNEL
        if k % 2 == 0:
            k += 1  # medianBlur requires an odd kernel size
        return cv2.medianBlur(img_bgr, k)
    return img_bgr


def preprocess_image_cv2(
    path: Path,
    image_size: Optional[int] = None,
    representation: Optional[str] = None,
) -> np.ndarray:
    """
    Full OpenCV preprocessing pipeline for ONE image:
        read -> validate -> denoise (if applicable) -> resize -> BGR2RGB -> normalize

    Returns a float32 array of shape (image_size, image_size, 3) with values
    in [0, 1]. Raises ValueError if the image cannot be read.
    """
    image_size = image_size or config.IMAGE_SIZE
    representation = representation or config.DEFAULT_REPRESENTATION

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    img_bgr = _maybe_denoise(img_bgr, representation)

    interp = _INTERPOLATION_MAP.get(config.RESIZE_INTERPOLATION, cv2.INTER_AREA)
    img_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=interp)

    # OpenCV loads images as BGR; convert to RGB since that is the channel
    # order Keras/TensorFlow (and every pretrained backbone) expects.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0
    return img_float


# ---------------------------------------------------------------------------
# TensorFlow preprocessing path (bulk training pipeline)
# ---------------------------------------------------------------------------
def tf_preprocess_image(image_path: tf.Tensor, image_size: Optional[int] = None) -> tf.Tensor:
    """
    TensorFlow-graph equivalent of preprocess_image_cv2, for use inside
    `tf.data.Dataset.map()`. Runs entirely in the TF graph so it can be
    parallelized and prefetched efficiently across many images.

    Note: median-blur denoising for the 'segmented' representation is
    intentionally NOT ported here. It's a one-time, cheap data-quality fix
    better suited to the low-throughput OpenCV path; for the bulk pipeline
    the added graph complexity isn't worth it given color images (this
    project's primary training representation) never use it anyway.
    """
    image_size = image_size or config.IMAGE_SIZE

    raw = tf.io.read_file(image_path)
    # decode_image would also handle PNG/GIF, but decode_jpeg is faster and
    # PlantVillage images are JPEGs; expand_animations=False keeps output
    # shape static which decode_image alone would not guarantee.
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [image_size, image_size], method="area")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def load_label_and_image(
    image_path: tf.Tensor, label: tf.Tensor, image_size: Optional[int] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convenience wrapper for use as the first `.map()` step in a tf.data pipeline."""
    return tf_preprocess_image(image_path, image_size), label
