"""
augmentation.py
----------------
Data augmentation for the TRAINING split only.

Why these specific augmentations and not others:

- Random horizontal AND vertical flip: leaf photos in PlantVillage have no
  canonical "up" orientation (a leaf photographed rotated 90 or 180 degrees
  is still a valid, correctly-labeled leaf), so flipping never creates an
  unrealistic or mislabeled example.
- Random rotation (small angle): same reasoning as flips - leaves are
  photographed at arbitrary angles in the field.
- Random zoom: simulates the camera being closer/farther from the leaf.
- Random contrast / brightness: simulates different lighting conditions
  across capture sessions/devices, which is a real source of variation in
  the original PlantVillage capture batches.

Explicitly NOT used:
- Large rotations/shears/perspective warps or color-channel shuffling:
  these can distort or discolor lesion patterns in ways that could change
  what disease the image actually looks like, which risks teaching the
  model incorrect texture/color cues for a disease it doesn't have.

Implemented as a small stack of `tf.keras.layers` so it can be:
  (a) inserted directly into the tf.data pipeline via `.map()`, AND
  (b) later attached as the first layers of the trained model (Phase 4),
      which is the standard Keras pattern - augmentation layers are
      no-ops in inference mode, so training and inference share one
      code path with zero risk of applying augmentation at prediction time.
"""

import tensorflow as tf
from tensorflow.keras import layers

from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)


def build_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Build the augmentation stack as a Keras Sequential model of
    preprocessing layers. `layers.Random*` layers are ONLY active when
    called with `training=True`, so this is safe to keep attached to a
    model used for inference later.
    """
    aug_layers = []

    if config.AUG_HORIZONTAL_FLIP and config.AUG_VERTICAL_FLIP:
        aug_layers.append(layers.RandomFlip("horizontal_and_vertical"))
    elif config.AUG_HORIZONTAL_FLIP:
        aug_layers.append(layers.RandomFlip("horizontal"))
    elif config.AUG_VERTICAL_FLIP:
        aug_layers.append(layers.RandomFlip("vertical"))

    if config.AUG_ROTATION_FACTOR > 0:
        aug_layers.append(layers.RandomRotation(config.AUG_ROTATION_FACTOR))

    if config.AUG_ZOOM_FACTOR > 0:
        aug_layers.append(layers.RandomZoom(config.AUG_ZOOM_FACTOR))

    if config.AUG_CONTRAST_FACTOR > 0:
        aug_layers.append(layers.RandomContrast(config.AUG_CONTRAST_FACTOR))

    if config.AUG_BRIGHTNESS_FACTOR > 0:
        aug_layers.append(layers.RandomBrightness(
            config.AUG_BRIGHTNESS_FACTOR, value_range=(0.0, 1.0)
        ))

    pipeline = tf.keras.Sequential(aug_layers, name="augmentation")
    logger.info("Built augmentation pipeline with %d layers.", len(aug_layers))
    return pipeline


def apply_augmentation(image: tf.Tensor, label: tf.Tensor, aug_pipeline: tf.keras.Sequential):
    """`.map()`-friendly wrapper: applies the augmentation stack to one image."""
    image = aug_pipeline(image, training=True)
    # Augmentation layers (esp. RandomContrast/Brightness) can push values
    # slightly outside [0, 1]; clip to keep the normalized range valid for
    # the model's expected input.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label
