from pathlib import Path

import numpy as np

from src.preprocessing import (
    preprocess_image_cv2,
    validate_image,
)


TEST_IMAGE = Path(
    r"D:\Computer Vision\plant-disease-detection\test_images\tomato.jpg"
)


def test_preprocess_output_shape():
    result = preprocess_image_cv2(TEST_IMAGE)

    assert result.shape == (224, 224, 3)


def test_preprocess_pixel_range():
    result = preprocess_image_cv2(TEST_IMAGE)

    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0


def test_preprocess_output_dtype():
    result = preprocess_image_cv2(TEST_IMAGE)

    assert result.dtype == np.float32


def test_validate_valid_image():
    is_valid, reason = validate_image(TEST_IMAGE)

    assert is_valid is True
    assert reason is None


def test_validate_invalid_image():
    invalid_image = Path(
        r"D:\Computer Vision\plant-disease-detection\test_images\does_not_exist.jpg"
    )

    is_valid, reason = validate_image(invalid_image)

    assert is_valid is False
    assert reason is not None