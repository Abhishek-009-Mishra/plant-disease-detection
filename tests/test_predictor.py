from pathlib import Path

import pytest

from src.predictor import PlantDiseasePredictor


MODEL_PATH = Path(
    r"D:\Computer Vision\plant-disease-detection\models\plant_disease_model.keras"
)

TEST_IMAGE = Path(
    r"D:\Computer Vision\plant-disease-detection\test_images\tomato.jpg"
)


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not available"
)
@pytest.mark.skipif(
    not TEST_IMAGE.exists(),
    reason="Test image not available"
)
def test_prediction():
    predictor = PlantDiseasePredictor()

    result = predictor.predict(
        str(TEST_IMAGE),
        save_history=False
    )

    assert "plant" in result
    assert "disease" in result
    assert "confidence" in result
    assert "top_predictions" in result

    assert 0 <= result["confidence"] <= 1

    assert len(result["top_predictions"]) == 3


def test_invalid_image():
    predictor = PlantDiseasePredictor()

    with pytest.raises((FileNotFoundError, ValueError)):
        predictor.predict(
            r"D:\Computer Vision\plant-disease-detection\test_images\does_not_exist.jpg",
            save_history=False
        )