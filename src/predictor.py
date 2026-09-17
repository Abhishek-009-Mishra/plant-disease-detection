"""
Plant disease prediction module.

Loads the trained MobileNetV2 model, performs inference on a
leaf image, displays the top predictions, and stores the
prediction in the SQLite history database.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import config
from src.database import PredictionDatabase


class PlantDiseasePredictor:
    """Predict plant diseases using the trained model."""

    def __init__(self):
        self.class_names = self._load_class_names()

        print("Loaded 38 disease classes.")
        print("Loading trained model...")

        if not Path(config.MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Trained model not found:\n{config.MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(
            config.MODEL_PATH
        )

        print("Model loaded successfully from:")
        print(config.MODEL_PATH)

        self.database = PredictionDatabase()

    def _load_class_names(self):
        """Load class names from JSON."""

        path = Path(config.CLASS_NAMES_PATH)

        if not path.exists():
            raise FileNotFoundError(
                f"Class names file not found:\n{path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _preprocess_image(self, image_path):
        """Load and preprocess an image for MobileNetV2."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise ValueError(
                f"Unable to read image:\n{image_path}"
            ) from error

        image = image.resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        )

        image_array = np.array(image, dtype=np.float32)

        # Convert pixel values from 0-255 to 0-1.
        image_array = image_array / 255.0

        # Add batch dimension.
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    @staticmethod
    def _parse_class_name(class_name):
        """Convert PlantVillage class name into plant and disease."""

        parts = class_name.split("___", 1)

        plant = parts[0].replace("_", " ")

        if len(parts) > 1:
            disease = parts[1].replace("_", " ")
        else:
            disease = "Unknown"

        return plant, disease

    def predict(self, image_path, save_history=True):
        """
        Predict disease from an image.

        Returns a dictionary containing the prediction results.
        """

        image_array = self._preprocess_image(image_path)

        predictions = self.model.predict(
            image_array,
            verbose=0
        )[0]

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        top_predictions = []

        for index in top_indices:
            class_name = self.class_names[index]
            confidence = float(predictions[index])

            plant, disease = self._parse_class_name(
                class_name
            )

            top_predictions.append({
                "class_name": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": confidence
            })

        best_prediction = top_predictions[0]

        result = {
            "image_path": str(Path(image_path)),
            "plant": best_prediction["plant"],
            "disease": best_prediction["disease"],
            "confidence": best_prediction["confidence"],
            "top_predictions": top_predictions
        }

        # Save prediction to SQLite.
        if save_history:
            prediction_id = self.database.save_prediction(
                image_path=result["image_path"],
                plant=result["plant"],
                disease=result["disease"],
                confidence=result["confidence"],
                top_predictions=top_predictions
            )

            result["prediction_id"] = prediction_id

        return result

    def display_prediction(self, result):
        """Display prediction results in the terminal."""

        print("\n" + "=" * 60)
        print("PLANT DISEASE PREDICTION")
        print("=" * 60)

        print(f"\nImage      : {result['image_path']}")
        print(f"Plant      : {result['plant']}")
        print(f"Disease    : {result['disease']}")
        print(
            f"Confidence : "
            f"{result['confidence'] * 100:.2f}%"
        )

        print("\n" + "-" * 60)
        print("TOP PREDICTIONS")
        print("-" * 60)

        for rank, prediction in enumerate(
            result["top_predictions"],
            start=1
        ):
            print(
                f"{rank}. "
                f"{prediction['plant']} - "
                f"{prediction['disease']} "
                f"({prediction['confidence'] * 100:.2f}%)"
            )

        if "prediction_id" in result:
            print(
                f"\nPrediction saved to history "
                f"(ID: {result['prediction_id']})"
            )

        print("\n" + "=" * 60)


def predict_image(image_path):
    """Convenience function used by main.py."""

    predictor = PlantDiseasePredictor()

    result = predictor.predict(
        image_path,
        save_history=True
    )

    predictor.display_prediction(result)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.predictor "
            '"path/to/image.jpg"'
        )
        raise SystemExit(1)

    predict_image(sys.argv[1])