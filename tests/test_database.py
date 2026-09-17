import tempfile
from pathlib import Path

from src.database import PredictionDatabase


def test_database_creation():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_predictions.db"

        db = PredictionDatabase(str(db_path))

        assert db_path.exists()


def test_save_and_retrieve_prediction():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_predictions.db"

        db = PredictionDatabase(str(db_path))

        prediction_id = db.save_prediction(
            image_path="test.jpg",
            plant="Tomato",
            disease="Early blight",
            confidence=0.92,
            top_predictions=[
                {
                    "class_name": "Tomato___Early_blight",
                    "confidence": 0.92,
                }
            ],
        )

        assert prediction_id > 0

        history = db.get_history(limit=10)

        assert len(history) == 1

        # get_history() returns tuples:
        # id, image_path, timestamp, plant, disease, confidence, top_predictions
        assert history[0][3] == "Tomato"
        assert history[0][4] == "Early blight"
        assert history[0][5] == 0.92