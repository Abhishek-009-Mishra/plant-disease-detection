"""
SQLite database module for storing plant disease predictions.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import config


class PredictionDatabase:
    """Manage prediction history using SQLite."""

    def __init__(self, database_path=None):
        self.database_path = Path(
            database_path if database_path else config.DATABASE_PATH
        )

        # Make sure the parent directory exists
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.create_table()

    def get_connection(self):
        """Create and return a SQLite connection."""
        return sqlite3.connect(str(self.database_path))

    def create_table(self):
        """Create the predictions table if it does not exist."""

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    plant TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    top_predictions TEXT
                )
            """)

            connection.commit()

        finally:
            connection.close()

    def save_prediction(
        self,
        image_path,
        plant,
        disease,
        confidence,
        top_predictions=None
    ):
        """Save a prediction to the database."""

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            cursor.execute(
                """
                INSERT INTO predictions
                (
                    image_path,
                    timestamp,
                    plant,
                    disease,
                    confidence,
                    top_predictions
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(image_path),
                    timestamp,
                    plant,
                    disease,
                    float(confidence),
                    str(top_predictions)
                    if top_predictions
                    else ""
                )
            )

            connection.commit()

            return cursor.lastrowid

        finally:
            connection.close()

    def get_history(self, limit=20):
        """Return recent prediction history."""

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    image_path,
                    timestamp,
                    plant,
                    disease,
                    confidence,
                    top_predictions
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )

            return cursor.fetchall()

        finally:
            connection.close()

    def get_count(self):
        """Return total number of predictions."""

        connection = self.get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM predictions"
            )

            return cursor.fetchone()[0]

        finally:
            connection.close()


if __name__ == "__main__":
    database = PredictionDatabase()

    print("\nPrediction database initialized successfully.")
    print("Database:", database.database_path)
    print("Total predictions:", database.get_count())