"""
config.py
---------
Single source of truth for all project configuration.

Why centralized config?
Scattering paths, hyperparameters, and API settings across multiple files
makes a project hard to maintain and easy to break (e.g. changing the image
size in one file but not another). Every other module in this project reads
its settings from this file only.

Configuration is loaded from environment variables (via a local .env file)
so that machine-specific values (like the Windows dataset path) and secrets
(like the Grok API key) never need to be hard-coded or committed to Git.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load variables from a local .env file, if present. This never overrides
# variables that are already set in the real environment.
load_dotenv()

# ---------------------------------------------------------------------------
# Project root (this file lives at <root>/src/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # NOTE: This is the ONE place you need to edit (or set the
    # PLANTVILLAGE_DATASET_ROOT environment variable) to point at your
    # local copy of the PlantVillage dataset.
    #
    # Expected structure:
    #   <DATASET_ROOT>/color/<Plant>___<Condition>/*.jpg
    #   <DATASET_ROOT>/grayscale/<Plant>___<Condition>/*.jpg
    #   <DATASET_ROOT>/segmented/<Plant>___<Condition>/*.jpg
    DATASET_ROOT: str = os.getenv(
        "PLANTVILLAGE_DATASET_ROOT",
        "D:\Computer Vision\plant-disease-detection\raw",  # e.g. "D:/PlantVillage" on Windows
    )

    # Which of these sub-folders we look for under DATASET_ROOT.
    SUPPORTED_REPRESENTATIONS: List[str] = field(
        default_factory=lambda: ["color", "grayscale", "segmented"]
    )

    # Representation used by default for training/analysis unless the user
    # passes --representation explicitly.
    DEFAULT_REPRESENTATION: str = os.getenv("DEFAULT_REPRESENTATION", "color")

    # Valid image extensions to look for inside class folders.
    IMAGE_EXTENSIONS: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    )

    # Class folders follow the PlantVillage convention "Plant___Condition".
    CLASS_NAME_SEPARATOR: str = "___"

    # ------------------------------------------------------------------
    # Image preprocessing / model input
    # ------------------------------------------------------------------
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "224"))  # MobileNetV2/EffNet default
    IMAGE_CHANNELS: int = 3

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    EPOCHS: int = int(os.getenv("EPOCHS", "20"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "1e-4"))
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "2"))

    # Split ratios must sum to 1.0. 70/15/15 is a common, defensible split
    # that leaves enough data for training while still giving statistically
    # meaningful validation/test sets given PlantVillage's class sizes.
    TRAIN_SPLIT: float = float(os.getenv("TRAIN_SPLIT", "0.70"))
    VAL_SPLIT: float = float(os.getenv("VAL_SPLIT", "0.15"))
    TEST_SPLIT: float = float(os.getenv("TEST_SPLIT", "0.15"))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    # OpenCV resize interpolation. INTER_AREA is generally best for
    # shrinking images (PlantVillage originals are >=256x256, larger than
    # our IMAGE_SIZE target), since it avoids the aliasing/moire artifacts
    # that nearest/linear interpolation can introduce when downscaling.
    RESIZE_INTERPOLATION: str = "INTER_AREA"

    # Mild denoising is only useful for the 'segmented' representation,
    # where PlantVillage's background-removal step can leave jagged,
    # noisy edges around the leaf mask. Color/grayscale leaf photos are
    # already clean lab images, so denoising them would blur away the
    # fine lesion texture that the model needs to tell diseases apart -
    # applying it there would actively hurt accuracy, not help it.
    DENOISE_REPRESENTATIONS: List[str] = field(default_factory=lambda: ["segmented"])
    DENOISE_MEDIAN_KERNEL: int = int(os.getenv("DENOISE_MEDIAN_KERNEL", "3"))

    # ------------------------------------------------------------------
    # Data augmentation (training split only - never applied to val/test)
    # ------------------------------------------------------------------
    AUG_ROTATION_FACTOR: float = float(os.getenv("AUG_ROTATION_FACTOR", "0.10"))   # ~36 deg
    AUG_ZOOM_FACTOR: float = float(os.getenv("AUG_ZOOM_FACTOR", "0.15"))
    AUG_CONTRAST_FACTOR: float = float(os.getenv("AUG_CONTRAST_FACTOR", "0.15"))
    AUG_BRIGHTNESS_FACTOR: float = float(os.getenv("AUG_BRIGHTNESS_FACTOR", "0.15"))
    AUG_HORIZONTAL_FLIP: bool = _get_bool("AUG_HORIZONTAL_FLIP", True)
    AUG_VERTICAL_FLIP: bool = _get_bool("AUG_VERTICAL_FLIP", True)

    # ------------------------------------------------------------------
    # Class imbalance handling
    # ------------------------------------------------------------------
    # 'balanced' mode (sklearn-style) reweights the loss so rare classes
    # count more per-sample, without duplicating any images on disk or
    # in memory. This is preferred over naive oversampling for this
    # project because PlantVillage's imbalance is mild-to-moderate
    # (not orders of magnitude), so reweighting is sufficient and keeps
    # every training epoch seeing only real, unmodified images.
    CLASS_WEIGHT_STRATEGY: str = os.getenv("CLASS_WEIGHT_STRATEGY", "balanced")

    # ------------------------------------------------------------------
    # Dataset splits (cached so train/evaluate always see the exact same
    # split - regenerating a random split independently in each phase
    # would silently leak test images into training)
    # ------------------------------------------------------------------
    SPLITS_DIR: str = str(PROJECT_ROOT / "data" / "splits")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # MobileNetV2 is chosen as the default backbone (see README for the
    # full rationale): it is small (~14MB), fast on CPU, and performs very
    # well on leaf-texture classification tasks such as PlantVillage.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "MobileNetV2")  # or "EfficientNetB0"
    MODEL_DIR: str = str(PROJECT_ROOT / "models")
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "plant_disease_model.keras")
    CLASS_NAMES_PATH: str = str(PROJECT_ROOT / "models" / "class_names.json")

    # ------------------------------------------------------------------
    # Reports / EDA output
    # ------------------------------------------------------------------
    REPORTS_DIR: str = str(PROJECT_ROOT / "reports")
    EDA_DIR: str = str(PROJECT_ROOT / "reports" / "eda")

    # ------------------------------------------------------------------
    # Database (prediction history)
    # ------------------------------------------------------------------
    DATABASE_PATH: str = str(PROJECT_ROOT / "data" / "predictions.db")

    # ------------------------------------------------------------------
    # RAG / Knowledge base
    # ------------------------------------------------------------------
    KNOWLEDGE_DIR: str = str(PROJECT_ROOT / "knowledge")
    VECTOR_DB_DIR: str = str(PROJECT_ROOT / "vector_db")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ------------------------------------------------------------------
    ## ------------------------------------------------------------------
    # Groq LLM configuration
        # ------------------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
       "GROQ_MODEL",
       "llama-3.3-70b-versatile"
    )
    GROQ_TEMPERATURE: float = float(
        os.getenv("GROQ_TEMPERATURE", "0.3")
    )
    GROQ_MAX_TOKENS: int = int(
        os.getenv("GROQ_MAX_TOKENS", "600")
    )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # Cap on how many images per class are opened (not just counted) when
    # checking for corruption / measuring dimensions during EDA. Opening
    # every one of ~54,000 images on every `analyze` run would be slow and
    # is not necessary to get a statistically useful picture of the data.
    EDA_SAMPLE_PER_CLASS: int = int(os.getenv("EDA_SAMPLE_PER_CLASS", "30"))

    def validate_dataset_root(self) -> None:
        """Raise a clear, actionable error if the dataset path is not configured."""
        if self.DATASET_ROOT == "YOUR_DATASET_PATH_HERE" or not self.DATASET_ROOT:
            raise FileNotFoundError(
                "Dataset path is not configured.\n"
                "Set it by either:\n"
                "  1) Editing DATASET_ROOT in src/config.py, or\n"
                "  2) Creating a .env file with:\n"
                "     PLANTVILLAGE_DATASET_ROOT=D:/PlantVillage\n"
            )
        if not Path(self.DATASET_ROOT).exists():
            raise FileNotFoundError(
                f"Configured dataset path does not exist: {self.DATASET_ROOT}\n"
                "Double-check the path and make sure it points to the folder "
                "that CONTAINS the 'color' / 'grayscale' / 'segmented' folders."
            )


# Single shared instance imported by every other module.
config = Config()
