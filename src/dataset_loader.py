"""
dataset_loader.py
------------------
Responsible ONLY for discovering what exists on disk:
  - which representations (color/grayscale/segmented) are present
  - which classes exist under a representation
  - which image files belong to a class
  - parsing "Plant___Condition" folder names into (plant, condition)

This module deliberately does NOT hard-code the number of classes, the
plant names, or the disease names anywhere. Everything is derived from
the actual folder structure at runtime, per project requirements, so the
same code works whether the dataset has 15 classes or 60.

Downstream modules (dataset_analysis.py, a future data pipeline in
preprocessing.py/train.py) build on top of the functions defined here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ClassInfo:
    """Everything derived from a single '<Plant>___<Condition>' folder name."""
    class_name: str      # raw folder name, e.g. "Tomato___Early_blight"
    plant: str            # e.g. "Tomato"
    condition: str        # e.g. "Early_blight"
    is_healthy: bool      # True if condition indicates a healthy leaf
    image_count: int
    representation: str   # "color" / "grayscale" / "segmented"


def discover_representations(dataset_root: str) -> List[str]:
    """
    Return which of the supported representations (color/grayscale/segmented)
    actually exist as sub-folders of dataset_root, preserving the configured
    order. Only representations that are present AND non-empty are returned.
    """
    root = Path(dataset_root)
    found = []
    for rep in config.SUPPORTED_REPRESENTATIONS:
        rep_path = root / rep
        if rep_path.is_dir() and any(rep_path.iterdir()):
            found.append(rep)
    if not found:
        raise FileNotFoundError(
            f"No representation folders (looked for {config.SUPPORTED_REPRESENTATIONS}) "
            f"found under '{dataset_root}'. Check that DATASET_ROOT points to the "
            f"folder that directly contains 'color'/'grayscale'/'segmented'."
        )
    return found


def parse_class_name(class_name: str) -> Tuple[str, str]:
    """
    Split a PlantVillage-style folder name into (plant, condition).

    Example:
        "Tomato___Early_blight" -> ("Tomato", "Early_blight")
        "Apple___healthy"       -> ("Apple", "healthy")

    Falls back gracefully (whole name as plant, "unknown" as condition) if a
    folder does not follow the expected naming convention, rather than
    crashing the whole analysis over one oddly named folder.
    """
    sep = config.CLASS_NAME_SEPARATOR
    if sep in class_name:
        plant, condition = class_name.split(sep, 1)
        return plant.strip(), condition.strip()
    logger.warning(
        "Class folder '%s' does not contain separator '%s'; "
        "treating whole name as plant with unknown condition.",
        class_name, sep,
    )
    return class_name.strip(), "unknown"


def is_healthy_condition(condition: str) -> bool:
    """A condition is considered 'healthy' if that word appears in its name."""
    return "healthy" in condition.lower()


def discover_classes(dataset_root: str, representation: str) -> List[str]:
    """
    Return the sorted list of class folder names found under
    <dataset_root>/<representation>/. The count is NEVER assumed - it comes
    directly from what's on disk.
    """
    rep_path = Path(dataset_root) / representation
    if not rep_path.is_dir():
        raise FileNotFoundError(f"Representation folder not found: {rep_path}")

    classes = sorted(
        [p.name for p in rep_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    if not classes:
        raise FileNotFoundError(f"No class folders found under: {rep_path}")
    return classes


def list_images(dataset_root: str, representation: str, class_name: str) -> List[Path]:
    """Return all image file paths belonging to one class folder."""
    class_dir = Path(dataset_root) / representation / class_name
    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix in config.IMAGE_EXTENSIONS
    ]
    return images


def build_class_index(dataset_root: str, representation: str) -> Dict[str, List[Path]]:
    """
    Return {class_name: [image_path, ...]} for every discovered class under
    the given representation. This is the core structure the rest of the
    pipeline (EDA, training data generators) is built on.
    """
    classes = discover_classes(dataset_root, representation)
    index: Dict[str, List[Path]] = {}
    for cls in classes:
        index[cls] = list_images(dataset_root, representation, cls)
    return index


def get_class_info_list(dataset_root: str, representation: str) -> List[ClassInfo]:
    """Return a fully-parsed ClassInfo object per class (used heavily by EDA)."""
    index = build_class_index(dataset_root, representation)
    infos = []
    for class_name, images in index.items():
        plant, condition = parse_class_name(class_name)
        infos.append(
            ClassInfo(
                class_name=class_name,
                plant=plant,
                condition=condition,
                is_healthy=is_healthy_condition(condition),
                image_count=len(images),
                representation=representation,
            )
        )
    return infos
