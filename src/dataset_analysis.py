"""
dataset_analysis.py
--------------------
Implements the "Dataset Analysis / EDA" requirement: inspects the dataset
that dataset_loader.py discovered, computes summary statistics, checks for
corrupt/unreadable images, and generates plots to support the project
report.

Design notes / why things are done this way:

- Counting images per class is done via cheap filesystem listing (fast,
  works for tens of thousands of files). Actually OPENING every image to
  check dimensions/corruption would be slow across the full PlantVillage
  dataset, so we take a bounded random sample per class
  (config.EDA_SAMPLE_PER_CLASS) for those checks. This keeps `analyze`
  usable on a normal laptop while still being statistically meaningful.
- Uses OpenCV (cv2) to open images because that is the same library used
  later in the Computer Vision preprocessing pipeline - if OpenCV can't
  decode an image, it can't be used for training either, so this is the
  correct corruption test for this project.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend - no display needed for CLI script
import matplotlib.pyplot as plt

from src.config import config
from src.dataset_loader import (
    discover_representations,
    get_class_info_list,
    list_images,
)
from src.utils import ensure_dir, format_bytes, get_logger, print_banner

logger = get_logger(__name__)


class DatasetAnalyzer:
    """Runs a full EDA pass over one representation of the dataset."""

    def __init__(self, dataset_root: str = None, representation: str = None):
        self.dataset_root = dataset_root or config.DATASET_ROOT
        self.representation = representation or config.DEFAULT_REPRESENTATION
        random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, generate_plots: bool = True) -> dict:
        config.validate_dataset_root()

        available_reps = discover_representations(self.dataset_root)
        if self.representation not in available_reps:
            logger.warning(
                "Representation '%s' not found; falling back to '%s'.",
                self.representation, available_reps[0],
            )
            self.representation = available_reps[0]

        print_banner("PLANTVILLAGE DATASET ANALYSIS")
        print(f"Dataset root      : {self.dataset_root}")
        print(f"Representation    : {self.representation}")
        print(f"Available reps    : {', '.join(available_reps)}")
        print()

        class_infos = get_class_info_list(self.dataset_root, self.representation)

        summary = self._compute_summary(class_infos, available_reps)
        corrupt_report = self._check_corrupt_and_dimensions(class_infos)
        summary.update(corrupt_report)

        self._print_report(summary)

        if generate_plots:
            plot_paths = self._generate_plots(class_infos, summary)
            summary["plots"] = plot_paths

        self._save_summary_json(summary)
        return summary

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_summary(self, class_infos, available_reps: List[str]) -> dict:
        total_images = sum(c.image_count for c in class_infos)
        plants = sorted(set(c.plant for c in class_infos))
        healthy_classes = [c.class_name for c in class_infos if c.is_healthy]
        diseased_classes = [c.class_name for c in class_infos if not c.is_healthy]

        images_per_class = {c.class_name: c.image_count for c in class_infos}
        counts = list(images_per_class.values())
        imbalance_ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else None

        plant_wise_counts: Dict[str, int] = defaultdict(int)
        plant_wise_class_count: Dict[str, int] = defaultdict(int)
        for c in class_infos:
            plant_wise_counts[c.plant] += c.image_count
            plant_wise_class_count[c.plant] += 1

        return {
            "dataset_root": self.dataset_root,
            "representation": self.representation,
            "available_representations": available_reps,
            "total_images": total_images,
            "total_classes": len(class_infos),
            "total_plants": len(plants),
            "plants": plants,
            "healthy_classes": healthy_classes,
            "diseased_classes": diseased_classes,
            "num_healthy_classes": len(healthy_classes),
            "num_diseased_classes": len(diseased_classes),
            "images_per_class": images_per_class,
            "min_images_in_a_class": min(counts) if counts else 0,
            "max_images_in_a_class": max(counts) if counts else 0,
            "avg_images_per_class": round(sum(counts) / len(counts), 1) if counts else 0,
            "class_imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
            "plant_wise_image_counts": dict(plant_wise_counts),
            "plant_wise_class_counts": dict(plant_wise_class_count),
        }

    def _check_corrupt_and_dimensions(self, class_infos) -> dict:
        """
        Sample up to EDA_SAMPLE_PER_CLASS images per class, try to decode
        them with OpenCV, and record dimensions. Returns corrupt file list
        and dimension statistics.
        """
        dims_width, dims_height = [], []
        corrupt_files: List[str] = []
        checked_count = 0

        for c in class_infos:
            images = list_images(self.dataset_root, self.representation, c.class_name)
            sample = random.sample(images, min(len(images), config.EDA_SAMPLE_PER_CLASS))
            for img_path in sample:
                checked_count += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_files.append(str(img_path))
                    continue
                h, w = img.shape[:2]
                dims_height.append(h)
                dims_width.append(w)

        dim_counter = Counter(zip(dims_width, dims_height))
        most_common_dims = dim_counter.most_common(5)

        return {
            "images_checked_for_integrity": checked_count,
            "corrupt_images_found": len(corrupt_files),
            "corrupt_image_paths_sample": corrupt_files[:20],  # cap what we print/save
            "most_common_dimensions": [
                {"width": w, "height": h, "count": cnt} for (w, h), cnt in most_common_dims
            ],
            "dimension_widths_sampled": dims_width,
            "dimension_heights_sampled": dims_height,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _print_report(self, s: dict) -> None:
        print(f"Total images            : {s['total_images']}")
        print(f"Total classes           : {s['total_classes']}")
        print(f"Total plant types       : {s['total_plants']}  -> {', '.join(s['plants'])}")
        print(f"Healthy classes         : {s['num_healthy_classes']}")
        print(f"Diseased classes        : {s['num_diseased_classes']}")
        print(f"Min images in a class   : {s['min_images_in_a_class']}")
        print(f"Max images in a class   : {s['max_images_in_a_class']}")
        print(f"Avg images per class    : {s['avg_images_per_class']}")
        print(f"Class imbalance ratio   : {s['class_imbalance_ratio']} (max/min)")
        print()
        print(f"Images sampled for integrity check : {s['images_checked_for_integrity']}")
        print(f"Corrupt/unreadable images found    : {s['corrupt_images_found']}")
        if s["most_common_dimensions"]:
            print("Most common image dimensions (sampled):")
            for d in s["most_common_dimensions"]:
                print(f"  {d['width']}x{d['height']}  ->  {d['count']} images")
        print()
        print("Images per class:")
        for cls, cnt in sorted(s["images_per_class"].items()):
            print(f"  {cls:<40} {cnt}")
        print()
        print_banner("ANALYSIS COMPLETE")

    def _save_summary_json(self, summary: dict) -> None:
        ensure_dir(config.EDA_DIR)
        out_path = Path(config.EDA_DIR) / f"eda_summary_{self.representation}.json"
        # dimension_widths/heights are only useful for plotting, not for the
        # saved report - drop the raw sample arrays to keep the JSON small.
        to_save = {k: v for k, v in summary.items()
                   if k not in ("dimension_widths_sampled", "dimension_heights_sampled")}
        with open(out_path, "w") as f:
            json.dump(to_save, f, indent=2)
        print(f"Summary saved to: {out_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def _generate_plots(self, class_infos, summary: dict) -> List[str]:
        ensure_dir(config.EDA_DIR)
        paths = []

        # 1) Class distribution (all classes)
        fig, ax = plt.subplots(figsize=(14, max(6, len(class_infos) * 0.25)))
        names = [c.class_name for c in class_infos]
        counts = [c.image_count for c in class_infos]
        order = sorted(range(len(names)), key=lambda i: counts[i])
        ax.barh([names[i] for i in order], [counts[i] for i in order], color="#4c8c4a")
        ax.set_xlabel("Number of images")
        ax.set_title(f"Class Distribution ({self.representation})")
        fig.tight_layout()
        p = str(Path(config.EDA_DIR) / f"class_distribution_{self.representation}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # 2) Plant-wise class distribution (image count per plant)
        plant_counts = summary["plant_wise_image_counts"]
        fig, ax = plt.subplots(figsize=(10, 6))
        plants = list(plant_counts.keys())
        vals = list(plant_counts.values())
        ax.bar(plants, vals, color="#3a6ea5")
        ax.set_ylabel("Number of images")
        ax.set_title(f"Plant-wise Image Distribution ({self.representation})")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        p = str(Path(config.EDA_DIR) / f"plant_distribution_{self.representation}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # 3) Image dimension distribution (from sampled images)
        widths = summary.get("dimension_widths_sampled", [])
        heights = summary.get("dimension_heights_sampled", [])
        if widths and heights:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(widths, heights, alpha=0.4, s=10, color="#a53a3a")
            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Height (px)")
            ax.set_title("Sampled Image Dimensions")
            fig.tight_layout()
            p = str(Path(config.EDA_DIR) / f"image_dimensions_{self.representation}.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

        # 4) Sample image grid (one representative image per plant, up to 12)
        p = self._save_sample_grid(class_infos)
        if p:
            paths.append(p)

        print(f"Plots saved to: {config.EDA_DIR}")
        return paths

    def _save_sample_grid(self, class_infos) -> str:
        """Save a grid of one sample image per plant type for visual inspection."""
        seen_plants = {}
        for c in class_infos:
            if c.plant not in seen_plants:
                imgs = list_images(self.dataset_root, self.representation, c.class_name)
                if imgs:
                    seen_plants[c.plant] = (c.class_name, random.choice(imgs))
            if len(seen_plants) >= 12:
                break

        if not seen_plants:
            return ""

        n = len(seen_plants)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for ax, (plant, (cls, img_path)) in zip(axes, seen_plants.items()):
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.set_title(cls, fontsize=9)
            ax.axis("off")
        for ax in axes[len(seen_plants):]:
            ax.axis("off")

        fig.suptitle(f"Sample Images per Plant ({self.representation})")
        fig.tight_layout()
        p = str(Path(config.EDA_DIR) / f"sample_images_{self.representation}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        return p
