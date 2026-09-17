"""
Model evaluation for the Plant Disease Detection project.

Evaluates the saved best model on the held-out test set and generates:
- Test loss and accuracy
- Classification report
- Confusion matrix
- Training/validation accuracy curve
- Training/validation loss curve
- JSON metrics
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import config
from src.data_pipeline import get_datasets


EVALUATION_DIR = Path(config.REPORTS_DIR) / "evaluation"


def save_training_curves(history_dict):
    """Save accuracy and loss curves from the training history."""

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["accuracy"], label="Training Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_dict["loss"], label="Training Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "loss_curve.png", dpi=200)
    plt.close()

    print("Training curves saved.")


def save_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save the confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Plant Disease Classification - Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(
        tick_marks,
        class_names,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        tick_marks,
        class_names,
        fontsize=7,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    path = EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {path}")

    return cm


def evaluate_model():
    """Evaluate the saved model on the test dataset."""

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Check model
    # ------------------------------------------------------------------

    if not Path(config.MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Trained model not found at:\n{config.MODEL_PATH}\n"
            "Run the training phase first."
        )

    print("=" * 60)
    print("PLANT DISEASE MODEL EVALUATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load test dataset
    # ------------------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_ds, class_names, _ = get_datasets()

    print(f"Number of classes: {len(class_names)}")
    print(f"Test batches: {test_ds.cardinality().numpy()}")

    # ------------------------------------------------------------------
    # Load saved model
    # ------------------------------------------------------------------

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(config.MODEL_PATH)

    print(f"Model loaded from:\n{config.MODEL_PATH}")

    # ------------------------------------------------------------------
    # Keras test evaluation
    # ------------------------------------------------------------------

    print("\nEvaluating model on test set...\n")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4%}")

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    print("\nGenerating predictions...")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ------------------------------------------------------------------
    # Accuracy verification
    # ------------------------------------------------------------------

    calculated_accuracy = accuracy_score(y_true, y_pred)

    print(f"Calculated Accuracy: {calculated_accuracy:.4%}")

    # ------------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------------

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)

    # Save classification report
    report_path = EVALUATION_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Plant Disease Detection - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    print(f"Classification report saved to: {report_path}")

    # Save JSON report
    json_report_path = EVALUATION_DIR / "classification_report.json"

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "calculated_accuracy": float(calculated_accuracy),
        "num_classes": len(class_names),
        "test_samples": int(len(y_true)),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(
            report_dict["weighted avg"]["precision"]
        ),
        "weighted_recall": float(
            report_dict["weighted avg"]["recall"]
        ),
        "weighted_f1": float(
            report_dict["weighted avg"]["f1-score"]
        ),
    }

    metrics_path = EVALUATION_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # ------------------------------------------------------------------
    # Save class names
    # ------------------------------------------------------------------

    class_names_path = Path(config.CLASS_NAMES_PATH)

    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    print(f"Class names saved to: {class_names_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    for file in sorted(EVALUATION_DIR.iterdir()):
        print(f"  - {file.name}")

    return metrics


if __name__ == "__main__":
    evaluate_model()