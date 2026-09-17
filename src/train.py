import tensorflow as tf

from src.config import config
from src.data_pipeline import get_datasets
from src.model import build_model


def train_model():

    print("Loading datasets...")

    train_ds, val_ds, test_ds, class_names, class_weights = get_datasets()

    print(f"Number of classes: {len(class_names)}")
    print(f"Training batches: {train_ds.cardinality().numpy()}")
    print(f"Validation batches: {val_ds.cardinality().numpy()}")
    print(f"Testing batches: {test_ds.cardinality().numpy()}")

    # Build MobileNetV2 model
    model = build_model(
        num_classes=len(class_names),
        image_size=config.IMAGE_SIZE,
        learning_rate=config.LEARNING_RATE,
    )

    callbacks = [

        # Save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        # Stop if validation accuracy stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),

        # Reduce learning rate when validation loss stops improving
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nStarting MobileNetV2 training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\nTraining completed.")

    # Save final model
    model.save(config.MODEL_PATH)

    print(f"Model saved to:")
    print(config.MODEL_PATH)

    return model, history, class_names


if __name__ == "__main__":
    train_model()