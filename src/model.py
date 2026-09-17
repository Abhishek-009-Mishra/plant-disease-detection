import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_model(
    num_classes: int,
    image_size: int = 224,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Build a MobileNetV2 transfer-learning classifier.

    Args:
        num_classes: Number of output classes.
        image_size: Input image height/width.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled TensorFlow Keras model.
    """

    inputs = layers.Input(
        shape=(image_size, image_size, 3),
        name="leaf_image"
    )

    # MobileNetV2 expects inputs scaled to [-1, 1].
    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_scaling"
    )(inputs)

    # Pretrained MobileNetV2 feature extractor
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="disease_prediction"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="PlantDisease_MobileNetV2"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    return model