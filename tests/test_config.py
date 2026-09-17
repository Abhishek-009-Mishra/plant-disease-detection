from src.config import config


def test_image_configuration():
    assert config.IMAGE_SIZE == 224
    assert config.IMAGE_CHANNELS == 3


def test_dataset_split():
    total = (
        config.TRAIN_SPLIT
        + config.VAL_SPLIT
        + config.TEST_SPLIT
    )

    assert abs(total - 1.0) < 1e-6


def test_training_configuration():
    assert config.BATCH_SIZE > 0
    assert config.EPOCHS > 0
    assert config.LEARNING_RATE > 0