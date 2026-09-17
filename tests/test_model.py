from src.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=38)

    assert model.output_shape == (None, 38)


def test_model_has_correct_number_of_classes():
    model = build_model(num_classes=38)

    assert model.layers[-1].units == 38