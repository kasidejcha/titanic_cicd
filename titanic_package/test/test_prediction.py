from sklearn.metrics import roc_auc_score

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    x_test, y_test = sample_input_data

    # When
    result = make_prediction(x_test, y_test)

    # Then
    assert result["roc_auc_score"] > 0.8
