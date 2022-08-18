import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    data = load_dataset(file_name=config.app_config.test_data)

    X_test = data.drop(config.model_config.target, axis=1)
    y_test = data[config.model_config.target]

    return X_test, y_test
