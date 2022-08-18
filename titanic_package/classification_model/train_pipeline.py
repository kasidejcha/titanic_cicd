import numpy as np
from config.core import config
from pipeline import titanic_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.data)
    # print(data.head())
    x = data.drop(config.model_config.target, axis=1)
    y = data[config.model_config.target]

    # fit model
    titanic_pipe.fit(x, y)

    # persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == "__main__":
    run_training()
    # print('Training completed')
