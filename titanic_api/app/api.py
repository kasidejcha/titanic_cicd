import json
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from classification_model import __version__ as model_version
from classification_model.predict import make_prediction
from classification_model.processing.data_manager import get_first_cabin, get_title
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from loguru import logger
from classification_model.config.core import config
from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: UploadFile = File(...)) -> Any:
    """
    Make survival predictions with the TID regression model
    """
    contents = input_data.file.read()
    buffer = BytesIO(contents)
    data = pd.read_csv(buffer)
    data = data.replace("?", np.nan)
    data = data.replace("?", np.nan)
    data["cabin"] = data["cabin"].apply(get_first_cabin)
    data["title"] = data["name"].apply(get_title)
    data["fare"] = data["fare"].astype("float")
    data["age"] = data["age"].astype("float")
    data.drop(
        labels=config.model_config.drop_vars, axis=1, inplace=True
    )
    # input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs")
    x = data.drop("survived", axis=1)
    y = data["survived"]
    results = make_prediction(x, y)

    if results["roc_auc_score"] < 0.8:
        logger.warning(
            f"Prediction validation roc score: {results.get('roc_auc_score')}"
        )
        raise HTTPException(
            status_code=400, detail=json.loads(results["roc_auc_score"])
        )

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results
