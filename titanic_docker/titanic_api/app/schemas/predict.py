from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    version: str
    roc_auc_score: Optional[Any]
    accuracy: Optional[Any]
