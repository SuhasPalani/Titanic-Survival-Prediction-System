from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class PredictionRequest(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Optional[str] = 'S'

class BatchPredictionRequest(BaseModel):
    passengers: List[Dict[str, Any]]
    filters: Optional[Dict[str, Any]] = {}
    sort_by: Optional[str] = 'survival_probability'
    ascending: Optional[bool] = False