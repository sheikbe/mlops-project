import os
from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SPECIES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Classifier API", version="1.0.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


class Prediction(BaseModel):
    species: str
    class_id: int


@lru_cache(maxsize=1)
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures) -> Prediction:
    row = [
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]
    class_id = int(get_model().predict([row])[0])
    return Prediction(species=SPECIES[class_id], class_id=class_id)
