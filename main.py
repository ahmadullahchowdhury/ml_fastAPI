from fastapi import FastAPI, HTTPException
import numpy as np
import uvicorn

from model.iris_model import IrisModel
from schemas.iris_schema import IrisFeatures, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for Iris Classification using RandomForest",
    version="1.0.0"
)

# Initialize and train model
model = IrisModel()
model.train()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    try:
        # Convert input features to array
        feature_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        result = model.predict(feature_array)
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    try:
        return model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

