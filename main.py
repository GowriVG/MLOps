from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI(title="Gowri's MLOps Demo API", version="1.0")

# Load the trained model
model = joblib.load("model.joblib")  # Make sure this file exists

# Define the input data model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def read_root():
    # return {"message": "Welcome to the Iris Predictor API", "version": "1.0"}
    return {"message": "Welcome to the Iris Predictor API", "version": "1.1"}


# Prediction endpoint
@app.post("/predict")
def predict_iris(data: IrisInput):
    # Convert input to numpy array
    features = np.array([[data.sepal_length,
                          data.sepal_width,
                          data.petal_length,
                          data.petal_width]])
    
    # Make prediction
    prediction_index = model.predict(features)[0]
    
    # Map index to species name
    species = ['setosa', 'versicolor', 'virginica']
    prediction_name = species[prediction_index]
    
    return {
        "prediction_index": int(prediction_index),
        "prediction_name": prediction_name
    }
