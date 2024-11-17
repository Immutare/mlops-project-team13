import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

with open("model.pkl", "rb") as f:
    model = joblib.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = joblib.load(f)


class MaternalHealthData(BaseModel):
    features: list[float]


app = FastAPI()


@app.post("/predict")
def predict(data: MaternalHealthData):
    features = np.array(data.features)

    if len(features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Features should be of length {model.n_features_in_}",
        )

    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0]),
        "class": label_encoder.inverse_transform(prediction)[0],
    }


@app.get("/")
def read_root():
    return {"message": "Maternal Health Risk Prediction API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
