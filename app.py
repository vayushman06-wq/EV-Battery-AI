
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="EV Battery Condition-Aware AI API",
    version="1.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ev_battery_model.joblib"
)

package = joblib.load(MODEL_PATH)

model = package["model"]
features = package["features"]
classes = package["classes"]

class BatteryData(BaseModel):
    initial_soc_percent: float = Field(ge=0, le=100)
    response200_mV: float
    response205_mV: float
    response209_mV: float
    peak_response_mV: float
    pulse_slope_mV_per_s: float
    residual220_mV: float
    recovery_slope_mV_per_s: float
    recovery_ratio_percent: float

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "EV Battery AI",
        "model": "Random Forest"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: BatteryData):

    try:
        input_values = {
            "SOC_percent": data.initial_soc_percent,
            "Response200_mV": data.response200_mV,
            "Response205_mV": data.response205_mV,
            "Response209_mV": data.response209_mV,
            "PeakResponse_mV": data.peak_response_mV,
            "PulseSlope_mV_per_s": data.pulse_slope_mV_per_s,
            "Residual220_mV": data.residual220_mV,
            "RecoverySlope_mV_per_s": data.recovery_slope_mV_per_s,
            "RecoveryRatio_percent": data.recovery_ratio_percent
        }

        X = np.array(
            [[input_values[name] for name in features]],
            dtype=float
        )

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = float(np.max(probabilities))

        probability_output = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

        return {
            "predicted_condition": str(prediction),
            "confidence": confidence,
            "class_probabilities": probability_output,
            "model_type": "Random Forest",
            "diagnostic_scope": "Simulated resistance scenario classification",
            "safety_status": "Not evaluated by AI"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
