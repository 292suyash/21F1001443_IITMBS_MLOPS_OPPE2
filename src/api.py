from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()
FastAPIInstrumentor().instrument_app(app)
model = mlflow.sklearn.load_model("model")  # Update path in Docker

@app.get("/live_check")
def live_check():
    return {"status": "alive"}

@app.get("/ready_check")
def ready_check():
    return {"status": "ready"}

@app.post("/predict")
def predict(data: dict):
    X = pd.DataFrame([data])
    pred = model.predict(X)
    return {"prediction": int(pred[0])}
