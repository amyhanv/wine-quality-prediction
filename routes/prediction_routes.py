from fastapi import APIRouter
from controllers.prediction_controller import predict

prediction_router = APIRouter()

@prediction_router.post("/")
def predict_endpoint(data: dict):
    return predict(data)
