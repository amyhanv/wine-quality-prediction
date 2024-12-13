from fastapi import APIRouter
from controllers.training_controller import train_model

training_router = APIRouter()

@training_router.post("/")
def training_endpoint():
    return train_model()
