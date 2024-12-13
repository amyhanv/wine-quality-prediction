from fastapi import FastAPI
from config.settings import settings
from routes.prediction_routes import prediction_router
from routes.training_routes import training_router

app = FastAPI(
    title=settings.app_name,
    version="1.0",
    description="A FastAPI application for machine learning predictions and training."
)

# Include routers
app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])
app.include_router(training_router, prefix="/train", tags=["Training"])

@app.get("/")
def root():
    return {"message": "Welcome to the ML FastAPI application!"}
