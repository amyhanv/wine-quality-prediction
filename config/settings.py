from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "ML FastAPI Application"
    environment: str = "development"
    model_path: str = "./models/model.pkl"

    class Config:
        env_file = ".env"

# Instantiate settings
settings = Settings()
