# Wine Quality Prediction API

This project is a FastAPI-based machine learning application that predicts wine quality based on its chemical properties. It includes both a training pipeline for the model and an inference endpoint for predictions.

## **Features**
- Predict wine quality (`Low`, `Medium`, or `High`) based on input features.
- Train a Random Forest model with hyperparameter tuning.
- API endpoints for prediction and training, designed for scalability and modularity.

## **Project Structure**
```
project/
├── main.py                       # Entry point for FastAPI application
├── transformations/              # Data preprocessing and feature engineering
│   ├── __init__.py
│   ├── data_preprocessing.py     # Handles data cleaning and preparation
│   └── feature_engineering.py    # Feature engineering logic
├── inference/                    # Model loading and inference logic
│   ├── __init__.py
│   ├── model_loader.py           # Loads the saved model
│   └── inference_service.py      # Processes inference requests
├── training/                     # Model training and hyperparameter tuning
│   ├── __init__.py
│   ├── train_model.py            # Training pipeline
│   └── hyperparameter_tuning.py  # Handles hyperparameter tuning
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── unit/                     # Unit tests for individual components
│   │   ├── __init__.py
│   │   └── test_transformations.py
│   ├── integration/              # Integration tests for the API
│   │   ├── __init__.py
│   │   └── test_routes.py
│   └── data_tests/               # Data quality validation tests
│       ├── __init__.py
│       └── test_data_quality.py
├── config/                       # Configuration files for the application
│   ├── __init__.py
│   └── settings.py               # Application settings
├── models/                       # Directory for storing trained models
│   ├── model.pkl                 # Saved Random Forest model
│   └── scaler.pkl                # Saved MinMaxScaler
├── a1_final.py                   # Assignment #1 Jupyter Notebook converted to Python file
├── winequality-red.csv           # Dataset used for training
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```


## **Setup Instructions**
1. Install all required packages:
`pip install -r requirements.txt`


2. Start the FastAPI server:
`uvicorn main:app --reload`

3. Test out the API by going to:
`http://127.0.0.1:8000`
    - Swagger UI: `http://127.0.0.1:8000/docs`

3. You can run tests with:
`pytest tests/`


## **Endpoints**
### 1. Prediction
#### POST `/predict/`
- Request:
    ```
    {
        "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    }
    ```

- Response:
    ```
    {
        "prediction": ["Medium"]
    }
    ```

### 2. Training
#### POST `/train/`
- Response:
    ```
    {
        "message": "Model trained successfully!",
        "details": "Model saved as 'model.pkl'"
    }
    ```

## Some Sample Requests for Endpoint Testing:
High quality:
```
{
  "features": [6.5, 0.3, 0.4, 2.1, 0.054, 18.0, 40.0, 0.9952, 3.25, 0.72, 14.2]
}
```

Medium quality (high residual sugar and chlorides):
```
{
  "features": [7.9, 0.6, 0.2, 6.8, 0.12, 22.0, 45.0, 0.9969, 3.18, 0.66, 10.8]
}
```


## Contact
For any issues or inquiries, please contact:

Email: `amy.hanvoravongchai@mail.mcgill.ca`