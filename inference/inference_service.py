import pandas as pd
import numpy as np
import joblib
from inference.model_loader import load_model

FEATURE_COLUMNS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Load the model and scaler
model = load_model()
scaler = joblib.load("./models/scaler.pkl")  # Load the scaler saved during training


def make_prediction(data: dict):
    """
    Accepts input in the format:
    {
        "features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    }
    Where, each index signifies:
    1.fixed_acidity, 2. volatile_acidity, 3. citric_acid, 4.residual_sugar, 5.chlorides, 6.free_sulfur_dioxide, 7.total_sulfur_dioxide, 8.density, 9.pH, 10.sulphates, 11.alcohol
    
    Returns the prediction as a list.
    """
    try:
        # Validate input format
        if "features" not in data:
            raise KeyError("Missing 'features' key in input data.")
        
        # Extract features from input
        features = np.array(data["features"]).reshape(1, -1)
         # Validate feature length
        if len(features) != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {len(features)}.")

        # Validate feature types
        if not all(isinstance(value, (int, float)) for value in features):
            raise ValueError("All features must be numeric.")
        
        # Convert to DataFrame for consistency
        input_data = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        
        # Preprocess the features using the scaler
        features_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(features_scaled)

        # Return the prediction
        return {"prediction": prediction.tolist()}
    except KeyError as e:
        return {"error": f"Missing key in input data: {str(e)}"}
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}

    
