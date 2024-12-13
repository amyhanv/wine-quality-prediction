import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def train():
    # Load the dataset
    data = pd.read_csv("./winequality-red.csv") 

    data = data.replace({'quality': {
                                    8: 'High',
                                    7: 'High',
                                    6: 'Medium',
                                    5: 'Medium',
                                    4: 'Low',
                                    3: 'Low',
                                }
                        })

    # Split features and target
    x = data.drop("quality", axis=1)  # Features
    y = data["quality"]               # Target

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, shuffle = True, random_state = 1)


    # Standardization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize the Random Forest model
    rfc = RandomForestClassifier(
        n_estimators=50, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        bootstrap=True, 
        random_state=42,
        class_weight='balanced'
    )

    # Train the model
    rfc.fit(x_train, y_train)

    # Save the trained model
    joblib.dump(rfc, "./models/model.pkl")

    # Save the scaler for preprocessing during inference
    joblib.dump(scaler, "./models/scaler.pkl")

    return {
        "message": "Training completed successfully!",
        "model_path": "./models/model.pkl",
        "scaler_path": "./models/scaler.pkl",
        "train_accuracy": rfc.score(x_train, y_train),
        "test_accuracy": rfc.score(x_test, y_test)
    }
