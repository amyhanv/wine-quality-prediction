
from transformations.data_preprocessing import preprocess_data
from transformations.feature_engineering import create_features
import pandas as pd


def test_preprocess_data():
    raw_data = pd.DataFrame({"A": [1, 2, None]})
    processed_data = preprocess_data(raw_data)
    assert processed_data.isna().sum().sum() == 0

def test_create_features():
    data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    new_data = create_features(data)
    assert "new_feature" in new_data.columns
