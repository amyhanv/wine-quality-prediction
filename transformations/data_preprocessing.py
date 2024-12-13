import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing steps
    data = data.fillna(0)
    return data
