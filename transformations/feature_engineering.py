def create_features(data):
    # Example feature engineering
    data['new_feature'] = data['feature1'] * data['feature2']
    return data
