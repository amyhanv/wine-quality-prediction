from inference.inference_service import make_prediction

def predict(data: dict):
    prediction = make_prediction(data)
    return {"prediction": prediction}
