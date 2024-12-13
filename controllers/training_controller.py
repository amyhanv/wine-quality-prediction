from training.train_model import train

def train_model():
    result = train()
    return {"message": "Model trained successfully!", "details": result}
