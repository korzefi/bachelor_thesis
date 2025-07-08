from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import os
import yaml
from scripts.training.model_definitions import RnnModel, AttnRnnModel

app = FastAPI()

class PredictRequest(BaseModel):
    data: list
    model_type: str
    attn_seq_length: int = None

class PredictResponse(BaseModel):
    predictions: list

model = None
model_type = None
attn_seq_length = None

CONFIG_PATH = os.getenv('DEPLOYMENT_CONFIG', 'scripts/deployment/deployment_config.yaml')
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

def get_config_value(key, default=None):
    return os.getenv(key.upper(), config.get(key, default))

@app.on_event("startup")
def load_model():
    global model, model_type, attn_seq_length
    model_path = get_config_value('model_path', 'model.pth')
    model_type = get_config_value('model_type', 'rnn')
    attn_seq_length = get_config_value('attn_seq_length', None)
    dummy_input = np.zeros((1, 10, 100))  # Adjust shape as needed
    input_dim = dummy_input.shape[2] if len(dummy_input.shape) == 3 else dummy_input.shape[1]
    if model_type == 'rnn':
        model = RnnModel(input_dim=input_dim)
    elif model_type == 'attn':
        if attn_seq_length is None:
            attn_seq_length = dummy_input.shape[1]
        model = AttnRnnModel(input_dim=input_dim, seq_length=int(attn_seq_length))
    else:
        raise ValueError('Unknown model type')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.data, dtype=np.float32)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor) if request.model_type == 'rnn' else model(X_tensor)[0]
        preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
    return PredictResponse(predictions=preds)
