# Model Deployment (FastAPI + Docker)

## Build Docker Image

From the project root:

```bash
docker build -t spikeprot-model-server -f scripts/deployment/Dockerfile .
```

## Run the Model Server

```bash
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/model.pth \
  -e MODEL_TYPE=rnn \
  spikeprot-model-server
```

- `MODEL_PATH`: Path to the trained model file inside the container (default: `model.pth`)
- `MODEL_TYPE`: `rnn` or `attn` (default: `rnn`)
- `ATTN_SEQ_LENGTH`: (optional) Sequence length for attention models

## Make Predictions

Send a POST request to `/predict`:

```json
POST http://localhost:8000/predict
{
  "data": [[...], [...], ...],
  "model_type": "rnn",
  "attn_seq_length": 10
}
```

Response:
```json
{
  "predictions": [0, 1, ...]
}
```

