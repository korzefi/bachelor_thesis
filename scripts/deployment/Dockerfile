FROM python:3.10-slim

WORKDIR /app

COPY ../../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

COPY ../../scripts ./scripts

EXPOSE 8000

ENV MODEL_PATH=model.pth
ENV MODEL_TYPE=rnn

CMD ["uvicorn", "scripts.deployment.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]

