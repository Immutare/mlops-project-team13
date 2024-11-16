FROM python:3.10-slim

ARG MODEL_NAME

WORKDIR /app

COPY maternalhealth/model/${MODEL_NAME}_model.pkl /app/model.pkl
COPY maternalhealth/data/preprocessed/train_label_encoder.pkl /app/label_encoder.pkl
COPY main.py /app/main.py

RUN pip install fastapi uvicorn scikit-learn

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
