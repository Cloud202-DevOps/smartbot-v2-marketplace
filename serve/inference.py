import os
import io
import base64
import numpy as np
import joblib
import librosa
import tensorflow as tf
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
from starlette.responses import JSONResponse

app = FastAPI()

# === Load model and label encoder from SageMaker default path ===
MODEL_PATH = "/opt/ml/model"
MODEL = tf.keras.models.load_model(os.path.join(MODEL_PATH, "Multimodal"))
ENCODER = joblib.load(os.path.join(MODEL_PATH, "label_encoder.joblib"))

EMOTIONS = list(ENCODER.classes_)

# === Helper: Process .wav to mel-spectrogram ===
def preprocess_audio(base64_audio: str) -> np.ndarray:
    audio_bytes = base64.b64decode(base64_audio)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_spec = librosa.power_to_db(mel_spec)
    log_spec_resized = tf.image.resize(log_spec[np.newaxis, ..., np.newaxis], size=(64, 64)).numpy()
    return log_spec_resized

# === Input schema ===
class Invocation(BaseModel):
    invocations: Dict[str, str]

@app.post("/invocations")
async def predict(request: Request):
    payload = await request.json()
    data = payload["invocations"]

    results = []
    for uid, base64_audio in data.items():
        input_tensor = preprocess_audio(base64_audio)
        preds = MODEL(input_tensor, training=False).numpy()[0]
        pred_idx = np.argmax(preds)
        pred_label = EMOTIONS[pred_idx]

        results.append({
            "uid": uid,
            "predicted_class": int(pred_idx),
            "predicted_label": pred_label,
            "probabilities": dict(zip(EMOTIONS, preds.round(6).tolist()))
        })

    return JSONResponse(content={"predictions": results})
