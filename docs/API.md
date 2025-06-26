# SmartBot V2 Inference API Documentation

This document describes the HTTP endpoints and payload structure for invoking the SmartBot V2 emotion recognition model deployed on AWS SageMaker.

---

## Health Check

- **URL**: `/ping`
- **Method**: `GET`
- **Response**:
  ```json
  { "message": "SmartBot V2 Inference API is healthy." }
  ```

---

## Emotion Prediction Endpoint

- **URL**: `/invocations`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "invocations": {
      "sample123": "<base64_encoded_wav_data>"
    }
  }
  ```

  - The key (e.g., `sample123`) is a unique user-defined UID.
  - The value must be a Base64-encoded WAV audio string.

- **Response**:
  ```json
  {
    "predictions": [
      {
        "uid": "sample123",
        "predicted_class": 1,
        "predicted_label": "happy",
        "probabilities": {
          "anger": 0.000001,
          "happy": 0.999294,
          "neutral": 0.000659,
          "sad": 0.000046
        }
      }
    ]
  }
  ```

---

## Notes
- The model only supports Base64-encoded `.wav` files.
- The UID key must be unique per audio entry in each request.
- Internally, audio is converted into mel-spectrograms before inference.
