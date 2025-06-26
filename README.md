# SmartBot V2 - Multimodal Emotion Recognition

SmartBot V2 is a real-time multimodal emotion recognition system that processes base64-encoded `.wav` audio inputs and classifies them into one of four emotions: **happy**, **sad**, **neutral**, or **anger**.

This repository provides the official inference notebook, API examples, input/output schema, and assets used in the **AWS Marketplace Model Package Listing**.

---

## Model Overview

- **Architecture**: TensorFlow-based Multimodal Deep Neural Network
- **Input**: Base64-encoded `.wav` audio (16kHz)
- **Output**: Emotion classification with probabilities for each emotion class
- **Use Case**: Sentiment analysis for customer support, real-time conversation analytics, voice-based emotion understanding

---

## How It Works

1. **Input**: Base64-encoded `.wav` audio provided by the user via JSON payload.
2. **Preprocessing**: Extracted mel-spectrograms from audio are resized to `(64, 64)`.
3. **Inference**: Predictions generated using the trained model.
4. **Output**: Predicted label and class probabilities returned in JSON.

---

## Files Included

| File | Description |
|------|-------------|
| `smartbot_v2_sample_inference.ipynb` | Sample SageMaker Notebook to invoke the API |
| `serve/inference.py` | FastAPI-powered inference server used inside SageMaker container |
| `serve/sample_input.json` | Example input payload for real-time endpoint |
| `serve/sample_output.json` | Sample prediction output |
| `docs/API.md` | API contract for integrating with the SageMaker endpoint |

---

## Testing the Endpoint

```bash
curl -X POST http://<your-endpoint-url>/invocations \
     -H "Content-Type: application/json" \
     -d @serve/sample_input.json
```

---

## License

See [LICENSE](LICENSE) for usage rights and distribution.

---

## Maintainers

This solution is developed by the MTU Research Team and productized by the Cloud202 MLOps Team for AWS SageMaker Marketplace.
