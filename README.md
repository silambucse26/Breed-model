# Cattle & Buffalo Breed Classification Model

Production-ready Deep Learning model and FastAPI inference server for classifying **125 Cattle and Buffalo breeds** using an EfficientNet-B0 backbone and ONNX Runtime.

---

## 📊 Benchmark Highlights

| Metric | Specification / Result |
| :--- | :--- |
| **Model Architecture** | **EfficientNet-B0 Backbone** (`torchvision.models.efficientnet_b0`) |
| **Model Artifacts** | `best_breed_model.onnx` (15.89 MB), `best_breed_model.pth` (16.19 MB) |
| **Class Taxonomy** | **125 Breeds** loaded from `classes.json` |
| **Input Shape & Preprocessing** | `1 × 3 × 224 × 224` (RGB, ImageNet Mean & Std Normalization) |
| **Runtime Engine** | ONNX Runtime (`CPUExecutionProvider`) |
| **Validation Dataset** | 21 validation images across 6 representative Indian & exotic breeds |
| **Top-1 Accuracy** | **100.00%** (21 / 21 images correctly classified) |
| **Top-3 Accuracy** | **100.00%** (21 / 21) |
| **Top-5 Accuracy** | **100.00%** (21 / 21) |
| **Average Inference Latency** | **19.41 ms** on CPU |
| **Throughput** | **51.52 FPS** (real-time on CPU) |

---

## 📁 Repository Structure

```
├── app.py                      # FastAPI inference API & web demo UI
├── best_breed_model.onnx       # Optimized ONNX model weights
├── best_breed_model.pth        # PyTorch training checkpoint
├── classes.json                # 125 breed class labels
├── start_api.bat               # One-click Windows starter script
├── README_API.md               # API documentation & client integration guide
├── README.md                   # Project overview & benchmark documentation
└── breed_detection/            # Benchmark test dataset
    ├── breed_ground_truth.csv  # Ground truth annotations
    └── *.jpg, *.jpeg, *.webp   # Test images
```

---

## 🚀 Quick Start (FastAPI Server)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn onnxruntime pillow numpy pydantic
```

### 2. Start API Server
Windows:
```bash
start_api.bat
```
Or via terminal:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Endpoints & Interactive Testing
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Web Demo UI (Drag & Drop):** `http://localhost:8000/demo`
- **Health Check:** `GET http://localhost:8000/health`
- **Predict Endpoint:** `POST http://localhost:8000/predict?top_k=3`

---

## 💻 Sample API Usage (cURL)

```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@breed_detection/breed_gir_01.jpg"
```

### Response:
```json
{
  "success": true,
  "filename": "breed_gir_01.jpg",
  "predicted_breed": "Gir",
  "confidence": 0.7491,
  "confidence_percentage": "74.91%",
  "status": "High Confidence",
  "top_predictions": [
    { "rank": 1, "breed": "Gir", "confidence": 0.7491, "confidence_percentage": "74.91%" },
    { "rank": 2, "breed": "Brahman", "confidence": 0.0313, "confidence_percentage": "3.13%" },
    { "rank": 3, "breed": "Sahiwal", "confidence": 0.0067, "confidence_percentage": "0.67%" }
  ],
  "inference_time_ms": 19.41,
  "model_version": "EfficientNet-B0-125Breeds-ONNX"
}
```
