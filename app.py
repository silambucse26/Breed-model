import os
import io
import json
import time
from typing import Optional, List
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_breed_model.onnx")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

# Preprocessing Constants (ImageNet Normalization)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Global variables for model session and classes
session: Optional[ort.InferenceSession] = None
classes: List[str] = []

def load_model_and_classes():
    global session, classes
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"classes.json not found at {CLASSES_PATH}")
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)["classes"]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Load ONNX session with CPU provider
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    
    # Warmup
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    for _ in range(3):
        _ = session.run(None, {"input": dummy})
    print(f"Model loaded successfully with {len(classes)} classes.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ONNX session and classes
    load_model_and_classes()
    yield

# Initialize FastAPI App
app = FastAPI(
    title="Cattle & Buffalo Breed Classification API",
    description="High-performance ONNX inference API for 125 Cattle and Buffalo breeds",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Mobile and Web Apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes: bytes, size: int = 224) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image format: {str(e)}")
    
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
    arr = np.expand_dims(arr, 0)        # (1, 3, H, W)
    return (arr - MEAN) / STD

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Response Schemas
class TopPrediction(BaseModel):
    rank: int
    breed: str
    confidence: float
    confidence_percentage: str

class PredictionResponse(BaseModel):
    success: bool
    filename: str
    predicted_breed: str
    confidence: float
    confidence_percentage: str
    status: str
    top_predictions: List[TopPrediction]
    inference_time_ms: float
    model_version: str

@app.get("/")
def root():
    return {
        "service": "Cattle & Buffalo Breed Classification API",
        "status": "online",
        "model_architecture": "EfficientNet-B0 (ONNX)",
        "total_classes": len(classes),
        "interactive_docs": "/docs",
        "web_demo_ui": "/demo",
        "predict_endpoint": "POST /predict"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "classes_count": len(classes)
    }

@app.get("/classes")
def get_classes():
    return {
        "total_classes": len(classes),
        "classes": classes
    }

# Fallback for when someone types /predict in browser address bar (GET request)
@app.get("/predict")
def predict_get_help():
    return JSONResponse(
        status_code=400,
        content={
            "error": "Method Not Allowed (GET)",
            "message": "The /predict endpoint requires an HTTP POST request with an uploaded image file (multipart/form-data).",
            "how_to_test": {
                "web_demo": "Visit http://localhost:8000/demo to drag & drop an image in your browser",
                "swagger_ui": "Visit http://localhost:8000/docs to test via interactive API UI",
                "curl_example": "curl -X POST http://localhost:8000/predict -F file=@image.jpg"
            }
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_breed(
    file: UploadFile = File(..., description="Upload an image (JPG, PNG, JPEG, WEBP)"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to return (1-10)")
):
    if session is None or not classes:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    # Read uploaded file
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Preprocess
    try:
        tensor = preprocess_image(image_bytes, size=224)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inference & Timing
    t0 = time.perf_counter()
    out = session.run(None, {"input": tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = softmax(out)[0]
    sorted_indices = np.argsort(probs)[::-1]

    top_results: List[TopPrediction] = []
    for rank, idx in enumerate(sorted_indices[:top_k], start=1):
        c_prob = float(probs[idx])
        top_results.append(TopPrediction(
            rank=rank,
            breed=classes[int(idx)],
            confidence=round(c_prob, 4),
            confidence_percentage=f"{c_prob * 100:.2f}%"
        ))

    top1 = top_results[0]
    # Production status threshold
    if top1.confidence >= 0.70:
        status_label = "High Confidence"
    elif top1.confidence >= 0.40:
        status_label = "Moderate Confidence"
    else:
        status_label = "Low Confidence - Manual Verification Advised"

    return PredictionResponse(
        success=True,
        filename=file.filename or "uploaded_image",
        predicted_breed=top1.breed,
        confidence=top1.confidence,
        confidence_percentage=top1.confidence_percentage,
        status=status_label,
        top_predictions=top_results,
        inference_time_ms=round(latency_ms, 2),
        model_version="EfficientNet-B0-125Breeds-ONNX"
    )

# Interactive Web Demo Page for easy browser testing
@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Breed Classification Demo</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f4f8; margin: 0; padding: 30px; display: flex; justify-content: center; }
            .card { background: white; max-width: 650px; width: 100%; border-radius: 12px; box-shadow: 0 4px 18px rgba(0,0,0,0.08); padding: 30px; }
            h2 { color: #1f497d; margin-top: 0; }
            .drop-zone { border: 2px dashed #93c5fd; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; background: #f8fafc; transition: all 0.2s; }
            .drop-zone:hover { background: #eff6ff; border-color: #3b82f6; }
            input[type=file] { display: none; }
            button { background: #1f497d; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 15px; width: 100%; }
            button:hover { background: #173860; }
            #preview { max-width: 100%; max-height: 260px; margin-top: 15px; border-radius: 8px; display: none; object-fit: contain; }
            .result-box { margin-top: 20px; padding: 15px; border-radius: 8px; background: #f1f5f9; display: none; }
            .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: bold; background: #10b981; color: white; }
            pre { background: #1e293b; color: #38bdf8; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 13px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🐄 Breed Classification API - Test UI</h2>
            <p style="color: #64748b;">Upload a cow or buffalo image to test the <code>POST /predict</code> API endpoint.</p>
            
            <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
                <div id="dropText">📁 Click to choose an image (or drag & drop here)</div>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
                <img id="preview">
            </div>

            <button id="predictBtn" onclick="runPrediction()" disabled>Predict Breed</button>

            <div id="resultBox" class="result-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 id="breedTitle" style="margin:0; color:#0f172a;"></h3>
                    <span id="badge" class="badge"></span>
                </div>
                <p id="latencyText" style="color:#64748b; font-size:13px; margin: 8px 0;"></p>
                <strong>JSON Output:</strong>
                <pre><code id="jsonOutput"></code></pre>
            </div>
        </div>

        <script>
            let selectedFile = null;
            function handleFile(file) {
                if (!file) return;
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.getElementById('preview');
                    img.src = e.target.result;
                    img.style.display = 'block';
                    document.getElementById('dropText').innerText = file.name;
                    document.getElementById('predictBtn').disabled = false;
                };
                reader.readAsDataURL(file);
            }

            async function runPrediction() {
                if (!selectedFile) return;
                const btn = document.getElementById('predictBtn');
                btn.innerText = "Analyzing...";
                btn.disabled = true;

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const res = await fetch('/predict?top_k=3', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    document.getElementById('resultBox').style.display = 'block';
                    document.getElementById('breedTitle').innerText = data.predicted_breed;
                    document.getElementById('badge').innerText = data.confidence_percentage;
                    document.getElementById('latencyText').innerText = "Inference Speed: " + data.inference_time_ms + " ms | Status: " + data.status;
                    document.getElementById('jsonOutput').innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    alert('Error: ' + err.message);
                } finally {
                    btn.innerText = "Predict Breed";
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
