# Cattle & Buffalo Breed Classification API

Production-ready HTTP API using FastAPI and ONNX Runtime to classify 125 cattle and buffalo breeds in real time (~19 ms latency).

---

## 1. Quick Start

Run the server with one click using:
```bash
start_api.bat
```
Or via terminal:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive Swagger Docs / UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Prediction Endpoint:** `POST http://localhost:8000/predict`

---

## 2. API Endpoints

### `POST /predict`
Uploads an image file and receives breed classification in JSON.

- **URL:** `/predict?top_k=3`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `file`: The image file (JPG, PNG, JPEG, WEBP)
  - `top_k`: *(Optional, default: 3)* Number of top candidate breeds to return (1-10).

#### Sample JSON Response:
```json
{
  "success": true,
  "filename": "cow_sample.jpg",
  "predicted_breed": "Gir",
  "confidence": 0.8935,
  "confidence_percentage": "89.35%",
  "status": "High Confidence",
  "top_predictions": [
    {
      "rank": 1,
      "breed": "Gir",
      "confidence": 0.8935,
      "confidence_percentage": "89.35%"
    },
    {
      "rank": 2,
      "breed": "Kankrej",
      "confidence": 0.0031,
      "confidence_percentage": "0.31%"
    },
    {
      "rank": 3,
      "breed": "Sahiwal",
      "confidence": 0.0022,
      "confidence_percentage": "0.22%"
    }
  ],
  "inference_time_ms": 18.52,
  "model_version": "EfficientNet-B0-125Breeds-ONNX"
}
```

---

## 3. Code Integration Examples for App Team

### A. cURL (Terminal / Postman)
```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/cow_image.jpg"
```

### B. JavaScript / React / React Native / Node.js
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/predict?top_k=3', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log('Predicted Breed:', data.predicted_breed);
console.log('Confidence:', data.confidence_percentage);
console.log('Top Breeds:', data.top_predictions);
```

### C. Flutter / Dart (Mobile Apps)
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<void> uploadCowImage(String filePath) async {
  var request = http.MultipartRequest(
    'POST', 
    Uri.parse('http://<YOUR_SERVER_IP>:8000/predict?top_k=3')
  );
  request.files.add(await http.MultipartFile.fromPath('file', filePath));
  
  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);
  
  if (response.statusCode == 200) {
    var data = jsonDecode(response.body);
    print("Predicted Breed: ${data['predicted_breed']}");
    print("Confidence: ${data['confidence_percentage']}");
  }
}
```

### D. Python
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("cow_sample.jpg", "rb")}
params = {"top_k": 3}

response = requests.post(url, files=files, params=params)
print(response.json())
```
