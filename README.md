# CRL Ultrasound Classifier + 🧠 AI Chat Assistant

This web application allows users to upload fetal ultrasound images and classify them as **"GOOD"** or **"BAD"** based on image quality, using a deep learning model. The app also includes a GPT-powered assistant that provides clinical explanations and answers questions about Crown-Rump Length (CRL).

## Features

- ✅ Upload ultrasound image and get model prediction
- 🔍 See classification **probability** from a trained CNN model
- 🤖 GPT-powered **AI Interpretation Agent** for clinical-style reasoning
- 💬 CRL chatbot for fetal growth Q&A

---

## Model

- **Architecture**: ResNet-50
- **Input size**: 224x224 RGB images
- **Output**: Binary classification (`good` or `bad`)
- **Training**: Trained using transfer learning on fetal CRL ultrasound dataset

---

## Web Services

### FastAPI Model Endpoint
- **URL**: `https://ultrasound-classification-webservice.onrender.com/predict/`
- Accepts: `POST` requests with image file
- Returns: JSON with prediction label and probability

Example response:
```json
{
  "prediction": "good",
  "probability": 0.92
}