import io
import torch

from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from .model_loader import model, device

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

@app.get("/")
def home():
    return {"message": "CIFAR10 API is running"}

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)

        prediction = torch.argmax(logits, dim=1).item()

        confidence = probabilities[0, prediction].item()

    return {
        "class_name": CLASS_NAMES[prediction],
        "confidence": round(confidence * 100, 2)
    }