from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch
from torchvision import models
import io
import json

# Load trained model
model_path = "resnet34_food101.pth"
model = models.resnet34()
model.fc = torch.nn.Linear(model.fc.in_features, 15)  # Adjust to 15 classes
model.load_state_dict(torch.load(model_path))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return {"predicted_class": int(predicted_class)}
