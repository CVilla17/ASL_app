from fastapi import FastAPI,File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from torchvision import  models
import torch

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import os

# Image manipulations
from PIL import Image

from torch import Tensor, nn

from model import *
from io import BytesIO
import uvicorn 

#Define file paths
traindir = f"data/train"
validdir = f"data/valid"
testdir = f"data/test"

save_file_name = f'resnet50-transfer3.pt'
checkpoint_path = f'resnet50-transfer3.pth'

# Change to fit hardware
batch_size = 512

# Whether to train on a gpu
train_on_gpu = False
multi_gpu = False


#Transforms to get to image net standards that base model uses
#No augmentations on train data because it has already been augmented
image_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])  
    ]),
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}


# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}


#START OF RELEVANT SERVER CODE
app = FastAPI(title="ASL api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello_server():
    return "server is up"

def load_model():
    saved_model = "resnet50-transfer2.pt"
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features 
    n_classes = 26
    model.fc = nn.Sequential(
                      nn.Linear(n_inputs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, n_classes),                   
                      nn.LogSoftmax(dim=1))
    
    
    model.load_state_dict(torch.load(saved_model))
    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }
    print("Model loaded")
    return model




def predict_api(image_path):
    """
    Uses predict from model.py to return the top prediction for an image
    """
    return predict(image_path, model,3)[1]


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    print("in the request")
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    print("can't read image file")
    prediction = predict(image,model,server_used=True)

    response = {}
    response["predicted"] = prediction[2][0]
    response["confidence"] = prediction[1].item(0)
    


    return [response]


if __name__ == "__main__":
    model=load_model()
   
    uvicorn.run(app,port=8000)