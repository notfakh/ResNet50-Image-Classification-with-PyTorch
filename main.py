import torch
from torchvision import models, transforms
from PIL import Image
import requests
import json
import os

# Define the image path
image_path = "image.jpg"  # Replace with your image file path

# Check if the image exists
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: The image '{image_path}' was not found.")
    exit()

# Load the pre-trained ResNet50 model
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),               # Resize the shortest edge to 256 pixels
    transforms.CenterCrop(224),           # Center crop to 224x224 pixels
    transforms.ToTensor(),                # Convert the image to a tensor
    transforms.Normalize(                 # Normalize with ImageNet's mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Preprocess the input image
print("Preprocessing the image...")
input_tensor = preprocess(img).unsqueeze(0)  # Add a batch dimension

# Perform inference
print("Classifying the image...")
with torch.no_grad():
    outputs = model(input_tensor)

# Download ImageNet class labels if not already available
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_file = "imagenet_classes.json"

if not os.path.exists(labels_file):
    print("Downloading ImageNet class labels...")
    with open(labels_file, "w") as f:
        f.write(requests.get(labels_url).text)

# Load the class labels
with open(labels_file, "r") as f:
    labels = json.load(f)

# Get the top predicted class
_, predicted_idx = outputs.max(1)
predicted_class = labels[predicted_idx.item()]

# Print the classification result
print(f"The image is classified as: {predicted_class}")
