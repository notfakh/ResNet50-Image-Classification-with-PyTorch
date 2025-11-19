# ResNet50 Image Classification with PyTorch
A deep learning image classifier using the pre-trained ResNet50 model from PyTorch to identify objects in images. Trained on ImageNet dataset with 1000 different classes.

## 📋 Project Overview

This project leverages transfer learning with ResNet50, a powerful 50-layer deep convolutional neural network pre-trained on the ImageNet dataset. It can classify images into 1000 different categories including animals, objects, vehicles, and more.

## 🎯 What Does This Do?

- **Loads** pre-trained ResNet50 model (25.6M parameters)
- **Preprocesses** images using ImageNet normalization
- **Classifies** images into 1000 ImageNet categories
- **Downloads** class labels automatically (first run only)
- **Predicts** the most likely category for your image
- **Uses** state-of-the-art deep learning architecture

## 🔑 Key Features

- ✅ Pre-trained ResNet50 (no training required)
- ✅ Automatic image preprocessing
- ✅ ImageNet class label download
- ✅ Error handling for missing images
- ✅ Fast inference with PyTorch
- ✅ Support for various image formats (JPG, PNG, etc.)
- ✅ Easy-to-use single-file implementation

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
Internet connection (first run only, for model and labels download)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notfakh/resnet50-image-classifier.git
cd resnet50-image-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Add your image:
   - Place your image file in the project directory
   - Name it `image.jpg` (or update the path in the code)

### Usage

Run the script:
```bash
python image_classifier.py
```

**First Run:**
- Downloads ResNet50 model (~100MB)
- Downloads ImageNet class labels (~50KB)
- Both are cached for future use
Donwload here: [Google Drive](https://drive.google.com/drive/folders/1dW2szi0U2MRwFd95Fq5989BKZOwjsx9l?usp=sharing)
**Output Example:**
```
Loading pre-trained ResNet50 model...
Preprocessing the image...
Classifying the image...
The image is classified as: golden retriever
```

## 📊 ResNet50 Architecture

### Model Details:
- **Layers**: 50 deep convolutional layers
- **Parameters**: 25.6 million trainable parameters
- **Input Size**: 224x224 pixels (RGB)
- **Output**: 1000 class probabilities
- **Training Dataset**: ImageNet (14M images, 1000 classes)

### Why ResNet50?
- **Residual Connections**: Solves vanishing gradient problem
- **Deep Learning**: 50 layers for complex feature extraction
- **Pre-trained**: Achieves 76% top-1 accuracy on ImageNet
- **Transfer Learning**: Ready to use without training
- **Industry Standard**: Widely used in production systems

## 📈 Image Preprocessing Pipeline

### Step-by-Step Process:

1. **Resize (256px)**
   ```python
   transforms.Resize(256)
   ```
   - Resizes shortest edge to 256 pixels
   - Maintains aspect ratio

2. **Center Crop (224x224)**
   ```python
   transforms.CenterCrop(224)
   ```
   - Crops center 224x224 region
   - Required input size for ResNet50

3. **Convert to Tensor**
   ```python
   transforms.ToTensor()
   ```
   - Converts PIL Image to PyTorch tensor
   - Scales pixel values to [0, 1]

4. **Normalize**
   ```python
   transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
   ```
   - Uses ImageNet statistics
   - Standardizes input distribution

5. **Add Batch Dimension**
   ```python
   .unsqueeze(0)
   ```
   - Changes shape from [3, 224, 224] to [1, 3, 224, 224]
   - Required for model input

## 🎨 Supported Classes

### ImageNet 1000 Categories Include:

| Category | Examples |
|----------|----------|
| **Animals** | dogs, cats, birds, fish, insects |
| **Vehicles** | cars, trucks, airplanes, boats |
| **Objects** | furniture, tools, electronics |
| **Food** | fruits, vegetables, dishes |
| **Nature** | trees, flowers, landscapes |
| **Buildings** | houses, monuments, structures |

**Full list**: See `imagenet_classes.json` after first run

## 🛠️ Customization

### Classify Multiple Images

```python
import glob

for image_path in glob.glob("images/*.jpg"):
    img = Image.open(image_path)
    # Process and classify...
    print(f"{image_path}: {predicted_class}")
```

### Get Top-K Predictions

```python
# Get top 5 predictions instead of just top 1
_, indices = outputs.topk(5)
print("Top 5 predictions:")
for idx in indices[0]:
    print(f"  - {labels[idx.item()]}")
```

### Add Confidence Scores

```python
import torch.nn.functional as F

# Get probabilities
probabilities = F.softmax(outputs, dim=1)
confidence, predicted_idx = probabilities.max(1)

print(f"Predicted: {labels[predicted_idx.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")
```

### Use Different ResNet Models

```python
# ResNet18 (lighter, faster)
model = models.resnet18(pretrained=True)

# ResNet101 (deeper, more accurate)
model = models.resnet101(pretrained=True)

# ResNet152 (deepest, most accurate)
model = models.resnet152(pretrained=True)
```

### Enable GPU Acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)
```

## 💡 Use Cases

- **Object Recognition**: Identify objects in photos
- **Wildlife Monitoring**: Classify animals in camera traps
- **Product Categorization**: Auto-tag e-commerce products
- **Content Moderation**: Filter inappropriate images
- **Medical Imaging**: Pre-trained features for transfer learning
- **Educational Tools**: Learn about deep learning and CNNs
- **Art Analysis**: Identify subjects in paintings/photos

## 🔬 Extending the Project

Ideas for enhancement:

1. **Batch Processing**
   ```python
   from torch.utils.data import DataLoader
   # Process multiple images efficiently
   ```

2. **Visualization**
   ```python
   import matplotlib.pyplot as plt
   
   # Display image with prediction
   plt.imshow(img)
   plt.title(f"Prediction: {predicted_class}")
   plt.axis('off')
   plt.show()
   ```

3. **Grad-CAM Visualization**
   ```python
   # Show which parts of image influenced the prediction
   from pytorch_grad_cam import GradCAM
   ```

4. **Fine-tuning**
   ```python
   # Adapt the model for your specific dataset
   model.fc = torch.nn.Linear(2048, num_custom_classes)
   ```

5. **Real-time Webcam Classification**
   ```python
   import cv2
   
   cap = cv2.VideoCapture(0)
   while True:
       ret, frame = cap.read()
       # Classify frame...
   ```

6. **REST API Deployment**
   ```python
   from flask import Flask, request, jsonify
   
   @app.route('/predict', methods=['POST'])
   def predict():
       # Handle image upload and return prediction
   ```

7. **Compare Multiple Models**
   ```python
   models_dict = {
       'ResNet50': models.resnet50(pretrained=True),
       'VGG16': models.vgg16(pretrained=True),
       'DenseNet': models.densenet121(pretrained=True)
   }
   ```

## 📊 Model Performance

### ImageNet Validation Set Results:

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters |
|-------|----------------|----------------|------------|
| ResNet50 | 76.13% | 92.86% | 25.6M |
| ResNet18 | 69.76% | 89.08% | 11.7M |
| ResNet101 | 77.37% | 93.55% | 44.5M |
| ResNet152 | 78.31% | 94.05% | 60.2M |

**Top-1**: Correct prediction as #1 choice  
**Top-5**: Correct prediction in top 5 choices

## 🤝 Contributing

Contributions welcome! Enhancement ideas:

- Add web interface with Streamlit/Gradio
- Implement ensemble predictions
- Add Grad-CAM visualization
- Create mobile app version
- Add custom class training
- Implement image preprocessing visualization
- Add confidence threshold filtering
- Create batch processing script

## 👤 Author

**Fakhrul Sufian**
- GitHub: [@notfakh](https://github.com/notfakh)
- LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- PyTorch team for torchvision models
- ImageNet dataset creators
- ResNet paper authors (He et al., 2015)
