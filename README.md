# Fashion_MNIST_Model
CNN-based REST API that classifies fashion items (clothes, shoes, bags) from grayscale images into 10 categories with 88% accuracy — built with TensorFlow, FastAPI, and Docker. 
Learned and done through watching the tutorial 
## Tutorial Link 

https://www.youtube.com/watch?v=sb2tm3pu17k&t=2333s

# Fashion MNIST Image Classifier API

![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)

A CNN-based image classifier trained on Fashion MNIST, served as a 
REST API using FastAPI and containerized with Docker.

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 88.3% |
| Training Epochs | 10 |
| Model Size | ~2MB |

## Architecture
- 3x Conv2D layers with ReLU activation
- 2x MaxPooling layers
- Dense(64) + Dense(10) output layer
- Trained on 60,000 images, tested on 10,000

## Classes
T-shirt, Trouser, Pullover, Dress, Coat, 
Sandal, Shirt, Sneaker, Bag, Ankle boot

## Run Locally

### With Docker (recommended)
```bash
git clone https://github.com/YOUR_USERNAME/fashion-mnist-api
cd fashion-mnist-api

# Download model (see Model Setup below)
docker build -t fashion-mnist-api .
docker run -p 8000:8000 fashion-mnist-api
```

### Model Setup
The model is not stored in this repo due to file size.
Train it yourself using the notebook in `/model_training`

### Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_images/Sneaker.png"
```


## API Response
<img width="1460" height="80" alt="Screenshot 2026-03-25 154341" src="https://github.com/user-attachments/assets/c932b696-c7cf-4ab9-9e3e-d62068bbe965" />


## Tech Stack
- **Model**: TensorFlow / Keras CNN
- **API**: FastAPI + Uvicorn
- **Container**: Docker
- **Training**: Google Colab (T4 GPU)
