import torch
from pathlib import Path

# File Paths 
MODEL_PATH = Path("Best_model.pth")
DEFAULT_IMAGE_PATH = Path("sample_test.jpg")  #change this if you want to edit the script or else pass your image path in --image_path

# Image Parameters
IMAGE_SIZE = (128, 128)  # Resize values
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0007
WEIGHT_DECAY = 1e-4
PATIENCE = 1

# Hardware & Mapping 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # will try to train on gpu
CLASSES = {0: "paper", 1: "rock", 2: "scissors"}
