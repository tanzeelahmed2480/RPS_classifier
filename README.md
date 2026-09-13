Here is the complete text without any triple backticks or nested code blocks that cause formatting breaks. You can copy everything below in one go:

==================== COPY BELOW THIS LINE ====================

# RPS_classifier: Rock-Paper-Scissors Image Classification with PyTorch

A PyTorch-based Convolutional Neural Network (CNN) designed to classify hand images into Rock, Paper, or Scissors. The repository RPS_classifier includes everything needed for data loading, model architecture, training with early stopping, model checkpointing, and command-line inference.

---

## Project Overview

RPS_classifier trains a custom CNN model (RPSCNN) to recognize Rock, Paper, and Scissors hand gestures.

### Key Features

* Custom CNN Architecture: 3 convolutional blocks with ReLU activations, Max Pooling, and Adaptive Average Pooling, connected to a multi-layer classifier head with Dropout for regularization.
* Training Pipeline: Loss evaluation, Adam optimizer with weight decay, StepLR learning rate scheduling, model checkpointing based on best validation loss, and early stopping.
* CLI-Based Inference: Utility script to evaluate individual test image files directly from the command line using saved weights (.pth).

---

## Project Folder Structure (RPS_classifier)

RPS_classifier/
├── Data.py           # PyTorch Dataset loaders & image transformations
├── config.py         # Configuration parameters (paths, hyperparameters, device)
├── model.py          # PyTorch model definition (RPSCNN)
├── train.py          # Training loop script with validation & early stopping
├── inference.py      # Core inference utility logic
├── main.py           # CLI entry point to run predictions on images
├── requirements.txt  # Python package dependencies
└── Best_model/       # Saved best model state dict (.pth)

---

## Model Architecture (RPSCNN)

The RPSCNN model consists of a feature extractor and a classifier head:

1. Feature Extractor (nn.Sequential):
* Conv2d(3, 16, kernel_size=3, padding=1) -> ReLU() -> MaxPool2d(2, 2)
* Conv2d(16, 32, kernel_size=3, padding=1) -> ReLU() -> MaxPool2d(2, 2)
* Conv2d(32, 64, kernel_size=3, padding=1) -> ReLU() -> MaxPool2d(2, 2)
* AdaptiveAvgPool2d((16, 16))


2. Classifier (nn.Sequential):
* Dropout(0.5)
* Linear(64 * 16 * 16, 128) -> ReLU()
* Dropout(0.3)
* Linear(128, 3) (Outputs logits for Rock, Paper, Scissors)



---

## Installation & Setup

1. Clone the Repository:
git clone [https://github.com/your-username/RPS_classifier.git](https://www.google.com/search?q=https://github.com/your-username/RPS_classifier.git)
cd RPS_classifier
2. Create and Activate a Virtual Environment (Recommended):
* On Linux/macOS:
python3 -m venv venv
source venv/bin/activate
* On Windows:
python -m venv venv
venv\Scripts\activate


3. Install Dependencies:
pip install -r requirements.txt

---

## Usage

### 1. Training the Model

Ensure your image dataset is placed according to the paths defined in config.py (e.g., class subfolders for rock, paper, scissors inside data/train and data/validation).

To start training:
python train.py

### 2. Running Inference on an Image

To run predictions on a specific hand image:
python main.py --image_path path/to/sample_hand.jpg

If --image_path is omitted, main.py defaults to the image path configured in config.py.

---

## Training Run Example

Below is the standard output log from a sample training run with early stopping enabled:

Epoch 1/10
No Improvement: 0/1
Learning rate: 0.0007
Train Loss: 0.44
Train Accuracy: 0.82
Saved as best model.
Validation Loss: 0.61
Validation Accuracy: 0.77

Epoch 2/10
No Improvement: 0/1
Learning rate: 0.0007
Train Loss: 0.02
Train Accuracy: 1.00
Validation Loss: 0.65
Validation Accuracy: 0.87

Epoch 3/10
No Improvement: 1/1
Early Stopping triggered.

Train Accuracies: [0.82, 1.0]
Validation Accuracies: [0.77, 0.87]
Train Losses: [0.44, 0.02]
Validation Losses: [0.61, 0.65]

### Performance Summary

* Peak Validation Accuracy: 87.0% achieved at Epoch 2.
* Model Checkpointing: Epoch 1 achieved the lowest validation loss (0.61), saving its weights to Best_model/.
* Early Stopping: Triggered after Epoch 3 when validation loss increased (0.65), preventing overfitting on the training set (100% train accuracy).

---

## Dependencies

* torch >= 2.0.0
* torchvision
* pillow
* numpy
