import torch
import torch.nn as nn

# The model architecture
class RPSCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__() # Activating the nn.Module
        # Extracting features from Images
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # First conv layer takes the RGB 3 channel input 
            nn.ReLU(), # Activation
            nn.MaxPool2d(2, 2), # Pooling to reduce the dimensions by half
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Second conv layer takes the 16 channel ouput from first layer
            nn.ReLU(), # Activation 
            nn.MaxPool2d(2, 2), # Pooling again
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Takes the 32 channels from 2nd layer and output 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.AdaptiveAvgPool2d((16, 16)) # Finally pooling to a fixed size of 16x16 
        )
        # Feeding the extracted features to the classifier to predict
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # It will drop half the features randomly to prevent over fitting 
            nn.Linear(64 * 16 * 16, 128), # learn 128 new features from all the 16,384 features for a single image
            nn.ReLU(), # Activation
            nn.Dropout(p=0.3), # Again Dropout
            nn.Linear(128, num_classes) # Predict for 3 classes from 128 features
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flattens the 2D tensor into a vector
        x = self.classifier(x)
        return x
