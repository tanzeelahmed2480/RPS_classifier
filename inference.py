import torch
from PIL import Image
from Data import get_transforms
import config

def predict_image(image_path, model, device=config.DEVICE):
    model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(config.IMAGE_SIZE)
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        
    return output