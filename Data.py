from torchvision import transforms
from torch.utils.data import DataLoader
import config
# Resizing and Scaling the images
def get_transforms(image_size=config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
# Adding labels to the images according to their respective place
def get_dataloaders(train_dir, val_dir, batch_size=config.BATCH_SIZE):
    transform = get_transforms()
    # Import ImageFolder locally to keep modularity flexible
    from torchvision.datasets import ImageFolder
    
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Run this script to test if the images are successfully loaded
if __name__ =="__main__":
  train_loader, val_loader = get_dataloaders(train_dir="./data/train", val_dir="./data/validation", batch_size=config.BATCH_SIZE)
  images, labels = next(iter(train_loader))
  print(f"Batch images shape: {images.shape}")  # Expected: [32, 3, 128, 128]
  print(f"Batch labels shape: {labels.shape}")  # Expected: [32]
