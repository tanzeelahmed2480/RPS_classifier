import argparse
from pathlib import Path
import torch
import config
from model import RPSCNN
from inference import predict_image

# Main Function to predict
def run():
    # Getting arguments if passed to script
    parser = argparse.ArgumentParser(description="Run inference on Rock-Paper-Scissors image.")
    parser.add_argument("--image_path", type=Path, nargs="?", default=None, help="Path to image file")
    args = parser.parse_args()
    # If no arguments are passed it will use the default image path in the config so make sure to change it 
    image_path = args.image_path if args.image_path is not None else config.DEFAULT_IMAGE_PATH
    # Check if file is image and available
    if not image_path.is_file():
        print(f"Error: File '{image_path}' not found.")
        return

    if image_path.suffix.lower() not in config.VALID_EXTENSIONS:
        print(f"Error: '{image_path.name}' unsupported. Allowed: {', '.join(config.VALID_EXTENSIONS)}")
        return

    print(f"Processing: {image_path.resolve()}")
    # Loads the model
    model = RPSCNN(num_classes=len(config.CLASSES))
    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    # Performs Inference
    output = predict_image(image_path, model)
    predicted_idx = torch.argmax(output, dim=1).item()
    # Shows prediction
    print(f"The logits : {output}")
    print(f"The image is a {config.CLASSES[predicted_idx]}")

if __name__ == "__main__":
    run()
