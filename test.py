from pathlib import Path

import torch
from PIL import Image

from models import DigitClassifier

#image_path = "data/test_images/1.png"
image_path = "data/test_images/colored_3.png"
image = Image.open(image_path)
model = DigitClassifier()
if Path("digit_classifier.pth").is_file():
    model.load_state_dict(torch.load("digit_classifier.pth"))
    print("loaded model from disc")




if __name__ == "__main__":
    model.eval()
    predicted = model.inference(image)

    print(f"Predicted digit: {predicted.item()}")