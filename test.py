from pathlib import Path

import torch
from PIL import Image

from models import DigitClassifier, LetterClassifier


config = {
    'model': 'letter'
}
#image_path = "data/test_images/1.png"
image_path = "data/test_images/B.jpg"
image = Image.open(image_path)
if config['model'] == 'digit':
    model = DigitClassifier()
else:
    model = LetterClassifier()
if Path(f"{config['model']}_classifier.pth").is_file():
    model.load_state_dict(torch.load(f"{config['model']}_classifier.pth"))
    print("loaded model from disc")




if __name__ == "__main__":
    model.eval()
    predicted = model.inference(image)

    print(f"Predicted digit: {predicted}")