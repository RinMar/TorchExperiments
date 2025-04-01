import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms


def _is_background_white(image: Image):
    image_array = np.array(image)

    corners = [
        image_array[0, 0],  # Top-left
        image_array[0, -1],  # Top-right
        image_array[-1, 0],  # Bottom-left
        image_array[-1, -1]  # Bottom-right
    ]
    avg_corner_intensity = np.mean(corners)
    return avg_corner_intensity > 127


def _binarize_image(image: Image, threshold: int = 128):
    image = image.convert("L")
    image = image.point(lambda x: 255 if x > threshold else 0)
    return image


def _prepare_image(image: Image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
        transforms.Resize((28, 28)),  # Resize to match MNIST format
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1]
    ])
    image = _binarize_image(image, 128)
    if _is_background_white(image):
        image = ImageOps.invert(image)

    image.show()

    image = transform(image).unsqueeze(0)
    return image


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d(1),

        )
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def inference(self, image: Image):
        image = _prepare_image(image)
        with torch.no_grad():
            output = self(image)
            _, predicted = torch.max(output, 1)
            return predicted


class LetterClassifier(nn.Module):
    def __init__(self):
        super(LetterClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d(1),

        )
        self.fc = nn.Linear(64, 62)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def inference(self, image: Image):
        image = _prepare_image(image)
        with torch.no_grad():
            output = self(image)
            _, predicted = torch.max(output, 1)
            return predicted
