from enum import Enum
from pathlib import Path

import matplotlib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

import models
from Chars74KDataset import Chars74KDataset
from models import DigitClassifier, LetterClassifier

matplotlib.use('QtAgg')


class Dataset(Enum):
    EMNIST = "emnist"
    MNIST = "mnist"


config = {
    'epochs': 5,
    'batch_size': 32,
    'lr': 0.001,
    'model': 'letter',  # digit,letter
    'mode': 'combines',  # solo, combined
    'loss_function': nn.CrossEntropyLoss,
}

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

chars74k_train = Chars74KDataset(train=True, transform=transform, download=True)
chars74k_test = Chars74KDataset(train=False, transform=transform, download=True)

mnist_train = datasets.MNIST(root="./data", train=True,
                             download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False,
                            download=True, transform=transform)

chars74k_path = "./data/Chars74K/English/Hnd/Img"
chars74k_dataset = Chars74KDataset(chars74k_path, transform=transform)

# Load EMNIST dataset (letters only)
emnist_train = datasets.EMNIST(root="./data", split="byclass",
                               train=True, transform=transform, download=True)
emnist_test = datasets.EMNIST(root="./data", split="byclass",
                              train=False, transform=transform, download=True)
if config['model'] == 'letter':
    model = LetterClassifier()
    train_dataset = emnist_train
    test_dataset = emnist_test
    if Path("letter_classifier.pth").is_file():
        model.load_state_dict(torch.load("letter_classifier.pth"))
        print("loaded model from disc")
else:
    model = DigitClassifier()
    train_dataset = mnist_train
    test_dataset = mnist_test
    if Path("digit_classifier.pth").is_file():
        model.load_state_dict(torch.load("digit_classifier.pth"))
        print("loaded model from disc")

train_loader = DataLoader(train_dataset,
                          batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=config['batch_size'], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available


def show_images(count: int):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img, label = train_dataset[i]  # Get image and label

        ax.imshow(img.squeeze(), cmap="gray")  # Remove extra dimension
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.show()


def train(epochs: int):
    criterion = config['loss_function']()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for _ in tqdm(range(epochs), desc="Epochs"):
        for images, labels in train_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate():
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device)
            total += labels.size(0)
            combination = (predicted == labels)
            correct += combination.int().sum()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def predict():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    images, labels = images.to('cpu'), labels.to('cpu')
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()  # Remove extra dimensions
        ax.imshow(img, cmap="gray")
        if config['model'] == DigitClassifier:
            ax.set_title(f"Pred: {predicted[i].item()} | True: {labels[i].item()}")
        else:
            ax.set_title(f"Pred: {models.emnist_mapping(predicted[i].item())} "
                         f"| True: {models.emnist_mapping(labels[i].item())}")
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    # show_images(10)

    train(config['epochs'])
    accuracy = evaluate()
    if accuracy >= 70.:
        if config['model'] == LetterClassifier:
            torch.save(model.state_dict(), "letter_classifier.pth")
        else:
            torch.save(model.state_dict(), "digit_classifier.pth")
        print("Model saved successfully!")

        predict()
