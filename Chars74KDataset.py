import os
import tarfile
import urllib
from glob import glob

import PIL
import requests
from torch.utils.data import Dataset


class Chars74KDataset(Dataset):
    URL = "https://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"
    DATASET_DIR = "./data/Chars74K/English/Hnd/Img"
    BASE_DIR = "./data/Chars74K"

    def __init__(self, train=True, transform=None, split_ratio=0.9, download=True):
        """
        Load Chars74K dataset and split into train/test based on the train flag.

        :param train: If True, returns the training set; if False, returns the test set.
        :param transform: Image transformations.
        :param split_ratio: Ratio of training data (default 90% train, 10% test).
        :param download: If True, downloads the dataset automatically if missing.
        """

        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.root_dir = self.DATASET_DIR

        if download:
            self._download_and_extract()

        valid_labels = {chr(i) for i in range(65, 91)} | {chr(i) for i in range(97, 123)}  # A-Z, a-z
        for char_folder in os.listdir(self.root_dir):
            if char_folder in valid_labels:
                label = char_folder
                char_folder_path = os.path.join(self.root_dir, char_folder)

                for img_path in glob(f"{char_folder_path}/*.png"):  # Adjust if images are .jpg or .bmp
                    self.image_paths.append(img_path)
                    self.labels.append(label)
        # Convert labels to indices (a=0, b=1, ..., z=25)
        self.labels = [ord(label.lower()) - 97 for label in self.labels]

        # Split into train/test
        split_idx = int(split_ratio * len(self.image_paths))
        if train:
            self.image_paths = self.image_paths[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.labels = self.labels[split_idx:]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = PIL.Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        # Convert label (A-Z, a-z) to numeric index
        label_index = ord(label.lower()) - 97  # 'a' -> 0, 'b' -> 1, ..., 'z' -> 25

        return image, label_index

    def _download_and_extract(self):
        """Downloads and extracts Chars74K if not found."""
        dataset_path = os.path.dirname(self.root_dir)  # Parent directory
        tgz_path = self.BASE_DIR + "/EnglishHnd.tgz"

        if not os.path.exists(self.root_dir):
            os.makedirs(tgz_path, exist_ok=True)
            print(f"Downloading Chars74K dataset from {self.URL}...")
            response = requests.get(self.URL, stream=True, verify=False)
            with open(tgz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                f.close()
            print("Download complete! Extracting...")

            # Extract the dataset
            with tarfile.open(tgz_path, "r:gz") as tar_ref:
                tar_ref.extractall(dataset_path)

            print("Extraction complete!")
        else:
            print("Dataset already exists. Skipping download.")
