import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# Image size (same as your code)
imageSize = 50

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Automatically get class names
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for image_name in os.listdir(class_folder):
                    image_path = os.path.join(class_folder, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms (resize + tensor conversion)
transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor()
])

# Load dataset
dataset = ASLDataset(
    root_dir="data/asl_alphabet_train",  # adjust if needed
    transform=transform
)


# Check
print("Total images:", len(dataset))
print("Classes:", dataset.classes)


