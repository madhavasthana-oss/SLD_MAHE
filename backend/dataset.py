import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ── Image size must match ResNet50 expectations ──────────────────────────────
imageSize = 224

# ── ImageNet normalization stats (required for pretrained ResNet50) ──────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Transforms ───────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

eval_transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# default transform (eval-safe, used when no explicit transform is passed)
transform = eval_transform


class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=eval_transform):
        self.root_dir  = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels      = []

        # Automatically get class names
        self.classes      = sorted(os.listdir(root_dir))
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


# ── Guard: only runs when this file is executed directly ─────────────────────
if __name__ == "__main__":
    dataset = ASLDataset(
        root_dir="data/asl_alphabet_train",
        transform=eval_transform
    )
    print("Total images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Num classes:", len(dataset.classes))
