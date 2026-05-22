import os
import json
from pathlib import Path
from typing import Dict, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT", "ai-vs-human-generated-dataset-hw")
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "models")
DEFAULT_LOG_DIR = os.getenv("LOG_DIR", "runs")
DEFAULT_METRICS_DIR = os.getenv("METRICS_DIR", "metrics")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "models")


class ImageDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

        if "file_name" not in self.data.columns or "label" not in self.data.columns:
            raise ValueError("CSV must contain columns: file_name, label")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = str(self.data.iloc[idx]["file_name"])

        file_name = (
            file_name
            .replace("train_data/", "")
            .replace("test_data/", "")
        )

        path_variants = [
            self.root_dir / "train_data" / file_name,
            self.root_dir / "test_data" / file_name,
            self.root_dir / file_name,
        ]

        img_path = None
        for path in path_variants:
            if path.exists():
                img_path = path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {file_name}")

        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def make_loaders(data_root: str, train_part: str, test_part: str, batch_size: int, num_workers: int):
    train_dir = Path(data_root) / train_part
    test_dir = Path(data_root) / test_part

    train_csv = train_dir / "train.csv"
    test_csv = test_dir / "test.csv"

    train_dataset = ImageDataset(str(train_csv), str(train_dir), transform=get_transforms(train=True))
    test_dataset = ImageDataset(str(test_csv), str(test_dir), transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def build_model(device: torch.device):
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(device)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(epoch_loss)
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="eval"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(epoch_loss)
    return metrics


def save_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: str, model, metadata: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)


def load_checkpoint(path: str, model, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def ensure_bucket(bucket_name: str = S3_BUCKET):
    s3 = get_s3_client()
    buckets = s3.list_buckets().get("Buckets", [])
    names = [b["Name"] for b in buckets]
    if bucket_name not in names:
        s3.create_bucket(Bucket=bucket_name)


def upload_to_s3(local_path: str, key: str, bucket_name: str = S3_BUCKET):
    ensure_bucket(bucket_name)
    s3 = get_s3_client()
    s3.upload_file(local_path, bucket_name, key)
    print(f"Uploaded {local_path} to s3://{bucket_name}/{key}")


def download_from_s3(key: str, local_path: str, bucket_name: str = S3_BUCKET):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    s3.download_file(bucket_name, key, local_path)
    print(f"Downloaded s3://{bucket_name}/{key} to {local_path}")
