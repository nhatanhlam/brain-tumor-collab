import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler, autocast
import json

# Custom Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, samples, feature_extractor, transform=None):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)["pixel_values"].squeeze()
        return image, torch.tensor(label)

# Function to evaluate model on test set
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = {i: 0 for i in range(4)}
    total = {i: 0 for i in range(4)}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(pixel_values=images).logits
                preds = outputs.argmax(dim=1)

            for label, pred in zip(labels, preds):
                total[label.item()] += 1
                if label == pred:
                    correct[label.item()] += 1

    accuracies = {i: (correct[i] / total[i] * 100) if total[i] > 0 else 0 for i in range(4)}
    return accuracies

# Training function
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)
    model.to(device)

    # Augmentation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3)
    ])

    # Load dataset
    data_dir = Path("data/Training")
    test_dir = Path("data/Testing")
    class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    samples = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Path not found: {class_dir}")
        for file_name in class_dir.iterdir():
            if file_name.is_file():
                samples.append((str(file_name), label))

    # Split into training and validation
    train_samples, val_samples = train_test_split(samples, test_size=0.2, stratify=[s[1] for s in samples])
    train_dataset = BrainTumorDataset(train_samples, feature_extractor, transform)
    val_dataset = BrainTumorDataset(val_samples, feature_extractor, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Prepare test dataset
    test_samples = []
    for label, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Test directory '{class_dir}' not found. Skipping...")
            continue
        for file_name in class_dir.iterdir():
            if file_name.is_file():
                test_samples.append((str(file_name), label))

    test_dataset = BrainTumorDataset(test_samples, feature_extractor, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Compute class weights
    labels = [label for _, label in train_samples]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Optimizer, scheduler, loss, and scaler for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    # Logging & model saving
    model_path = "models/best_vit_model.pth"
    log_file = "models/training_log.txt"
    os.makedirs("models", exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10

    try:
        with open(log_file, "w") as log:
            log.write("Epoch | Train Loss | Test Loss | Train Acc | Test Acc | Time (s)\n")
            log.write("-" * 70 + "\n")

            for epoch in range(100):
                start_time = time.time()
                model.train()
                total_loss = 0
                correct = 0

                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with autocast(device_type='cuda'):
                        outputs = model(pixel_values=images).logits
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    correct += (outputs.argmax(dim=1) == labels).sum().item()

                train_loss = total_loss / len(train_loader)
                train_acc = correct / len(train_dataset)

                # Validation
                model.eval()
                val_loss = 0
                correct = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        with autocast(device_type='cuda'):
                            outputs = model(pixel_values=images).logits
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            correct += (outputs.argmax(dim=1) == labels).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / len(val_dataset)
                elapsed_time = time.time() - start_time

                print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, Time {elapsed_time:.2f}s")

                log.write(f"{epoch+1:<5} | {train_loss:.4f} | {val_loss:.4f} | {train_acc:.4f} | {val_acc:.4f} | {elapsed_time:.2f}\n")

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered!")
                        break

                scheduler.step(val_loss)

    except KeyboardInterrupt:
        print("Training interrupted. Saving last checkpoint...")
        torch.save(model.state_dict(), "models/last_checkpoint.pth")

    # Evaluate final model on test set
    best_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    test_results = evaluate_model(best_model, test_loader, device)

    with open("models/test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    print("✅ Final test accuracy saved to `test_results.json`.")

if __name__ == "__main__":
    train_model()
