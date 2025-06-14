
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np
from tqdm.notebook import trange

from Astroconformer.Astroconformer.Model.models import Astroconformer as AstroConformer

class AstroConformerClassifier(nn.Module):
    def __init__(self, args, num_classes, freeze_encoder=False):
        super().__init__()
        self.encoder = AstroConformer(args)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=args.dropout)
        self.classifier = nn.Linear(args.encoder_dim, num_classes)

    def forward(self, x, mask):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if mask.dim() > 2:
            mask = mask.view(mask.size(0), -1)
        x = x * mask
        out = self.encoder.extractor(x.unsqueeze(1))
        out = out.permute(0, 2, 1)
        RoPE = self.encoder.pe(out, out.shape[1])
        out = self.encoder.encoder(out, RoPE)
        out = out.mean(dim=1)
        logits = self.classifier(self.dropout(out))
        return logits

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        outputs = model(x, mask)
        loss = criterion(outputs, y)
        if torch.isnan(loss): continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        outputs = model(x, mask)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return total_loss / len(loader), correct / total, report

def main(train_loader, val_loader, num_classes, device="cuda", epochs=50, lr=1e-5, freeze_encoder=True, patience=5, debug=False):
    with open("data/train/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    args = argparse.Namespace(
        input_dim=1, in_channels=1,
        encoder_dim=192,          # ← antes 128
        hidden_dim=256,           # ← antes 128
        output_dim=num_classes,
        num_heads=8, num_layers=6,  # ← antes 5
        dropout=0.3, dropout_p=0.3,
        stride=20, kernel_size=3,
        norm="postnorm", encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False, device=device
    )
    
    model = AstroConformerClassifier(args, num_classes, freeze_encoder=freeze_encoder).to(device)

    all_labels = [y.item() for _, y, _ in train_loader.dataset]
    class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in trange(1, epochs + 1 if not debug else 2, desc="Entrenamiento del modelo"):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, report = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)

        print(f"\n🧪 Epoch {epoch}/{epochs}")
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/mejor_modelo_optimizado.pt")
            print("💾 Guardado modelo mejorado en outputs/mejor_modelo_optimizado.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping activado tras {patience} épocas sin mejora.")
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Curva de Pérdida")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.title("Curva de Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/curvas_entrenamiento_optimizado.png")
    plt.show()
    return model
