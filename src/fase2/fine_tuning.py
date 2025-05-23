
# fine_tuning_with_progress.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import trange

import sys
astroconformer_path = os.path.abspath('./Astroconformer')
if astroconformer_path not in sys.path:
    sys.path.append(astroconformer_path)

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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        outputs = model(x, mask)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), all_preds, all_labels

def main(train_loader, val_loader, num_classes, device="cuda", epochs=15, lr=5e-6, patience=3):
    label_encoder_path = "data/train/label_encoder.pkl"
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.keys())

    args = argparse.Namespace(
        input_dim=1,
        in_channels=1,
        encoder_dim=128,
        hidden_dim=128,
        output_dim=num_classes,
        num_heads=8,
        num_layers=5,
        dropout=0.1,
        dropout_p=0.1,
        stride=20,
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )

    model = AstroConformerClassifier(args, num_classes, freeze_encoder=False).to(device)
    model.load_state_dict(torch.load("outputs/mejor_modelo.pt"))
    print("✅ Modelo base cargado desde outputs/mejor_modelo.pt")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in trange(1, epochs + 1, desc="Fine-tuning"):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(x, mask)
            loss = criterion(outputs, y)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(correct / total)

        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy_score(val_true, val_preds))

        print(f"\n🧪 Epoch {epoch}/{epochs}")
        print(f"Train loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}")
        print(f"Train acc: {train_accuracies[-1]:.4f}, Val acc: {val_accuracies[-1]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/mejor_modelo_finetuned.pt")
            print("💾 Guardado mejor modelo afinado (outputs/mejor_modelo_finetuned.pt)")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Fine-tuning detenido tras {patience} épocas sin mejora.")
                break

    os.makedirs("outputs", exist_ok=True)
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
    plt.savefig("outputs/curvas_finetuning.png")
    plt.show()

    # Matriz de confusión
    cm = confusion_matrix(val_true, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Matriz de Confusión (Val)")
    plt.tight_layout()
    plt.savefig("outputs/matriz_confusion_finetuning.png")
    plt.show()

    return model
