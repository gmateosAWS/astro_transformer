
# script_2_transformer_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import pandas as pd

# Añadir AstroConformer al path si es necesario
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

        print(f"📊 x.mean: {x.mean().item():.4f}, std: {x.std().item():.4f}, max: {x.max().item():.4f}, min: {x.min().item():.4f}")
        x = x * mask
        out = self.encoder.extractor(x.unsqueeze(1))
        out = out.permute(0, 2, 1)
        RoPE = self.encoder.pe(out, out.shape[1])
        out = self.encoder.encoder(out, RoPE)
        out = out.mean(dim=1)
        print(f"📈 out.mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        logits = self.classifier(self.dropout(out))
        return logits

def train(model, loader, optimizer, criterion, device, debug=False):
    model.train()
    total_loss = 0
    for i, (x, y, mask) in enumerate(loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        assert not torch.isnan(x).any(), "❌ Tensores con NaNs detectados en input"
        assert not torch.isinf(x).any(), "❌ Tensores con Inf detectados en input"
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            outputs = model(x, mask)
            loss = criterion(outputs, y)
            if torch.isnan(loss):
                print("❌ NAN detected in loss. Skipping batch.")
                continue
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        if debug:
            print("🧪 Diagnóstico completo tras primer batch (modo debug=True).")
            break
    return total_loss / max(1, i + 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, save_errors=False, label_encoder=None):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    misclassified = []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        outputs = model(x, mask)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        if save_errors:
            for i in range(len(y)):
                if preds[i] != y[i]:
                    misclassified.append({
                        "true_label": y[i].item(),
                        "pred_label": preds[i].item(),
                        "curve_mean": x[i].mean().item(),
                        "curve_std": x[i].std().item()
                    })

    if save_errors and misclassified:
        df_errors = pd.DataFrame(misclassified)
        if label_encoder:
            rev_map = {v: k for k, v in label_encoder.items()}
            df_errors["true_label"] = df_errors["true_label"].map(rev_map)
            df_errors["pred_label"] = df_errors["pred_label"].map(rev_map)
        os.makedirs("outputs", exist_ok=True)
        df_errors.to_csv("outputs/errores_mal_clasificados.csv", index=False)

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return total_loss / len(loader), report

def main(train_loader, val_loader, num_classes, device="cpu", epochs=50, lr=1e-5, freeze_encoder=True, patience=5, debug=False):
    label_encoder_path = "data/train/label_encoder.pkl"
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print("Label encoder (clase → índice):", label_encoder)
        print("Número de clases:", len(label_encoder))
    else:
        print("⚠️ No se ha encontrado label_encoder.pkl. Asegúrate de generarlo en el script 1.")
        label_encoder = None

    all_labels = [y.item() for _, y, _ in train_loader.dataset]
    print("Valores únicos en etiquetas:", sorted(set(all_labels)))
    print("Máximo índice:", max(all_labels))

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

    model = AstroConformerClassifier(args, num_classes, freeze_encoder=freeze_encoder).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        loss_train = train(model, train_loader, optimizer, criterion, device, debug)
        loss_val, report = evaluate(model, val_loader, criterion, device, save_errors=True, label_encoder=label_encoder)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

        print(f"\n🧪 Epoch {epoch}/{epochs}")
        print(f"Loss entrenamiento: {loss_train:.4f}")
        print(f"Loss validación  : {loss_val:.4f}")
        print("\n📊 Clasificación (val):")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"Clase {label:>8}: Precisión={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1-score']:.2f}")

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            torch.save(model.state_dict(), "outputs/mejor_modelo.pt")
            print("💾 Modelo mejorado guardado (mejor_modelo.pt).")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping activado tras {patience} épocas sin mejora.")
                break

        if debug:
            print("🛑 Modo debug activo. Entrenamiento detenido tras primera época.")
            break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolución de la pérdida")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/curva_loss.png")
    plt.show()

    return model
