import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from tqdm.notebook import trange
import time
import pandas as pd  # Add this import for saving the report as CSV

from Astroconformer.Astroconformer.Model.models import Astroconformer as AstroConformer
from torch.cuda.amp import autocast, GradScaler

# Define directories as constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../../outputs")

scaler = GradScaler()  # Escalador global

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AstroConformerClassifier(nn.Module):
    def __init__(self, args, num_classes, feature_dim=7, freeze_encoder=False):  # Cambiar feature_dim a 7
        super().__init__()
        self.encoder = AstroConformer(args)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=args.dropout)
        #self.classifier = nn.Linear(args.encoder_dim, num_classes)
        # Añadimos un head mas profundo
        self.classifier = nn.Sequential(
            nn.Linear(args.encoder_dim + feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, args.output_dim)
        )

    def forward(self, x, mask, features):
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

        # Validar shapes y contenido antes de concatenar
        assert torch.isfinite(out).all(), "❌ out contiene NaN antes de concat"
        assert torch.isfinite(features).all(), "❌ features contiene NaN"
        assert out.shape[0] == features.shape[0], f"❌ batch_size mismatch: out {out.shape}, features {features.shape}"
        assert features.shape[1] == 7, f"❌ features debe tener 7 columnas: got {features.shape[1]}"

        # Concatenar features
        out = torch.cat([out, features], dim=1)
        assert torch.isfinite(out).all(), "❌ out contiene NaN después de concat"

        logits = self.classifier(self.dropout(out))
        assert torch.isfinite(logits).all(), "❌ logits contiene NaN"
        return logits


def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()
    for x, y, mask, features in loader:  # Añadir features al dataloader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Reparar y limitar valores anómalos en x
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-5.0, max=5.0)
        #print(f"To device, optimizer y clamp: {time.time() - start:.4f}s")

        # Forward + loss con autocast (float16 por defecto en CUDA)
        #start = time.time()
        #with autocast():
        #    outputs = model(x, mask, features)  # Pasar features al forward
        #    loss = criterion(outputs, y)
        #    assert torch.isfinite(loss), "loss es NaN"

        with autocast():
            outputs = model(x, mask, features)  # Pasar features al forward

            # Verifica logits
            if not torch.isfinite(outputs).all():
                print(f"❌ [Epoch {epoch}] Logits contienen NaN o Inf")
                print("  Logits stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                print("  Sample logits:", outputs[:3])
                print("  Labels:", y[:3])
                raise ValueError("Logits inválidos")

            # Verifica etiquetas
            if not torch.isfinite(y).all() or (y.min() < 0 or y.max() >= outputs.size(1)):
                print(f"❌ [Epoch {epoch}] Etiquetas fuera de rango o inválidas: {y}")
                raise ValueError("Etiquetas fuera de rango")

            # Calcula loss
            loss = criterion(outputs, y)

            # Verifica loss
            if not torch.isfinite(loss):
                print(f"❌ [Epoch {epoch}] Loss es NaN o Inf")
                print("  Loss:", loss)
                print("  Logits (sample):", outputs[:3])
                print("  Labels (sample):", y[:3])
                print("  Class weights:", criterion.weight if hasattr(criterion, 'weight') else "N/A")
                raise ValueError("Loss inválido")
        #print(f"Forward + loss time: {time.time() - start:.4f}s")

        # Backward con GradScaler (evita NaNs y es más rápido en float16)
        #start = time.time()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        loss.backward()
        optimizer.step()
        #print(f"Backward pass ONLY: {time.time() - start:.4f}s")

        # Acumulación de métricas sin sincronización
        #start = time.time()
        # Sin .item() por batch (esto es GPU friendly)
        total_loss += loss.detach()
        correct += (outputs.argmax(1) == y).sum()
        total += y.size(0)
        #print(f"Acumulacion metricas time: {time.time() - start:.4f}s")

    print(f"[TRAIN] TIEMPO ÉPOCA: {time.time() - start:.4f}s")
    #return total_loss / len(loader), correct / total
    return total_loss.item() / len(loader), correct.item() / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    start = time.time()
    for x, y, mask, features in loader:  # Añadir features al dataloader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        # Reparar y limitar valores anómalos en x
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-5.0, max=5.0)

        with autocast():  # Consistencia con train()
            outputs = model(x, mask, features)  # Pasar features al forward
            loss = criterion(outputs, y)

        total_loss += loss.detach().cpu().item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    print(f"[VAL] TIEMPO ÉPOCA: {time.time() - start:.4f}s")    
    return total_loss / len(loader), correct / total, report


def main(train_loader, val_loader, label_encoder, device="cuda", epochs=50, lr=3e-5, freeze_encoder=False, patience=6, debug=False):
    # Activar optimización de CuDNN
    torch.backends.cudnn.benchmark = True

    num_classes = len(label_encoder)

    args = argparse.Namespace(
        input_dim=1,
        in_channels=1,
        #encoder_dim=192,
        encoder_dim=256,
        #hidden_dim=256,
        hidden_dim=384,
        output_dim=num_classes,
        num_heads=8,
        #num_layers=6,
        num_layers=8,
        dropout=0.3, dropout_p=0.3,
        stride=20,
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )

    model = AstroConformerClassifier(args, num_classes, feature_dim=7, freeze_encoder=freeze_encoder).to(device)  # Cambiar feature_dim a 7
    model = torch.compile(model)

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique([y.item() for _, y, _, _ in train_loader.dataset]),  # Ignorar features
        y=[y.item() for _, y, _, _ in train_loader.dataset]  # Ignorar features
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Modelo en:", next(model.parameters()).device)

    for epoch in trange(1, epochs + 1 if not debug else 2, desc="Entrenamiento del modelo"):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
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
            torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, "mejor_modelo_optimizado.pt"))
            print(f"💾 Guardado modelo mejorado en {os.path.join(OUTPUTS_DIR, 'mejor_modelo_optimizado.pt')}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping activado tras {patience} épocas sin mejora.")
                break

    # Plot training curves
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
    plt.savefig(os.path.join(OUTPUTS_DIR, "curvas_entrenamiento_optimizado2.png"))
    plt.show()

    # Print and save the classification report
    print("\n📊 Classification Report:")
    print(report)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(OUTPUTS_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path, index=True)
    print(f"📁 Reporte guardado en: {report_csv_path}")

    return model

if __name__ == "__main__":
    # Activar optimización de CuDNN
    torch.backends.cudnn.benchmark = True

    # Load label_encoder from file
    with open(os.path.join(DATA_DIR, "train/label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # Define los argumentos del modelo
    args = argparse.Namespace(
        input_dim=1, in_channels=1,
        encoder_dim=192,
        hidden_dim=256,
        output_dim=7,  # Numero de clases
        num_heads=8, num_layers=6,
        dropout=0.3, dropout_p=0.3,
        stride=20, kernel_size=3,
        norm="postnorm", encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False, device="cuda"
    )

    # Instancia el modelo
    model = AstroConformerClassifier(args, num_classes=9).to("cuda")

    # Carga los pesos del modelo entrenado
    model.load_state_dict(torch.load(os.path.join(OUTPUTS_DIR, "mejor_modelo_optimizado.pt")))
