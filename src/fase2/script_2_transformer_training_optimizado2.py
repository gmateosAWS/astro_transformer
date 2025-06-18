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
    def __init__(self, args, num_classes, feature_dim=7, freeze_encoder=False):
        super().__init__()
        self.encoder = AstroConformer(args)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(p=args.dropout)

        # [MOD CNN] Añadimos una CNN 1D previa al extractor
        # | Capa                      | Propósito                                                 |
        # | ------------------------- | --------------------------------------------------------- |
        # | `Conv1d(1 → 8, kernel=7)` | Captura patrones locales más amplios con mayor capacidad. |
        # | `MaxPool1d(kernel=2)`     | Reduce a mitad la resolución temporal (más robustez).     |
        # | `Conv1d(8 → 4, kernel=5)` | Aprendizaje intermedio (fusión de patrones).              |
        # | `Conv1d(4 → 1, kernel=3)` | Reduce a 1 canal para compatibilidad con AstroConformer.  |
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),  # +expresividad
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1),  # Reconduce a 1 canal
        #     nn.ReLU()
        # )

        # Clasificador denso
        #self.classifier = nn.Linear(args.encoder_dim, num_classes)
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

        x = x * mask  # Aplica máscara de validez
        x = x.unsqueeze(1)  # [batch_size, 1, seq_length]

        # [MOD CNN] Aplicamos CNN 1D antes del extractor
        # x_cnn = self.cnn(x)  # <- guarda el resultado para debug
        # if self.training and not hasattr(self, '_printed_cnn_info'):
        #     print("✅ CNN aplicada: shape tras conv+pool:", x_cnn.shape)
        #     self._printed_cnn_info = True
        # x = x_cnn  # continúa con el flujo normal

        # Extracción con el AstroConformer
        out = self.encoder.extractor(x)
        out = out.permute(0, 2, 1)
        RoPE = self.encoder.pe(out, out.shape[1])
        out = self.encoder.encoder(out, RoPE)
        out = out.mean(dim=1)

        # Validaciones de integridad
        assert torch.isfinite(out).all(), "❌ out contiene NaN antes de concat"
        assert torch.isfinite(features).all(), "❌ features contiene NaN"
        assert out.shape[0] == features.shape[0], f"❌ batch_size mismatch: out {out.shape}, features {features.shape}"
        assert features.shape[1] == 7, f"❌ features debe tener 7 columnas: got {features.shape[1]}"

        # Concatenación y clasificación
        out = torch.cat([out, features], dim=1)
        assert torch.isfinite(out).all(), "❌ out contiene NaN después de concat"

        logits = self.classifier(self.dropout(out))
        assert torch.isfinite(logits).all(), "❌ logits contiene NaN"
        return logits


# === DATA AUGMENTATION ===
def apply_augmentation(
    x,
    modes=["gaussian", "jitter", "masking", "scaling", "offset", "flipping", "window_dropout"],
    p=0.4,
    sigma=0.01,
    jitter_shift=100,
    mask_len=500,
):
    """
    Aplica una secuencia de transformaciones de data augmentation sobre la curva x.
    - x: tensor [batch_size, seq_length]
    - modes: lista de strings con augmentations a aplicar ("gaussian", "jitter", "masking")
    - p: probabilidad total de aplicar cada transformación (independiente por modo)
    """
    x_aug = x.clone()

    for mode in modes:
        if torch.rand(1).item() < p:
            if mode == "gaussian":
                noise = torch.randn_like(x_aug) * sigma
                x_aug += noise

            elif mode == "jitter":
                shift = torch.randint(low=-jitter_shift, high=jitter_shift + 1, size=(1,)).item()
                x_aug = torch.roll(x_aug, shifts=shift, dims=1)

            elif mode == "masking":
                for i in range(x_aug.size(0)):
                    if x_aug.size(1) > mask_len + 1:
                        start = torch.randint(0, x_aug.size(1) - mask_len, (1,)).item()
                        x_aug[i, start:start + mask_len] = 0.0

            elif mode == "scaling":
                scale = torch.empty(x_aug.size(0), 1, device=x_aug.device).uniform_(0.9, 1.1)
                x_aug *= scale

            elif mode == "offset":
                offset = torch.empty(x_aug.size(0), 1, device=x_aug.device).uniform_(-0.05, 0.05)
                x_aug += offset

            elif mode == "flipping":
                x_aug = torch.flip(x_aug, dims=[1])

            elif mode == "window_dropout":
                for i in range(x_aug.size(0)):
                    for _ in range(5):
                        start = torch.randint(0, x_aug.size(1) - 50, (1,)).item()
                        x_aug[i, start:start + 50] = 0.0
    return x_aug


def train(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for batch in loader:
        # Adaptar para aceptar datasets con o sin IDs
        if len(batch) == 5:
            x, y, mask, features, _ = batch  # Ignorar IDs en entrenamiento
        else:
            x, y, mask, features = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True) # Añadir features al dataloader

        # === AUGMENTATION === aplicar solo en entrenamiento
        # === AUGMENTATION === aplicar solo en entrenamiento
        # x = apply_augmentation(
        #     x,
        #     modes=["gaussian", "jitter", "masking"],  # Tres transformaciones moderadas
        #     p=0.4,           # Probabilidad individual de aplicar cada una (40%)
        #     sigma=0.01,      # Ruido gaussiano leve (1% de la escala normalizada)
        #     jitter_shift=100,  # Desplazamiento máximo de 100 puntos (0.4% del total)
        #     mask_len=500       # Bloque a enmascarar de hasta 500 puntos (2%)
        # )

        optimizer.zero_grad()

        # Reparar y limitar valores anómalos en x
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-5.0, max=5.0)

        outputs = model(x, mask, features)  # Pasar features al forward

        # Verifica logits
        if not torch.isfinite(outputs).all():
            print(f"❌ [Epoch {epoch}] Logits contienen NaN o Inf")
            print("  Logits stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
            print("  Sample logits:", outputs[:3])
            print("  Labels:", y[:3])
            raise ValueError("Logits inválidos")

        # Clamp logits solo en Epoch 1 para evitar explosiones numéricas
        logits = torch.clamp(outputs, min=-10, max=10)
        #logits = outputs

        if not torch.isfinite(logits).all():
            print(f"❌ [Epoch {epoch}] Logits contienen NaN o Inf")
            print("  Logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())
            print("  Sample logits:", logits[:3])
            print("  Labels:", y[:3])
            raise ValueError("Logits inválido")

        # Verifica etiquetas
        if not torch.isfinite(y).all() or (y.min() < 0 or y.max() >= logits.size(1)):
            print(f"❌ [Epoch {epoch}] Etiquetas fuera de rango o inválidas: {y}")
            raise ValueError("Etiquetas fuera de rango")

        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            print(f"❌ [Epoch {epoch}] Loss es NaN o Inf")
            print("  Loss:", loss)
            print("  Logits (sample):", logits[:3])
            print("  Labels (sample):", y[:3])
            print("  Class weights:", criterion.weight if hasattr(criterion, 'weight') else "N/A")
            raise ValueError("Loss inválido")

        loss.backward()
        optimizer.step()

        # Acumulación de métricas sin sincronización
        # Sin .item() por batch (esto es GPU friendly)
        total_loss += loss.detach()
        correct += (logits.argmax(1) == y).sum()
        total += y.size(0)

    print(f"[TRAIN] TIEMPO ÉPOCA: {time.time() - start:.4f}s")
    #return total_loss / len(loader), correct / total
    return total_loss.item() / len(loader), correct.item() / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    start = time.time()
    for batch in loader:
        # Adaptar para aceptar datasets con o sin IDs
        if len(batch) == 5:
            x, y, mask, features, _ = batch  # Ignorar IDs en evaluación
        else:
            x, y, mask, features = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        # Reparar y limitar valores anómalos en x
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-5.0, max=5.0)

        outputs = model(x, mask, features)  # Pasar features al forward
        outputs = torch.clamp(outputs, min=-10, max=10)  # Clamp fijo en evaluación

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

        loss = criterion(outputs, y)

        # Verifica loss
        if not torch.isfinite(loss):
            print(f"❌ [Epoch {epoch}] Loss es NaN o Inf")
            print("  Loss:", loss)
            print("  Logits (sample):", outputs[:3])
            print("  Labels (sample):", y[:3])
            print("  Class weights:", criterion.weight if hasattr(criterion, 'weight') else "N/A")
            raise ValueError("Loss inválido")

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
        encoder_dim=256, # 192
        hidden_dim=384, # 256
        output_dim=num_classes,
        num_heads=8, # 6,
        num_layers=8,
        dropout=0.3, dropout_p=0.3,
        stride=32, #20
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )

    model = AstroConformerClassifier(args, num_classes, feature_dim=7, freeze_encoder=freeze_encoder).to(device)  # Cambiar feature_dim a 7
    #model = torch.compile(model)

    print(model)

    # Adaptar para datasets con o sin IDs (5 elementos si incluye IDs)
    def get_label_from_sample(sample):
        return sample[1]

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique([get_label_from_sample(sample).item() for sample in train_loader.dataset]),
        y=[get_label_from_sample(sample).item() for sample in train_loader.dataset]
    )
    print("🔍 Pesos de clase (antes del clip):", class_weights)
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
        input_dim=1,
        in_channels=1,
        encoder_dim=256, # 192
        hidden_dim=384, # 256
        output_dim=num_classes,
        num_heads=8, # 6,
        num_layers=8,
        dropout=0.3, dropout_p=0.3,
        stride=32, #20
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )

    # Instancia el modelo
    model = AstroConformerClassifier(args, num_classes=9).to("cuda")

    # Carga los pesos del modelo entrenado
    model.load_state_dict(torch.load(os.path.join(OUTPUTS_DIR, "mejor_modelo_optimizado.pt")))
