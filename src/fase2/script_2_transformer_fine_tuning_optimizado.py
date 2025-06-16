import os
import torch
torch.set_float32_matmul_precision('high')  # ✅ Activa uso de Tensor Cores para float32
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from tqdm.notebook import trange
import pandas as pd
from Astroconformer.Astroconformer.Model.models import Astroconformer as AstroConformer
from src.utils.focal_loss import FocalLoss
from torch.cuda.amp import autocast
import time

# Define directories as constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../../outputs")

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
        # Concatenar features con la salida del encoder
        out = torch.cat([out, features], dim=1)
        logits = self.classifier(self.dropout(out))
        return logits



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y, mask, features in loader:  # Añadir features al dataloader
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)

        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, min=-5.0, max=5.0)

        outputs = model(x, mask, features)  # Pasar features al forward
        outputs = torch.clamp(outputs, min=-10, max=10)  # Añadido para evitar NaNs

        loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            print(f"❌ Epoch {epoch}: Loss inválido")
            print("  Loss:", loss)
            print("  Logits (sample):", outputs[:3])
            print("  Labels (sample):", y[:3])
            raise ValueError("Loss inválido")

        # total_loss += loss.detach().cpu().item()
        total_loss += loss.detach()
        all_preds.extend(outputs.argmax(1).detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    #return total_loss / len(loader), all_preds, all_labels
    return total_loss.item() / len(loader), all_preds, all_labels

    
def main(train_loader, val_loader, label_encoder, model_name="mejor_modelo_optimizado.pt", device="cuda", epochs=20, patience=4, debug=False,
         freeze_encoder=True, freeze_epochs=5, encoder_lr=2e-6, head_lr=1e-5, gamma=3.0, use_scheduler=True):
    # Activar optimización de CuDNN
    torch.backends.cudnn.benchmark = True

    num_classes = len(label_encoder)
    class_names = list(label_encoder.keys())
    
    args = argparse.Namespace(
        input_dim=1,
        in_channels=1,
        encoder_dim=256, # 192
        hidden_dim=384, # 256
        output_dim=num_classes,
        num_heads=8, # 6,
        num_layers=8,
        dropout=0.4, dropout_p=0.4,
        stride=32,
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )
    model = AstroConformerClassifier(args, num_classes, feature_dim=7, freeze_encoder=freeze_encoder).to(device)  # Cambiar feature_dim a 7

    # Carga los pesos del modelo entrenado
    model_path = os.path.join(OUTPUTS_DIR, model_name)
    state_dict = torch.load(model_path, map_location=device)    
    # Si el modelo fue compilado, puede tener prefijo _orig_mod.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("⚠️ Detected _orig_mod. prefix in state_dict. Stripping prefixes...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    # Compilar el modelo para mejorar rendimiento (solo una vez)
    model = torch.compile(model)
    print(f"✅ Modelo cargado desde {model_path}")

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.classifier.parameters(), "lr": head_lr}
    ], weight_decay=1e-4)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',         # minimizar val_loss
            factor=0.5,         # reduce LR a la mitad
            patience=3,         # espera 3 epochs sin mejora
            min_lr=1e-7         # no baja de este valor
        )

    #criterion = FocalLoss(gamma=gamma, reduction="mean", label_smoothing=0.1)
    # El label_smoothing=0.1 puede estar difuminando las etiquetas reales, afectando a la capacidad del modelo de tomar decisiones claras. 
    # Esto es útil cuando hay overfitting, pero aquí hay infraajuste (underfitting).
    # Quitar label_smoothing hará que el modelo se esfuerce más en acertar la clase correcta en vez de repartir confianza difusa.
    #criterion = FocalLoss(gamma=gamma, reduction="mean")
    # Probamos con CrossEntropyLoss: estable, bien conocida, buena en datasets equilibrados (como el tuyo).
    # label_smoothing=0.05 suaviza decisiones sin perder foco.
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("Modelo en:", next(model.parameters()).device)

    for epoch in trange(1, epochs + 1 if not debug else 2, desc="Fine-tuning"):
        epoch_start = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        if freeze_encoder and epoch == freeze_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = True
            # ⚠️ Reconfigura el optimizer para aplicar correctamente el learning rate al encoder ya activo
            optimizer = optim.AdamW([
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.classifier.parameters(), "lr": head_lr}
            ], weight_decay=1e-4)
            print(f"🔓 Encoder descongelado y optimizador actualizado en epoch {epoch}")

        t_train = time.time()
        for i, (x, y, mask, features) in enumerate(train_loader):
            #batch_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)

            # Reparar y limitar valores anómalos en x
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            x = torch.clamp(x, min=-5.0, max=5.0)

            optimizer.zero_grad()

            outputs = model(x, mask, features)
            outputs = torch.clamp(outputs, min=-10, max=10)  # clamp logits

            if not torch.isfinite(outputs).all():
                print(f"❌ Epoch {epoch}: Logits inválidos")
                print("  Logits stats:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
                print("  Sample logits:", outputs[:3])
                print("  Labels:", y[:3])
                raise ValueError("Logits inválidos")

            if not torch.isfinite(y).all() or (y.min() < 0 or y.max() >= outputs.size(1)):
                print(f"❌ Epoch {epoch}: Etiquetas fuera de rango o inválidas: {y}")
                raise ValueError("Etiquetas fuera de rango")

            loss = criterion(outputs, y)

            if not torch.isfinite(loss):
                print(f"❌ Epoch {epoch}: Loss inválido")
                print("  Loss:", loss)
                raise ValueError("Loss inválido")

            loss.backward()
            optimizer.step()

            total_loss += loss.detach()
            correct += (outputs.argmax(1) == y).sum()
            total += y.size(0)

            #torch.cuda.synchronize()

        if total == 0:
            print(f"⚠️ Epoch {epoch}: no se procesaron muestras. Posible error en el dataloader o en el modelo.")
            train_losses.append(0)
            train_accs.append(0)
            continue

        # Convertimos a escalar al final para guardar resultados
        train_losses.append(total_loss.item() / len(train_loader))
        train_accs.append(correct.item() / total)
        print(f"⏱️ Tiempo entrenamiento: {time.time() - t_train:.2f}s")

        t_eval = time.time()
        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(accuracy_score(val_true, val_preds))
        print(f"🔍 Tiempo evaluación: {time.time() - t_eval:.2f}s")

        # Añadir scheduler.step(val_loss)
        if use_scheduler and scheduler is not None:
            scheduler.step(val_loss)

        print(f"\n🧪 Epoch {epoch}/{epochs}")
        print(f"Train loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}")
        print(f"Train acc: {train_accs[-1]:.4f}, Val acc: {val_accs[-1]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, "mejor_modelo_finetuned_optimizado2.pt"))
            print(f"💾 Guardado mejor modelo fine-tuned en {os.path.join(OUTPUTS_DIR, 'mejor_modelo_finetuned_optimizado2.pt')}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Fine-tuning detenido tras {patience} épocas sin mejora.")
                break
        print(f"⏱️ Tiempo época: {time.time() - epoch_start:.2f}s")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Curva de Pérdida (Fine-tuning)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Curva de Accuracy (Fine-tuning)")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "curvas_finetuning_optimizado.png"))
    plt.show()

    cm = confusion_matrix(val_true, val_preds, labels=sorted(set(val_true) | set(val_preds)))
    display_names = [class_names[i] for i in sorted(set(val_true) | set(val_preds))]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Matriz de Confusión (Fine-tuning)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "matriz_confusion_finetuning_optimizado.png"))
    plt.show()

    errores = []
    for idx, (pred, true) in enumerate(zip(val_preds, val_true)):
        if pred != true:
            errores.append({
                "indice": idx,
                "clase_real": class_names[true],
                "clase_predicha": class_names[pred]
            })

    df_errores = pd.DataFrame(errores)
    df_errores.to_csv(os.path.join(OUTPUTS_DIR, "errores_mal_clasificados.csv"), index=False)
    print(f"💾 Guardado CSV con errores: {os.path.join(OUTPUTS_DIR, 'errores_mal_clasificados.csv')}")

    report = classification_report(val_true, val_preds, target_names=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(OUTPUTS_DIR, "class_report_finetuning_optimizado.csv"))
    print("📄 Reporte de clasificación guardado en class_report_finetuning_optimizado.csv")


    if debug:
        print("🛑 Debug activo: fine-tuning detenido tras primera época.")

    # Export the fine-tuned model to ONNX
    export_model_to_onnx(device=device)

    return model

def export_model_to_onnx(device="cuda"):
    """
    Exporta el modelo fine-tuned al formato ONNX para su visualización y despliegue.
    """
    import torch.onnx

    # Carga del codificador de etiquetas
    with open(os.path.join(DATA_DIR, "train/label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    num_classes = len(label_encoder)

    # Argumentos del modelo (deben coincidir con los usados en entrenamiento)
    args = argparse.Namespace(
        input_dim=1,
        in_channels=1,
        encoder_dim=256,
        hidden_dim=384,
        output_dim=num_classes,
        num_heads=8,
        num_layers=8,
        dropout=0.4, dropout_p=0.4,
        stride=32,
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )
    
    # Inicializa el modelo y carga pesos
    model = AstroConformerClassifier(args, num_classes, feature_dim=7).to(device)  # Cambiar feature_dim a 7
    state_dict = torch.load(os.path.join(OUTPUTS_DIR, "mejor_modelo_finetuned_optimizado2.pt"), map_location=device)

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("⚠️ Detected _orig_mod. prefix in state_dict. Stripping prefixes...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Preparar entrada dummy
    dummy_input = (
        torch.randn(1, 1, 25000).to(device),  # x
        torch.ones(1, 25000).bool().to(device),  # mask
        torch.randn(1, 7).to(device)  # features
    )

    # Exportar a ONNX
    onnx_path = os.path.join(OUTPUTS_DIR, "mejor_modelo_finetuned_optimizado2.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input", "mask", "features"],  # Añadir features
            output_names=["output"],
            opset_version=16,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes={
                "input": {0: "batch_size", 2: "sequence_length"},
                "mask": {0: "batch_size", 1: "sequence_length"},
                "features": {0: "batch_size"},  # Añadir features
                "output": {0: "batch_size"}
            }
        )
    print(f"✅ Modelo exportado a ONNX en: {onnx_path}")
