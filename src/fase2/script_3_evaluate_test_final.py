import os
import time
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from tqdm import trange

from src.utils.focal_loss import FocalLoss
from src.fase2.model import AstroConformerClassifier
from src.fase2.evaluate import evaluate
from src.utils.directories import OUTPUTS_DIR

def main(train_loader, val_loader, num_classes, device="cuda", epochs=30, patience=5, debug=False,
         freeze_encoder=True, freeze_epochs=5, encoder_lr=1e-6, head_lr=5e-6, gamma=3):

    model = AstroConformerClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUTS_DIR, "mejor_modelo_optimizado.pt"), map_location=device))
    print(f"✅ Modelo cargado desde {os.path.join(OUTPUTS_DIR, 'mejor_modelo_optimizado.pt')}")
    print(f"Modelo en: {device}")

    # Inicialmente congela el encoder
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.classifier.parameters(), "lr": head_lr}
    ], weight_decay=1e-4)

    scaler = GradScaler()
    criterion = FocalLoss(gamma=gamma, reduction="mean", label_smoothing=0.1)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in trange(1, epochs + 1 if not debug else 2, desc="Fine-tuning"):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()

        if freeze_encoder and epoch == freeze_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.classifier.parameters(), "lr": head_lr}
            ], weight_decay=1e-4)
            print(f"🔓 Encoder descongelado y optimizador actualizado en epoch {epoch}")

        for x, y, mask in train_loader:
            start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            x = torch.clamp(x, min=-5.0, max=5.0)

            optimizer.zero_grad()

            with autocast():
                outputs = model(x, mask)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach().cpu().item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

            torch.cuda.synchronize()
            print(f"⏱️ Tiempo batch: {time.time() - start:.4f}s")

        if total == 0:
            print(f"⚠️ Epoch {epoch}: no se procesaron muestras. Posible error en el dataloader o en el modelo.")
            train_losses.append(0)
            train_accs.append(0)
            continue

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / total)

        eval_start = time.time()
        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)
        torch.cuda.synchronize()
        print(f"🔍 Tiempo evaluación: {time.time() - eval_start:.2f}s")

        val_losses.append(val_loss)
        val_accs.append(accuracy_score(val_true, val_preds))

        print(f"\n🧪 Epoch {epoch}/{epochs}")
        print(f"Train loss: {train_losses[-1]:.4f}, Val loss: {val_loss:.4f}")
        print(f"Train acc: {train_accs[-1]:.4f}, Val acc: {val_accs[-1]:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, "mejor_modelo_finetuned_optimizado.pt"))
            print(f"💾 Guardado mejor modelo fine-tuned en {os.path.join(OUTPUTS_DIR, 'mejor_modelo_finetuned_optimizado.pt')}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Fine-tuning detenido tras {patience} épocas sin mejora.")
                break

        print(f"⏱️ Tiempo total entrenamiento: {time.time() - epoch_start:.2f}s")

    return model
