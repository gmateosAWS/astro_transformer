import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

from src.fase2.script_2_transformer_fine_tuning_optimizado import AstroConformerClassifier


def evaluate_model_test(
    test_loader: DataLoader,
    model,
    label_encoder,
    model_name_in: str,
    output_dir: str = "../outputs/evaluacion_test"
):
    # Crear directorio de salida si no existe
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraer solo el nombre base del modelo (sin path ni extensión)
    model_name_base = Path(model_name_in).stem

    model.eval()

    y_true, y_pred = [], []
    all_ids = []

    # Predicción sobre datos de test
    with torch.no_grad():
        for batch in test_loader:
            # Soporta datasets con o sin features adicionales

            if len(batch) == 5:
                x_batch, y_batch, mask_batch, features, ids = batch
            elif len(batch) == 4:
                x_batch, y_batch, mask_batch, features = batch
                ids = None
            elif len(batch) == 3:
                x_batch, y_batch, mask_batch = batch
                features = None
                ids = None
            elif len(batch) == 2:
                x_batch, y_batch = batch
                mask_batch = None
                features = None
                ids = None
            else:
                raise ValueError(f"Formato de batch no soportado: longitud {len(batch)}")

            x_batch = x_batch.to(next(model.parameters()).device)
            mask_batch = mask_batch.to(next(model.parameters()).device)
            if features is not None:
                features = features.to(next(model.parameters()).device)
                logits = model(x_batch, mask_batch, features)
            else:
                logits = model(x_batch, mask_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)
            all_ids.extend(ids if ids is not None else [""] * y_batch.shape[0])


    # Obtener nombres de clase
    if hasattr(label_encoder, "classes_"):
        class_names = label_encoder.classes_
    elif isinstance(label_encoder, dict):
        # Si es un dict {clase: idx} o {idx: clase}
        if all(isinstance(k, int) for k in label_encoder.keys()):
            # idx -> clase
            class_names = [label_encoder[i] for i in range(len(label_encoder))]
        else:
            # clase -> idx
            inv = {v: k for k, v in label_encoder.items()}
            class_names = [inv[i] for i in range(len(inv))]
    else:
        raise ValueError("label_encoder no reconocido")

    # Classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    report_csv_path = output_dir / f"{model_name_base}_test_class_report.csv"
    report_df.to_csv(report_csv_path, index=True)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    conf_matrix_csv_path = output_dir / f"{model_name_base}_test_confusion_matrix.csv"
    pd.DataFrame(conf_matrix, index=class_names, columns=class_names).to_csv(conf_matrix_csv_path)

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(conf_matrix, display_labels=class_names).plot(cmap="Blues", xticks_rotation="vertical")
    plt.title(f"Matriz de Confusión ({model_name_base})")
    plt.savefig(output_dir / f"{model_name_base}_test_confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Reconstruir la lista de clases a partir del dict
    if isinstance(label_encoder, dict):
        idx_to_class = {v: k for k, v in label_encoder.items()}
        class_list = np.array([idx_to_class[i] for i in range(len(idx_to_class))])
    else:
        # Si fuera un LabelEncoder real
        class_list = label_encoder.classes_
    
    # Construir DataFrame con predicciones
    df = pd.DataFrame({
        "id": all_ids,
        "y_true": class_list[y_true],
        "y_pred": class_list[y_pred],
        "y_true_encoded": y_true,
        "y_pred_encoded": y_pred
    })

    pred_csv_path = output_dir / f"{model_name_base}_test_predictions.csv"
    df.to_csv(pred_csv_path, index=False)

    print("Evaluación completada. Resultados guardados en:", output_dir)
    return report_df


def main(
    test_loader,
    label_encoder,
    model_name_in,
    device="cuda",
    output_dir="../outputs/evaluacion_test",
):
    # --- Definir num_classes y argumentos del modelo igual que en entrenamiento ---
    num_classes = len(label_encoder)
    args = argparse.Namespace(
        input_dim=1,
        in_channels=1,
        encoder_dim=256,
        hidden_dim=384,
        output_dim=num_classes,
        num_heads=8,
        num_layers=8,
        dropout=0.3, dropout_p=0.3,
        stride=32,
        kernel_size=3,
        norm="postnorm",
        encoder=["mhsa_pro", "conv", "conv"],
        timeshift=False,
        device=device
    )
    model = AstroConformerClassifier(args, num_classes, feature_dim=7).to(device)

    state_dict = torch.load(model_name_in, map_location=device)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("⚠️ Detected _orig_mod. prefix in state_dict. Stripping prefixes...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    return evaluate_model_test(
        test_loader=test_loader,
        model=model,
        label_encoder=label_encoder,
        model_name_in=model_name_in,
        output_dir=output_dir
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Carga de dataset
    test_dataset = torch.load("../data/processed/test_dataset.pt")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Cargar label_encoder desde archivo
    import pickle
    with open("../data/processed/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Importa tu clase y argumentos del modelo aquí
    # from ... import AstroConformerClassifier
    # args = ( ... )  # argumentos necesarios para instanciar el modelo

    # main(
    #     test_loader=test_loader,
    #     label_encoder=label_encoder,
    #     model_class=AstroConformerClassifier,
    #     model_args=args,
    #     model_path="../outputs/mejor_modelo_optimizado_YSO_curated_fine_tuned.pt",
    #     model_name_in="mejor_modelo_optimizado_YSO_curated_fine_tuned.pt",
    #     output_dir="../outputs/evaluacion_test",
    #     device=device
    # )
    # )
