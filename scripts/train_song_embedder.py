#!/usr/bin/env python3
"""
Train a supervised song embedding model from SAPPHIRE feature tables.

This script is intentionally standalone and does not depend on the internal
pipeline modules. It only uses standard ML libraries (pandas, numpy,
scikit-learn, PyTorch, matplotlib, seaborn).

Example usage:

  python scripts/train_song_embedder.py \
      --features features/mirex/features_flat_clean_clean.parquet \
      --output models/embedder_mirex \
      --embedding-dim 64 \
      --epochs 40

The script will:
  - Load features and mood labels
  - Train a supervised embedding model (MLP + embedding layer)
  - Evaluate classification performance
  - Extract embeddings for all tracks
  - Create PCA/t-SNE plots of the embedding space
  - Save a detailed JSON/Markdown report
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


class SongDataset(Dataset):
    """Simple torch Dataset for song features and mood labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SongEmbedder(nn.Module):
    """MLP-based song embedding model with a classification head."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_classes: int,
        hidden_dims=(256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.embedding_layer = nn.Linear(prev_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_extractor(x)
        emb = self.embedding_layer(h)
        # L2-normalize embeddings for stability / interpretability
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        logits = self.classifier(emb)
        return emb, logits


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare numeric feature matrix X and encoded labels y from a feature table.

    The function mirrors the logic of the main pipeline but is self-contained.
    """
    if "mood_cluster" not in df.columns:
        raise ValueError("Input features must contain a 'mood_cluster' column for supervised training.")

    # Drop rows without labels
    df = df.dropna(subset=["mood_cluster"]).copy()

    # Separate metadata and features
    metadata_cols = ["track_id", "mood_cluster", "mood_category", "dataset", "language"]
    meta = {col: df[col].values for col in metadata_cols if col in df.columns}

    # Choose feature columns (exclude metadata/label)
    exclude_cols = set(metadata_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    X = X[numeric_cols]

    # Replace inf and NaNs with column means
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    # Targets
    y_raw = df["mood_cluster"].values

    info = {
        "numeric_feature_cols": numeric_cols,
        "metadata": meta,
    }

    return X.values, y_raw, info


def train_embedder(
    X: np.ndarray,
    y_raw: np.ndarray,
    embedding_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the embedding model and return artifacts and metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Datasets and loaders
    train_ds = SongDataset(X_train_scaled, y_train)
    val_ds = SongDataset(X_val_scaled, y_val)
    test_ds = SongDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SongEmbedder(
        input_dim=X.shape[1],
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                _, logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs} - "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "scaler": scaler,
                "label_encoder_classes": label_encoder.classes_.tolist(),
            }

    # Load best state for evaluation
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Test evaluation
    model.eval()
    all_logits = []
    all_targets = []
    all_embeddings = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            emb, logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            all_embeddings.append(emb.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    y_pred = all_logits.argmax(axis=1)

    test_acc = accuracy_score(all_targets, y_pred)
    cls_report = classification_report(
        all_targets,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    cm = confusion_matrix(all_targets, y_pred)

    # Compute additional embedding quality metrics and downstream probe tasks
    # Train embeddings for downstream probes
    train_embeddings_list: List[np.ndarray] = []
    train_targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            emb_train, _ = model(xb)
            train_embeddings_list.append(emb_train.cpu().numpy())
            train_targets_list.append(yb.cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings_list, axis=0)
    train_targets = np.concatenate(train_targets_list, axis=0)

    embedding_quality = evaluate_embedding_quality(all_embeddings, all_targets)
    downstream_results = evaluate_downstream_probes(
        train_embeddings=train_embeddings,
        train_labels=train_targets,
        test_embeddings=all_embeddings,
        test_labels=all_targets,
    )

    feature_reconstruction = evaluate_feature_reconstruction(
        train_embeddings=train_embeddings,
        train_features=X_train_scaled,
        test_embeddings=all_embeddings,
        test_features=X_test_scaled,
    )

    # Save model state
    model_path = output_dir / "song_embedder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "label_encoder_classes": label_encoder.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        model_path,
    )

    artifacts = {
        "model": model,
        "device": device,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "history": history,
        "test_acc": float(test_acc),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "embeddings_test": all_embeddings,
        "y_test": all_targets,
        "y_test_pred": y_pred,
        "y_test_proba": all_probs,
        "embedding_quality": embedding_quality,
        "downstream_probes": downstream_results,
        "feature_reconstruction": feature_reconstruction,
        "model_path": str(model_path),
    }

    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "num_classes": num_classes,
        "num_features": int(X.shape[1]),
        "embedding_dim": embedding_dim,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(cls_report, f, indent=2)

    # Save additional evaluation summaries
    with open(output_dir / "embedding_quality.json", "w") as f:
        json.dump(embedding_quality, f, indent=2)

    with open(output_dir / "downstream_probes.json", "w") as f:
        json.dump(downstream_results, f, indent=2)

    with open(output_dir / "feature_reconstruction.json", "w") as f:
        json.dump(feature_reconstruction, f, indent=2)

    return artifacts


def evaluate_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute basic embedding quality metrics on the test set."""

    results: Dict[str, Any] = {}

    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        try:
            sil = silhouette_score(embeddings, labels, metric="euclidean")
        except Exception:
            sil = None
        results["silhouette_score"] = sil
    else:
        results["silhouette_score"] = None

    # Per-class centroids and intra/inter-class distances
    centroids = {}
    intra_dists = []
    for cls in unique_labels:
        mask = labels == cls
        if not np.any(mask):
            continue
        emb_cls = embeddings[mask]
        centroid = emb_cls.mean(axis=0)
        centroids[int(cls)] = centroid.tolist()
        dists = np.linalg.norm(emb_cls - centroid, axis=1)
        intra_dists.extend(dists.tolist())

    results["mean_intra_class_distance"] = float(np.mean(intra_dists)) if intra_dists else None

    centroid_list = list(centroids.values())
    if len(centroid_list) > 1:
        centroid_arr = np.stack(centroid_list, axis=0)
        # pairwise distances between centroids (upper triangle)
        diff = centroid_arr[:, None, :] - centroid_arr[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices_from(dist_mat, k=1)
        inter_dists = dist_mat[iu]
        results["mean_inter_class_centroid_distance"] = float(np.mean(inter_dists))
    else:
        results["mean_inter_class_centroid_distance"] = None

    return results


def evaluate_downstream_probes(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Any]:
    """Train simple downstream classifiers on frozen embeddings.

    This approximates how useful the embedding is for mood classification itself
    when using lightweight models.
    """

    results: Dict[str, Any] = {}

    # Logistic regression probe
    try:
        lr = LogisticRegression(max_iter=1000, multi_class="auto")
        lr.fit(train_embeddings, train_labels)
        y_pred_lr = lr.predict(test_embeddings)
        results["logreg_accuracy"] = float(accuracy_score(test_labels, y_pred_lr))
    except Exception:
        results["logreg_accuracy"] = None

    # k-NN probe
    try:
        knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
        knn.fit(train_embeddings, train_labels)
        y_pred_knn = knn.predict(test_embeddings)
        results["knn_accuracy"] = float(accuracy_score(test_labels, y_pred_knn))
    except Exception:
        results["knn_accuracy"] = None

    return results


def evaluate_feature_reconstruction(
    train_embeddings: np.ndarray,
    train_features: np.ndarray,
    test_embeddings: np.ndarray,
    test_features: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate how well original features can be reconstructed from embeddings."""

    results: Dict[str, Any] = {}

    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_embeddings, train_features)
        recon = ridge.predict(test_embeddings)
        mse = float(np.mean((recon - test_features) ** 2))
        r2 = float(ridge.score(test_embeddings, test_features))
        results["mse"] = mse
        results["r2"] = r2
    except Exception:
        results["mse"] = None
        results["r2"] = None

    return results


def plot_training_curves(history: Dict[str, Any], output_dir: Path) -> None:
    """Plot training/validation loss and accuracy curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmax(history["val_acc"])) + 1 if history["val_acc"] else 1

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Val (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names, output_dir: Path) -> None:
    """Plot confusion matrix heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Song Embedder Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names,
    output_dir: Path,
) -> None:
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    classes = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(10, 8))
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"micro-average (AUC={roc_auc_micro:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curves_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8))
    for i, cls_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f"{cls_name} (AP={ap:.3f})")

    precision_micro, recall_micro, _ = precision_recall_curve(
        y_bin.ravel(), y_proba.ravel()
    )
    ap_micro = average_precision_score(y_bin, y_proba, average="micro")
    plt.plot(
        recall_micro,
        precision_micro,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"micro-average (AP={ap_micro:.3f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curves_dir / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    classification_report: Dict[str, Any],
    output_dir: Path,
) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_labels = []
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls, stats in classification_report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        class_labels.append(str(cls))
        precisions.append(stats.get("precision", 0.0))
        recalls.append(stats.get("recall", 0.0))
        f1s.append(stats.get("f1-score", 0.0))
        supports.append(stats.get("support", 0))

    if not class_labels:
        return

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    order = np.argsort(f1s)[::-1]
    class_labels = [class_labels[i] for i in order]
    precisions = precisions[order]
    recalls = recalls[order]
    f1s = f1s[order]
    supports = supports[order]

    x = np.arange(len(class_labels))
    width = 0.25

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1-score")
    plt.xticks(x, class_labels, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.title("Per-Class Metrics (Sorted by F1)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x, supports, color="gray")
    plt.xticks(x, class_labels, rotation=45, ha="right")
    plt.ylabel("Support")
    plt.title("Per-Class Support (Test Set)")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(metrics_dir / "per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_embedding_correctness(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
    output_dir: Path,
    random_state: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=random_state)
    emb_pca = pca.fit_transform(embeddings)

    correct = y_true == y_pred
    incorrect = ~correct

    plt.figure(figsize=(10, 8))
    plt.scatter(
        emb_pca[correct, 0],
        emb_pca[correct, 1],
        c="tab:blue",
        alpha=0.5,
        s=20,
        label="Correct",
    )
    if np.any(incorrect):
        plt.scatter(
            emb_pca[incorrect, 0],
            emb_pca[incorrect, 1],
            c="tab:red",
            alpha=0.8,
            s=40,
            label="Incorrect",
        )

    acc = float(correct.sum()) / max(len(correct), 1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Embedding Space (Correct vs Incorrect)\nTest Accuracy={acc:.3f}")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_correctness.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names,
    output_dir: Path,
    random_state: int = 42,
) -> None:
    """Create PCA and t-SNE visualizations of the embedding space."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    emb_pca = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if not np.any(mask):
            continue
        plt.scatter(emb_pca[mask, 0], emb_pca[mask, 1], label=cls_name, alpha=0.7, s=30)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Embedding Space PCA (Test Set)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_pca.png", dpi=300, bbox_inches="tight")
    plt.close()

    # t-SNE (may be slower; limit samples if very large)
    max_samples = 1000
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        emb_tsne_input = embeddings[idx]
        labels_tsne = labels[idx]
    else:
        emb_tsne_input = embeddings
        labels_tsne = labels

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(emb_tsne_input) - 1))
    emb_tsne = tsne.fit_transform(emb_tsne_input)

    plt.figure(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels_tsne == cls_idx
        if not np.any(mask):
            continue
        plt.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1], label=cls_name, alpha=0.7, s=30)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Embedding Space t-SNE (Test Set)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_tsne.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_full_embeddings(
    model: SongEmbedder,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    X: np.ndarray,
    y_raw: np.ndarray,
    meta: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> None:
    """Compute and save embeddings for the full dataset (not just test set)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_scaled = scaler.transform(X)

    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X_scaled.astype(np.float32)).to(device)
        emb, _ = model(xb)
        emb_np = emb.cpu().numpy()

    # Build DataFrame with metadata + embedding columns
    df_dict = {}
    for key, values in meta.items():
        df_dict[key] = values

    df_dict["mood_cluster"] = y_raw
    for i in range(emb_np.shape[1]):
        df_dict[f"emb_{i}"] = emb_np[:, i]

    emb_df = pd.DataFrame(df_dict)

    emb_df.to_parquet(output_dir / "song_embeddings.parquet", index=False)
    emb_df.to_csv(output_dir / "song_embeddings.csv", index=False)


def save_embedder_report(
    metrics: Dict[str, Any],
    classification_report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create a simple Markdown report summarizing the embedder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_lines = []
    md_lines.append("# Song Embedder Report\n")
    md_lines.append("## Overview\n")
    md_lines.append(f"- **Test Accuracy**: {metrics['test_accuracy']:.4f}\n")
    md_lines.append(f"- **Best Validation Accuracy**: {metrics['best_val_accuracy']:.4f}\n")
    md_lines.append(f"- **Embedding Dimension**: {metrics['embedding_dim']}\n")
    md_lines.append(f"- **Number of Classes**: {metrics['num_classes']}\n")
    md_lines.append(f"- **Number of Features**: {metrics['num_features']}\n")

    md_lines.append("\n## Per-Class Metrics (Test Set)\n")
    md_lines.append("| Class | Precision | Recall | F1-score | Support |\n")
    md_lines.append("|-------|-----------|--------|----------|---------|\n")
    for cls, stats in classification_report.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        md_lines.append(
            f"| {cls} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
            f"{stats['f1-score']:.3f} | {stats['support']} |\n"
        )

    if "accuracy" in classification_report:
        md_lines.append("\n### Global Averages\n")
        acc = classification_report["accuracy"]
        macro = classification_report.get("macro avg", {})
        weighted = classification_report.get("weighted avg", {})
        md_lines.append(f"- **Accuracy**: {acc:.4f}\n")
        if macro:
            md_lines.append(
                f"- **Macro F1**: {macro.get('f1-score', 0.0):.4f} (precision={macro.get('precision', 0.0):.4f}, "
                f"recall={macro.get('recall', 0.0):.4f})\n"
            )
        if weighted:
            md_lines.append(
                f"- **Weighted F1**: {weighted.get('f1-score', 0.0):.4f} "
                f"(precision={weighted.get('precision', 0.0):.4f}, recall={weighted.get('recall', 0.0):.4f})\n"
            )

    report_path = output_dir / "EMBEDDER_REPORT.md"
    with open(report_path, "w") as f:
        f.write("".join(md_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised song embedding model from SAPPHIRE feature tables.",
    )

    parser.add_argument("--features", type=str, required=True, help="Path to features file (.parquet or .csv)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for model and reports")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_path = Path(args.features)
    output_dir = Path(args.output)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    print(f"📂 Loading features from: {features_path}")

    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    print(f"📊 Loaded {len(df)} rows and {len(df.columns)} columns")

    X, y_raw, info = prepare_features(df)
    print(f"🧮 Using {X.shape[1]} numeric features for {X.shape[0]} labeled tracks")

    artifacts = train_embedder(
        X=X,
        y_raw=y_raw,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
        random_state=args.random_state,
    )

    print(f"✅ Training complete. Test accuracy: {artifacts['test_acc']:.4f}")

    plots_dir = output_dir / "plots"
    plot_training_curves(artifacts["history"], plots_dir)

    cm = np.array(artifacts["confusion_matrix"], dtype=np.int64)
    class_names = artifacts["label_encoder"].classes_
    plot_confusion_matrix(cm, class_names, plots_dir)

    plot_roc_pr_curves(
        y_true=artifacts["y_test"],
        y_proba=artifacts["y_test_proba"],
        class_names=class_names,
        output_dir=plots_dir,
    )

    plot_per_class_metrics(
        classification_report=artifacts["classification_report"],
        output_dir=plots_dir,
    )

    emb_dir = output_dir / "embeddings"
    plot_embedding_space(
        embeddings=artifacts["embeddings_test"],
        labels=artifacts["y_test"],
        class_names=class_names,
        output_dir=emb_dir,
    )

    plot_embedding_correctness(
        embeddings=artifacts["embeddings_test"],
        y_true=artifacts["y_test"],
        y_pred=artifacts["y_test_pred"],
        class_names=class_names,
        output_dir=emb_dir,
    )

    # Save full-dataset embeddings
    save_full_embeddings(
        model=artifacts["model"],
        scaler=artifacts["scaler"],
        label_encoder=artifacts["label_encoder"],
        X=X,
        y_raw=y_raw,
        meta=info["metadata"],
        output_dir=emb_dir,
        device=artifacts["device"],
    )

    # Save Markdown report
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {
            "test_accuracy": artifacts["test_acc"],
            "best_val_accuracy": max(artifacts["history"]["val_acc"]),
            "embedding_dim": args.embedding_dim,
            "num_classes": len(class_names),
            "num_features": int(X.shape[1]),
        }

    save_embedder_report(metrics, artifacts["classification_report"], output_dir)

    print(f"📁 All embedder artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
