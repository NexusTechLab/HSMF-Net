from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_training_curves(
    history: dict[str, list[float]], save_path: str | Path, title: str = "training"
) -> None:
    target = _ensure_parent(save_path)
    epochs = np.asarray(history.get("epoch", []), dtype=float)
    if epochs.size == 0:
        return
    train_loss = np.asarray(history.get("train_loss", []), dtype=float)
    val_loss = np.asarray(history.get("val_loss", []), dtype=float)
    train_acc = np.asarray(history.get("train_acc", []), dtype=float)
    val_acc = np.asarray(history.get("val_acc", []), dtype=float)

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_acc, label="train_acc")
    ax2.plot(epochs, val_acc, label="val_acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.suptitle(f"{title} curves")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    annotate: bool | None = None,
    annotate_fmt: str = ".2f",
) -> None:
    target = _ensure_parent(save_path)
    mat = matrix.astype(float)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mat = mat / row_sum

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_step = max(1, len(class_names) // 20)
    ticks = np.arange(0, len(class_names), tick_step)
    tick_labels = [class_names[i] for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=90, fontsize=7)
    plt.yticks(ticks, tick_labels, fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if annotate is None:
        annotate = len(class_names) <= 25
    if annotate:
        threshold = mat.max() * 0.6 if mat.size > 0 else 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text = format(value, annotate_fmt)
                color = "white" if value > threshold else "black"
                plt.text(j, i, text, ha="center", va="center", color=color, fontsize=6)

    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def save_confusion_matrix_csv(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> Path:
    target = _ensure_parent(save_path)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for i, row in enumerate(matrix):
            writer.writerow([class_names[i], *row.tolist()])
    return target


def plot_class_accuracy_bar(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    topk: int = 20,
) -> Path:
    target = _ensure_parent(save_path)
    diag = np.diag(matrix).astype(float)
    total = matrix.sum(axis=1).astype(float)
    total[total == 0] = 1.0
    class_acc = diag / total
    topk = max(1, min(topk, len(class_names)))
    indices = np.argsort(class_acc)[:topk]
    labels = [class_names[i] for i in indices]
    values = class_acc[indices]

    plt.figure(figsize=(12, 6))
    plt.bar(range(topk), values, color="tab:orange")
    plt.xticks(range(topk), labels, rotation=70, fontsize=8)
    plt.ylabel("Per-class accuracy")
    plt.title(f"Lowest {topk} class accuracies")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()
    return target


def plot_confusion_matrix_topk(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    topk: int = 20,
) -> Path:
    diag = np.diag(matrix)
    row_sum = matrix.sum(axis=1)
    error = row_sum - diag
    topk = max(1, min(topk, len(class_names)))
    indices = np.argsort(error)[-topk:]
    sub_mat = matrix[np.ix_(indices, indices)]
    sub_names = [class_names[i] for i in indices]
    plot_confusion_matrix(
        sub_mat,
        sub_names,
        save_path,
        normalize=True,
        title=f"Top-{topk} confusing classes",
        annotate=True,
        annotate_fmt=".2f",
    )
    return Path(save_path)


def save_top_confusion_pairs_csv(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    topn: int = 100,
) -> Path:
    target = _ensure_parent(save_path)
    pairs: list[tuple[int, int, int]] = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count > 0:
                pairs.append((i, j, count))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[: max(1, topn)]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "pred_class", "count"])
        for i, j, count in pairs:
            writer.writerow([class_names[i], class_names[j], count])
    return target


def save_per_class_prf_csv(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> Path:
    target = _ensure_parent(save_path)
    tp = np.diag(matrix).astype(float)
    pred_pos = matrix.sum(axis=0).astype(float)
    true_pos = matrix.sum(axis=1).astype(float)
    precision = tp / np.maximum(pred_pos, 1.0)
    recall = tp / np.maximum(true_pos, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = true_pos
    total_support = max(1.0, float(support.sum()))
    macro_precision = float(np.mean(precision)) if len(precision) > 0 else 0.0
    macro_recall = float(np.mean(recall)) if len(recall) > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0
    weighted_precision = float(np.sum(precision * support) / total_support)
    weighted_recall = float(np.sum(recall * support) / total_support)
    weighted_f1 = float(np.sum(f1 * support) / total_support)

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1", "support"])
        for idx, name in enumerate(class_names):
            writer.writerow(
                [
                    name,
                    f"{precision[idx]:.6f}",
                    f"{recall[idx]:.6f}",
                    f"{f1[idx]:.6f}",
                    int(support[idx]),
                ]
            )
        writer.writerow(["macro_avg", f"{macro_precision:.6f}", f"{macro_recall:.6f}", f"{macro_f1:.6f}", int(total_support)])
        writer.writerow(
            [
                "weighted_avg",
                f"{weighted_precision:.6f}",
                f"{weighted_recall:.6f}",
                f"{weighted_f1:.6f}",
                int(total_support),
            ]
        )
    return target


def plot_feature_space_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Feature Space t-SNE",
) -> Path:
    target = _ensure_parent(save_path)
    if features.shape[0] == 0:
        return target
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(features)
    plt.figure(figsize=(9, 7))
    unique_labels = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab10", max(1, len(unique_labels)))
    for idx, cls_id in enumerate(unique_labels):
        mask = labels == cls_id
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(cls_id)
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=10,
            alpha=0.65,
            label=name,
            color=cmap(idx),
        )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(target, dpi=220)
    plt.close()
    return target
