from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from mobilenet_enterprise.visualization.metrics_viz import (
    plot_class_accuracy_bar,
    plot_confusion_matrix,
    plot_confusion_matrix_topk,
    plot_feature_space_tsne,
    save_confusion_matrix_csv,
    save_per_class_prf_csv,
    save_top_confusion_pairs_csv,
)


@dataclass
class Evaluator:
    model: nn.Module
    device: torch.device

    def _move_to_device(self, value: Any):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        return value

    def _forward_model(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        if isinstance(inputs, list):
            return self.model(*inputs)
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    def _feature_tensor(self, inputs: Any) -> torch.Tensor:
        feats = self.model.forward_features(inputs)
        if feats.ndim > 2:
            feats = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)
        else:
            feats = torch.flatten(feats, 1)
        return feats

    def evaluate(
        self,
        loader,
        class_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        tsne_max_samples: int = 2000,
    ) -> dict[str, Any]:
        self.model.eval()
        total = 0
        correct = 0
        confusion = None
        feat_buf: list[np.ndarray] = []
        label_buf: list[np.ndarray] = []
        prediction_rows: list[dict[str, Any]] = []
        sample_offset = 0
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="eval", leave=False):
                images = self._move_to_device(images)
                targets = targets.to(self.device, non_blocking=True)
                if hasattr(self.model, "forward_features"):
                    feats = self._feature_tensor(images)
                    feat_buf.append(feats.detach().cpu().numpy())
                    label_buf.append(targets.detach().cpu().numpy())
                logits = self._forward_model(images)
                preds = logits.argmax(dim=1)
                if confusion is None:
                    num_classes = logits.size(1)
                    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                for t, p in zip(targets.view(-1), preds.view(-1), strict=False):
                    confusion[int(t.item()), int(p.item())] += 1
                probs = torch.softmax(logits, dim=1)
                logits_cpu = logits.detach().cpu()
                probs_cpu = probs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                preds_cpu = preds.detach().cpu()
                for local_idx in range(targets_cpu.numel()):
                    row: dict[str, Any] = {
                        "sample_index": sample_offset + local_idx,
                        "true_label": int(targets_cpu[local_idx].item()),
                        "true_class": "",
                        "pred_label": int(preds_cpu[local_idx].item()),
                        "pred_class": "",
                        "correct": int(preds_cpu[local_idx].item() == targets_cpu[local_idx].item()),
                    }
                    if class_names is not None:
                        true_idx = int(targets_cpu[local_idx].item())
                        pred_idx = int(preds_cpu[local_idx].item())
                        if 0 <= true_idx < len(class_names):
                            row["true_class"] = class_names[true_idx]
                        if 0 <= pred_idx < len(class_names):
                            row["pred_class"] = class_names[pred_idx]
                    for class_idx in range(logits_cpu.shape[1]):
                        row[f"logit_{class_idx}"] = float(logits_cpu[local_idx, class_idx].item())
                        row[f"prob_{class_idx}"] = float(probs_cpu[local_idx, class_idx].item())
                    prediction_rows.append(row)
                sample_offset += targets.size(0)
                total += targets.size(0)
                correct += (preds == targets).sum().item()
        acc = correct / max(1, total)
        result: dict[str, Any] = {"acc": acc}
        if confusion is not None and output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            names = (
                class_names
                if class_names is not None
                else [str(i) for i in range(confusion.shape[0])]
            )
            cm_path = out_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                confusion, names, cm_path, normalize=True, title="Normalized Confusion Matrix"
            )
            cm_csv_path = out_dir / "confusion_matrix_raw.csv"
            cm_top20_path = out_dir / "confusion_matrix_top20.png"
            cls_acc_bar_path = out_dir / "class_accuracy_bar.png"
            top_pairs_path = out_dir / "top_confusion_pairs.csv"
            prf_csv_path = out_dir / "per_class_prf.csv"
            predictions_path = out_dir / "eval_predictions.csv"
            save_confusion_matrix_csv(confusion, names, cm_csv_path)
            plot_confusion_matrix_topk(confusion, names, cm_top20_path, topk=20)
            plot_class_accuracy_bar(confusion, names, cls_acc_bar_path, topk=20)
            save_top_confusion_pairs_csv(confusion, names, top_pairs_path, topn=100)
            save_per_class_prf_csv(confusion, names, prf_csv_path)
            if prediction_rows:
                fieldnames = list(prediction_rows[0].keys())
                with predictions_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(prediction_rows)
            if feat_buf and label_buf:
                feats_np = np.concatenate(feat_buf, axis=0)
                labels_np = np.concatenate(label_buf, axis=0)
                feature_npz_path = out_dir / "feature_space_raw.npz"
                np.savez_compressed(
                    feature_npz_path,
                    features=feats_np,
                    labels=labels_np,
                    class_names=np.asarray(names),
                )
                max_samples = max(100, tsne_max_samples)
                if feats_np.shape[0] > max_samples:
                    rng = np.random.default_rng(seed=42)
                    indices = rng.choice(feats_np.shape[0], size=max_samples, replace=False)
                    feats_np = feats_np[indices]
                    labels_np = labels_np[indices]
                tsne_path = out_dir / "feature_space_tsne.png"
                plot_feature_space_tsne(
                    feats_np,
                    labels_np,
                    names,
                    tsne_path,
                    title="Feature Space t-SNE",
                )
                result["feature_space_tsne_path"] = str(tsne_path)
                result["feature_space_raw_npz"] = str(feature_npz_path)
            result["confusion_matrix_path"] = str(cm_path)
            result["confusion_matrix_raw_csv"] = str(cm_csv_path)
            result["confusion_matrix_top20_path"] = str(cm_top20_path)
            result["class_accuracy_bar_path"] = str(cls_acc_bar_path)
            result["top_confusion_pairs_csv"] = str(top_pairs_path)
            result["per_class_prf_csv"] = str(prf_csv_path)
            result["eval_predictions_csv"] = str(predictions_path)
        return result
