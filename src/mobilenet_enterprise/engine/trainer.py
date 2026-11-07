from __future__ import annotations

import csv
from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mobilenet_enterprise.utils.runtime import dump_json
from mobilenet_enterprise.visualization.metrics_viz import plot_training_curves


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _gpu_mem_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(device) / (1024**3)


@dataclass
class Trainer:
    model: nn.Module
    device: torch.device
    cfg: dict[str, Any]
    output_dir: Path
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        opt_cfg = self.cfg.get("optimizer", {})
        sch_cfg = self.cfg.get("scheduler", {})
        train_cfg = self.cfg.get("train", {})
        lr = _as_float(opt_cfg.get("lr", 1e-3), 1e-3)
        weight_decay = _as_float(opt_cfg.get("weight_decay", 1e-4), 1e-4)
        train_epochs = _as_int(train_cfg.get("epochs", 50), 50)
        t_max_raw = sch_cfg.get("t_max", None)
        t_max = _as_int(t_max_raw, train_epochs)
        # Prevent cosine schedule from finishing too early when epochs are larger.
        if t_max < train_epochs:
            t_max = train_epochs
        eta_min = _as_float(sch_cfg.get("eta_min", 1e-6), 1e-6)
        self.label_smoothing = _as_float(train_cfg.get("label_smoothing", 0.1), 0.1)
        self.loss_name = str(train_cfg.get("loss_name", "cross_entropy")).lower()
        self.focal_gamma = _as_float(train_cfg.get("focal_gamma", 2.0), 2.0)
        self.class_weighting_enabled = bool(train_cfg.get("class_weighting_enabled", False))
        self.class_weight_power = _as_float(train_cfg.get("class_weight_power", 1.0), 1.0)
        self.class_weight_target_boost = {
            str(k): _as_float(v, 1.0) for k, v in train_cfg.get("class_weight_target_boost", {}).items()
        }
        self.warmup_epochs = _as_int(train_cfg.get("warmup_epochs", 5), 5)
        self.ema_enabled = bool(train_cfg.get("ema_enabled", True))
        self.ema_decay = _as_float(train_cfg.get("ema_decay", 0.999), 0.999)
        self.ema_tau = _as_float(train_cfg.get("ema_tau", 2000.0), 2000.0)
        self.ema_eval_start_epoch = _as_int(
            train_cfg.get("ema_eval_start_epoch", max(15, self.warmup_epochs + 1)),
            max(15, self.warmup_epochs + 1),
        )
        self.ema_eval_min_updates = _as_int(train_cfg.get("ema_eval_min_updates", 1500), 1500)
        self.ema_eval_after_warmup_epochs = _as_int(
            train_cfg.get("ema_eval_after_warmup_epochs", 3),
            3,
        )
        self.ema_switch_max_drop = _as_float(train_cfg.get("ema_switch_max_drop", 0.01), 0.01)
        self.early_stopping_min_epoch = _as_int(train_cfg.get("early_stopping_min_epoch", 0), 0)
        self.mixup_alpha = _as_float(train_cfg.get("mixup_alpha", 0.0), 0.0)
        self.mixup_prob = _as_float(train_cfg.get("mixup_prob", 1.0), 1.0)
        self.grad_clip_norm = _as_float(train_cfg.get("grad_clip_norm", 0.0), 0.0)
        self.abort_on_non_finite = bool(train_cfg.get("abort_on_non_finite", True))

        self.class_weights: torch.Tensor | None = None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, t_max - self.warmup_epochs),
            eta_min=eta_min,
        )
        self.optimizer_name = str(opt_cfg.get("name", "adamw")).lower()
        self.scheduler_name = str(sch_cfg.get("name", "cosine")).lower()
        if self.optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer name: {self.optimizer_name}")
        if self.scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler name: {self.scheduler_name}")
        amp_enabled = bool(train_cfg.get("amp", True))
        self.scaler = torch.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.log_interval = _as_int(self.cfg.get("logging", {}).get("log_interval", 50), 50)
        self.tensorboard_enabled = bool(self.cfg.get("logging", {}).get("tensorboard", True))
        self.show_progress_bar = bool(self.cfg.get("logging", {}).get("show_progress_bar", True))
        self.show_batch_log = bool(self.cfg.get("logging", {}).get("show_batch_log", False))
        self.writer = (
            SummaryWriter(log_dir=str(self.output_dir / "logs"))
            if self.tensorboard_enabled
            else None
        )
        self.start_epoch = 1
        self.best_acc = -1.0
        self.no_improve_epochs = 0
        self.ema_updates = 0
        self.ema_eval_active = False
        self.ema_model: nn.Module | None = None
        if self.ema_enabled:
            self.ema_model = self._build_ema_model()
        self.smoke_monitor_cfg = train_cfg.get("smoke_monitor", {})
        self.smoke_monitor_enabled = bool(self.smoke_monitor_cfg.get("enabled", False))
        self.smoke_monitor_start_epoch = _as_int(self.smoke_monitor_cfg.get("start_epoch", 1), 1)
        self.smoke_majority_ratio_threshold = _as_float(
            self.smoke_monitor_cfg.get("majority_ratio_threshold", 0.85), 0.85
        )
        self.smoke_entropy_threshold = _as_float(
            self.smoke_monitor_cfg.get("entropy_threshold", 0.35), 0.35
        )
        self.smoke_majority_only_threshold = _as_float(
            self.smoke_monitor_cfg.get("majority_only_threshold", 0.98), 0.98
        )
        self.smoke_abort_on_collapse = bool(self.smoke_monitor_cfg.get("abort_on_collapse", False))

    def _build_ema_model(self) -> nn.Module:
        ema_model = copy.deepcopy(self.model).to(self.device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return ema_model

    def _update_ema(self) -> None:
        if not self.ema_enabled or self.ema_model is None:
            return
        self.ema_updates += 1
        decay = self._current_ema_decay()
        with torch.no_grad():
            model_state = self.model.state_dict()
            ema_state = self.ema_model.state_dict()
            for key, value in ema_state.items():
                src = model_state[key].detach()
                if value.dtype.is_floating_point:
                    value.copy_(value * decay + src * (1.0 - decay))
                else:
                    # For integer buffers (e.g. BatchNorm counters), use direct copy.
                    value.copy_(src)

    def _current_ema_decay(self) -> float:
        if not self.ema_enabled:
            return 0.0
        return self.ema_decay * (1.0 - math.exp(-self.ema_updates / max(1.0, self.ema_tau)))

    def _ema_gate_flags(self, epoch: int) -> dict[str, bool]:
        epoch_ready = epoch >= self.ema_eval_start_epoch
        updates_ready = self.ema_updates >= self.ema_eval_min_updates
        warmup_ready = epoch >= (self.warmup_epochs + self.ema_eval_after_warmup_epochs)
        eligible = (
            self.ema_enabled
            and self.ema_model is not None
            and epoch_ready
            and updates_ready
            and warmup_ready
        )
        return {
            "epoch_ready": epoch_ready,
            "updates_ready": updates_ready,
            "warmup_ready": warmup_ready,
            "eligible": eligible,
        }

    def _ema_gate_status_text(self, flags: dict[str, bool]) -> str:
        return (
            f"E{int(flags['epoch_ready'])}"
            f"/U{int(flags['updates_ready'])}"
            f"/W{int(flags['warmup_ready'])}"
            f"/A{int(self.ema_eval_active)}"
        )

    def _apply_warmup_lr(self, epoch: int) -> None:
        if self.warmup_epochs <= 0 or epoch > self.warmup_epochs:
            return
        scale = epoch / max(1, self.warmup_epochs)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=False):
            group["lr"] = base_lr * scale

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

    def _batch_size(self, value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.size(0))
        if isinstance(value, (tuple, list)) and value:
            return self._batch_size(value[0])
        if isinstance(value, dict) and value:
            return self._batch_size(next(iter(value.values())))
        raise TypeError("Unable to infer batch size for mixup inputs.")

    def _mixup_value(self, value: Any, index: torch.Tensor, lam: float):
        if torch.is_tensor(value):
            return lam * value + (1.0 - lam) * value[index]
        if isinstance(value, tuple):
            return tuple(self._mixup_value(item, index, lam) for item in value)
        if isinstance(value, list):
            return [self._mixup_value(item, index, lam) for item in value]
        if isinstance(value, dict):
            return {key: self._mixup_value(item, index, lam) for key, item in value.items()}
        raise TypeError(f"Unsupported mixup input type: {type(value)}")

    def _forward_model(self, model: nn.Module, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, tuple):
            return model(*inputs)
        if isinstance(inputs, list):
            return model(*inputs)
        if isinstance(inputs, dict):
            return model(**inputs)
        return model(inputs)

    def _apply_mixup(self, images: Any, targets: torch.Tensor):
        if self.mixup_alpha <= 0.0:
            return images, targets, targets, 1.0
        device = targets.device
        if self.mixup_prob < 1.0 and torch.rand(1, device=device).item() > self.mixup_prob:
            return images, targets, targets, 1.0
        beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = float(beta.sample().item())
        index = torch.randperm(self._batch_size(images), device=device)
        mixed_images = self._mixup_value(images, index, lam)
        return mixed_images, targets, targets[index], lam

    def _compute_class_weights(self, train_loader) -> torch.Tensor | None:
        if not self.class_weighting_enabled:
            return None
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None:
            return None
        labels: list[int] = []
        if hasattr(dataset, "samples"):
            labels = [int(sample[1]) for sample in getattr(dataset, "samples")]
        elif hasattr(dataset, "targets"):
            labels = [int(v) for v in list(getattr(dataset, "targets"))]
        elif hasattr(dataset, "tensors") and len(getattr(dataset, "tensors")) >= 2:
            labels = [int(v) for v in dataset.tensors[1].tolist()]
        if not labels:
            return None

        num_classes = int(max(labels)) + 1
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
        inv = (counts.clamp_min(1.0)).pow(-self.class_weight_power)
        if inv.sum() > 0:
            inv = inv * (float(num_classes) / inv.sum())

        if self.class_weight_target_boost:
            name_to_idx = {
                str(name): idx for idx, name in enumerate(self.class_names or [])
            }
            for key, boost in self.class_weight_target_boost.items():
                idx: int | None = None
                if key in name_to_idx:
                    idx = name_to_idx[key]
                else:
                    try:
                        idx = int(key)
                    except ValueError:
                        idx = None
                if idx is not None and 0 <= idx < len(inv):
                    inv[idx] *= max(1.0, boost)
        return inv.to(self.device)

    def _ce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.focal_gamma) * ce).mean()

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "focal":
            return self._focal_loss(logits, targets)
        return self._ce_loss(logits, targets)

    def _handle_non_finite(self, message: str) -> None:
        if self.abort_on_non_finite:
            raise FloatingPointError(message)
        print(f"[NON_FINITE_SKIP] {message}")

    def load_checkpoint(self, ckpt_path: str | Path) -> int:
        state = torch.load(ckpt_path, map_location=self.device)
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state, strict=False)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        if self.ema_enabled and self.ema_model is not None:
            if "ema_model" in state:
                self.ema_model.load_state_dict(state["ema_model"], strict=False)
            else:
                # Backward compatibility: old checkpoints may not contain ema_model.
                # In that case, align EMA weights with resumed model to avoid random-val collapse.
                self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
        self.best_acc = float(state.get("best_acc", 0.0))
        self.no_improve_epochs = int(state.get("no_improve_epochs", 0))
        self.ema_updates = int(state.get("ema_updates", 0))
        self.ema_eval_active = bool(state.get("ema_eval_active", False))
        self.start_epoch = int(state.get("epoch", 0)) + 1
        return self.start_epoch

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        """Load model/EMA weights for fine-tuning; reset optimizer schedule and epoch."""
        state = torch.load(ckpt_path, map_location=self.device)
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state, strict=False)
        if self.ema_enabled and self.ema_model is not None:
            if "ema_model" in state:
                self.ema_model.load_state_dict(state["ema_model"], strict=False)
            else:
                self.ema_model.load_state_dict(self.model.state_dict(), strict=False)
        self.best_acc = 0.0
        self.no_improve_epochs = 0
        self.ema_updates = int(state.get("ema_updates", 0))
        self.ema_eval_active = False
        self.start_epoch = 1

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
            "no_improve_epochs": self.no_improve_epochs,
            "ema_updates": self.ema_updates,
            "ema_eval_active": self.ema_eval_active,
        }
        if self.ema_enabled and self.ema_model is not None:
            payload["ema_model"] = self.ema_model.state_dict()
        torch.save(payload, path)

    def _run_epoch(
        self,
        loader,
        train: bool,
        epoch: int,
        epochs: int,
        model_override: nn.Module | None = None,
        stage_override: str | None = None,
    ) -> tuple[float, float, dict[str, float | int | str]]:
        current_model = model_override if model_override is not None else self.model
        current_model.train(mode=train)
        total_loss = 0.0
        total_acc = 0.0
        monitor_counts: torch.Tensor | None = None
        entropy_sum = 0.0
        sample_count = 0
        stage = stage_override if stage_override is not None else ("train" if train else "val")
        pbar = tqdm(
            loader,
            desc=f"{stage:>5} {epoch:>3}/{epochs}",
            leave=False,
            total=len(loader),
            ncols=130,
            disable=not self.show_progress_bar,
        )
        for step, (images, targets) in enumerate(pbar, start=1):
            images = self._move_to_device(images)
            targets = targets.to(self.device, non_blocking=True)
            targets_a = targets
            targets_b = targets
            lam = 1.0
            if train:
                images, targets_a, targets_b, lam = self._apply_mixup(images, targets)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self._forward_model(current_model, images)
                    if train and lam < 1.0:
                        loss = lam * self._loss(logits, targets_a) + (1.0 - lam) * self._loss(
                            logits, targets_b
                        )
                    else:
                        loss = self._loss(logits, targets_a)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    if self.abort_on_non_finite:
                        raise FloatingPointError(
                            f"non-finite forward stage={stage} epoch={epoch} step={step}"
                        )
                    print(
                        f"[NON_FINITE_SKIP] stage={stage} epoch={epoch} step={step} "
                        "(non-finite logits/loss)"
                    )
                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip_norm > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_norm,
                            error_if_nonfinite=False,
                        )
                        if not torch.isfinite(grad_norm):
                            print(
                                f"[NON_FINITE_GRAD_SKIP] stage={stage} epoch={epoch} step={step} "
                                f"grad_norm={grad_norm}"
                            )
                            self.optimizer.zero_grad(set_to_none=True)
                            self.scaler.update()
                            continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._update_ema()
            total_loss += loss.item()
            if train and lam < 1.0:
                batch_acc = lam * accuracy(logits.detach(), targets_a) + (1.0 - lam) * accuracy(
                    logits.detach(), targets_b
                )
            else:
                batch_acc = accuracy(logits.detach(), targets_a)
            total_acc += batch_acc
            if not train:
                probs = torch.softmax(logits.detach(), dim=1)
                preds = probs.argmax(dim=1)
                this_count = torch.bincount(preds, minlength=probs.shape[1]).cpu()
                if monitor_counts is None:
                    monitor_counts = this_count
                else:
                    monitor_counts += this_count
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                entropy_sum += float(entropy.sum().item())
                sample_count += int(probs.shape[0])
            avg_loss = total_loss / step
            avg_acc = total_acc / step
            if (
                self.show_batch_log
                and self.log_interval > 0
                and train
                and step % self.log_interval == 0
            ):
                tqdm.write(
                    f"train {epoch}/{epochs} batch {step}/{len(loader)} "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f} mem={_gpu_mem_gb(self.device):.2f}G"
                )
            pbar.set_postfix(
                mem=f"{_gpu_mem_gb(self.device):.2f}G",
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
            )
        n = max(1, len(loader))
        monitor: dict[str, float | int | str] = {
            "pred_majority_idx": -1,
            "pred_majority_ratio": 0.0,
            "pred_entropy": 0.0,
        }
        if not train and monitor_counts is not None and int(monitor_counts.sum().item()) > 0:
            total_preds = int(monitor_counts.sum().item())
            majority_idx = int(torch.argmax(monitor_counts).item())
            majority_ratio = float(monitor_counts[majority_idx].item()) / max(1, total_preds)
            monitor["pred_majority_idx"] = majority_idx
            monitor["pred_majority_ratio"] = majority_ratio
            monitor["pred_entropy"] = entropy_sum / max(1, sample_count)
        return total_loss / n, total_acc / n, monitor

    def _run_val_epoch_with_ema_guard(self, val_loader, epoch: int, epochs: int):
        gate_flags = self._ema_gate_flags(epoch)
        gate_status = self._ema_gate_status_text(gate_flags)
        ema_decay_effective = self._current_ema_decay()
        main_loss, main_acc, main_monitor = self._run_epoch(
            val_loader, train=False, epoch=epoch, epochs=epochs
        )
        if not gate_flags["eligible"]:
            return main_loss, main_acc, "main", gate_status, ema_decay_effective, main_monitor

        if self.ema_model is None:
            return main_loss, main_acc, "main", gate_status, ema_decay_effective, main_monitor

        if self.ema_eval_active:
            ema_loss, ema_acc, ema_monitor = self._run_epoch(
                val_loader,
                train=False,
                epoch=epoch,
                epochs=epochs,
                model_override=self.ema_model,
                stage_override="val(ema)",
            )
            return ema_loss, ema_acc, "ema", gate_status, ema_decay_effective, ema_monitor

        # First-switch guard: only activate EMA eval when it is not significantly worse.
        ema_loss, ema_acc, ema_monitor = self._run_epoch(
            val_loader,
            train=False,
            epoch=epoch,
            epochs=epochs,
            model_override=self.ema_model,
            stage_override="val(ema)",
        )
        acc_drop = main_acc - ema_acc
        if acc_drop > self.ema_switch_max_drop:
            print(
                "[EMA_SWITCH_REJECT] "
                f"epoch={epoch} main_acc={main_acc:.4f} ema_acc={ema_acc:.4f} "
                f"drop={acc_drop:.4f} threshold={self.ema_switch_max_drop:.4f} gate={gate_status}"
            )
            return main_loss, main_acc, "main", gate_status, ema_decay_effective, main_monitor

        self.ema_eval_active = True
        print(
            "[EMA_SWITCH_ON] "
            f"epoch={epoch} main_acc={main_acc:.4f} ema_acc={ema_acc:.4f} "
            f"drop={acc_drop:.4f} threshold={self.ema_switch_max_drop:.4f} gate={gate_status}"
        )
        return ema_loss, ema_acc, "ema", gate_status, ema_decay_effective, ema_monitor

    def fit(self, train_loader, val_loader) -> dict[str, float]:
        train_cfg = self.cfg.get("train", {})
        epochs = _as_int(train_cfg.get("epochs", 50), 50)
        save_every = _as_int(train_cfg.get("save_every", 5), 5)
        early_stopping_patience = _as_int(train_cfg.get("early_stopping_patience", 10), 10)
        metrics: dict[str, float] = {}
        epoch_rows: list[dict[str, float | int | str]] = []
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "val_pred_majority_ratio": [],
            "val_pred_entropy": [],
        }
        self.class_weights = self._compute_class_weights(train_loader)
        if self.class_weights is not None:
            print(f"[CLASS_WEIGHTING] enabled=True weights={self.class_weights.detach().cpu().tolist()}")
        if self.loss_name not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")
        print(f"[LOSS] name={self.loss_name} focal_gamma={self.focal_gamma:.3f}")
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"{'Epoch':>10}{'GPU_mem':>12}{'train_loss':>14}{'train_acc':>12}"
            f"{'val_loss':>12}{'val_acc':>10}{'lr':>12}{'val_model':>12}"
            f"{'ema_upd':>10}{'ema_decay':>12}{'ema_gate':>14}{'tag':>8}"
        )

        for epoch in range(self.start_epoch, epochs + 1):
            self._apply_warmup_lr(epoch)
            train_loss, train_acc, _ = self._run_epoch(
                train_loader, train=True, epoch=epoch, epochs=epochs
            )
            val_loss, val_acc, val_model, ema_gate_status, ema_decay_effective, val_monitor = (
                self._run_val_epoch_with_ema_guard(val_loader, epoch, epochs)
            )
            majority_idx = int(val_monitor.get("pred_majority_idx", -1))
            majority_ratio = float(val_monitor.get("pred_majority_ratio", 0.0))
            pred_entropy = float(val_monitor.get("pred_entropy", 0.0))
            majority_label = (
                self.class_names[majority_idx]
                if self.class_names and 0 <= majority_idx < len(self.class_names)
                else str(majority_idx)
            )
            collapse_flag = (
                self.smoke_monitor_enabled
                and epoch >= self.smoke_monitor_start_epoch
                and majority_ratio >= self.smoke_majority_ratio_threshold
                and (
                    pred_entropy <= self.smoke_entropy_threshold
                    or majority_ratio >= self.smoke_majority_only_threshold
                    or not math.isfinite(pred_entropy)
                )
            )
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            is_best = val_acc > self.best_acc
            row_tag = f"BEST({val_acc:.4f})" if is_best else "-"
            gpu_mem = _gpu_mem_gb(self.device)
            print(
                f"{f'{epoch}/{epochs}':>10}{f'{gpu_mem:.2f}G':>12}"
                f"{train_loss:>14.4f}{train_acc:>12.4f}{val_loss:>12.4f}{val_acc:>10.4f}{lr:>12.6f}"
                f"{val_model:>12}{self.ema_updates:>10}{ema_decay_effective:>12.4f}"
                f"{ema_gate_status:>14}{row_tag:>14}"
            )
            if self.smoke_monitor_enabled and epoch >= self.smoke_monitor_start_epoch:
                print(
                    "[SMOKE_MONITOR] "
                    f"epoch={epoch} majority={majority_label} ratio={majority_ratio:.4f} "
                    f"entropy={pred_entropy:.4f} collapse={int(collapse_flag)}"
                )
            epoch_rows.append(
                {
                    "Epoch": f"{epoch}/{epochs}",
                    "GPU_mem": f"{gpu_mem:.2f}G",
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": lr,
                    "val_model": val_model,
                    "ema_upd": self.ema_updates,
                    "ema_decay": ema_decay_effective,
                    "ema_gate": ema_gate_status,
                    "val_pred_majority": majority_label,
                    "val_pred_majority_ratio": majority_ratio,
                    "val_pred_entropy": pred_entropy,
                    "collapse_flag": int(collapse_flag),
                    "tag": row_tag,
                }
            )

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("train/acc", train_acc, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/acc", val_acc, epoch)
                self.writer.add_scalar("train/lr", lr, epoch)
                self.writer.add_scalar("ema/updates", float(self.ema_updates), epoch)
                self.writer.add_scalar("ema/decay_effective", ema_decay_effective, epoch)
                self.writer.add_scalar("val/use_ema", 1.0 if val_model == "ema" else 0.0, epoch)

            if is_best:
                self.best_acc = val_acc
                self.no_improve_epochs = 0
                self._save_checkpoint(ckpt_dir / "best.pt", epoch)
            else:
                self.no_improve_epochs += 1
            if epoch % save_every == 0 or epoch == epochs:
                self._save_checkpoint(ckpt_dir / "last.pt", epoch)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": self.best_acc,
            }
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(lr)
            history["val_pred_majority_ratio"].append(majority_ratio)
            history["val_pred_entropy"].append(pred_entropy)
            if collapse_flag and self.smoke_abort_on_collapse:
                print(
                    "[SMOKE_MONITOR_ABORT] "
                    f"epoch={epoch} majority_ratio={majority_ratio:.4f} entropy={pred_entropy:.4f}"
                )
                break
            if (
                early_stopping_patience > 0
                and epoch >= self.early_stopping_min_epoch
                and self.no_improve_epochs >= early_stopping_patience
            ):
                print(f"Early stopping triggered at epoch {epoch}, best_val_acc={self.best_acc:.4f}")
                break

        dump_json(self.output_dir / "metrics.json", metrics)
        dump_json(self.output_dir / "history.json", history)
        csv_path = self.output_dir / "history_epoch_table.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "Epoch",
                "GPU_mem",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
                "val_model",
                "ema_upd",
                "ema_decay",
                "ema_gate",
                "val_pred_majority",
                "val_pred_majority_ratio",
                "val_pred_entropy",
                "collapse_flag",
                "tag",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_rows:
                writer.writerow(row)
        plot_training_curves(
            history=history,
            save_path=self.output_dir / "figures" / "training_curves.png",
            title=self.cfg.get("experiment_name", "training"),
        )
        if self.writer is not None:
            self.writer.close()
        return metrics
