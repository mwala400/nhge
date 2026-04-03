"""
NHGE Training Engine
=====================
Full training pipeline for the Neuro-Harmonic Graph Engine.
Supports:
  - Language modelling (causal or masked)
  - Text classification
  - Custom task heads
  - Mixed precision training
  - Gradient clipping
  - Warmup + cosine LR schedule
  - Convergence metrics logging
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Dict, Any, Tuple
import math
import time

try:
    from nhge.nhge_model import NHGE
except ImportError:
    from nhge_model import NHGE


# ---------------------------------------------------------------------------
# Learning rate schedule with linear warmup
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup → cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self._step += 1
        lrs = self._get_lrs()
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g["lr"] = lr

    def _get_lrs(self):
        s, W, T = self._step, self.warmup_steps, self.total_steps
        out = []
        for base_lr in self._base_lrs:
            if s < W:
                lr = base_lr * (s / max(1, W))
            else:
                progress = (s - W) / max(1, T - W)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            out.append(lr)
        return out

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Simple in-memory text dataset (toy / demo)
# ---------------------------------------------------------------------------
class TokenDataset(Dataset):
    """
    Wraps pre-tokenized sequences.
    tokens: list of List[int]
    labels: list of int (for classification) or None (for LM)
    """

    def __init__(
        self,
        tokens: list,
        labels: Optional[list] = None,
        max_len: int = 128,
        pad_id: int = 0,
    ):
        self.tokens = tokens
        self.labels = labels
        self.max_len = max_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        seq = self.tokens[idx][: self.max_len]
        pad_len = self.max_len - len(seq)
        ids = seq + [self.pad_id] * pad_len
        mask = [False] * len(seq) + [True] * pad_len

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class NHGETrainer:
    """
    Full-featured trainer for NHGE models.

    Usage:
        trainer = NHGETrainer(model, train_loader, val_loader,
                              task="lm", device="cuda")
        trainer.train(epochs=10)
    """

    def __init__(
        self,
        model: NHGE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task: str = "lm",  # "lm" | "cls"
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        warmup_ratio: float = 0.05,
        device: str = "cpu",
        use_amp: bool = False,  # mixed precision (requires cuda)
        log_interval: int = 100,
        save_path: Optional[str] = None,
        callback: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.grad_clip = grad_clip
        self.device = device
        self.log_interval = log_interval
        self.save_path = save_path
        self.callback = callback
        self.use_amp = use_amp and device.startswith("cuda")

        # Optimizer — separate weight decay from embeddings/norms
        decay_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in ["bias", "norm", "embed"])
        ]
        nodecay_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and any(nd in n for nd in ["bias", "norm", "embed"])
        ]

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # Scheduler (set up after we know total steps)
        self._lr = lr
        self._warmup_ratio = warmup_ratio
        self.scheduler: Optional[WarmupCosineScheduler] = None

        # Loss
        if task == "lm":
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "avg_iters": [],
            "lr": [],
        }

    def _setup_scheduler(self, total_steps: int):
        warmup = max(1, int(total_steps * self._warmup_ratio))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup, total_steps
        )

    def _forward(self, batch) -> Tuple[float, int]:
        ids = batch["input_ids"].to(self.device)
        mask = batch["mask"].to(self.device)

        if self.task == "lm":
            out = self.model(ids, mask, lm_mode=True, return_iterations=True)
            # Causal LM: predict next token at each position
            logits = out["logits"][:, :-1, :].contiguous()
            labels = ids[:, 1:].contiguous()
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        else:
            out = self.model(ids, mask, lm_mode=False, return_iterations=True)
            labels = batch["labels"].to(self.device)
            loss = self.criterion(out["logits"], labels)

        return loss, out.get("iterations", 1)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_iters = 0
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, iters = self._forward(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, iters = self._forward(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            total_iters += iters
            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "avg_iters": total_iters / max(1, n_batches),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_iters = 0
        n_batches = 0

        for batch in self.val_loader:
            loss, iters = self._forward(batch)
            total_loss += loss.item()
            total_iters += iters
            n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "avg_iters": total_iters / max(1, n_batches),
        }

    def train(self, epochs: int = 10):
        total_steps = epochs * len(self.train_loader)
        self._setup_scheduler(total_steps)

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = self.train_epoch()
            val = self.evaluate()
            elapsed = time.time() - t0

            lr_now = self.scheduler.current_lr if self.scheduler else self._lr

            self.history["train_loss"].append(tr["loss"])
            self.history["avg_iters"].append(tr["avg_iters"])
            self.history["lr"].append(lr_now)
            if val:
                self.history["val_loss"].append(val["loss"])

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={tr['loss']:.4f} | "
                f"val_loss={val.get('loss', float('nan')):.4f} | "
                f"avg_iters={tr['avg_iters']:.1f} | "
                f"lr={lr_now:.2e} | "
                f"time={elapsed:.1f}s"
            )

            # Save best model
            if (
                self.save_path
                and val
                and val.get("loss", float("inf")) < best_val_loss
            ):
                best_val_loss = val["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    self.save_path,
                )
                print(f" → Saved best model (val_loss={best_val_loss:.4f})")

            if self.callback:
                self.callback(epoch, self.model, self.history)

        return self.history