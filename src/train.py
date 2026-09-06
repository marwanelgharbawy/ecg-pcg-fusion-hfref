from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.88, gamma=2.0):
        super().__init__()
        if not 0 < alpha < 1 or gamma < 0:
            raise ValueError("alpha must be in (0, 1) and gamma must be nonnegative.")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def _forward_batch(model, batch, device):
    # ECG-only: (ecg, labels); fusion: (ecg, pcg, labels).
    *inputs, labels = batch
    inputs = [values.to(device, non_blocking=True) for values in inputs]
    labels = labels.to(device, non_blocking=True).float().reshape(-1, 1)
    logits = model(*inputs)
    if logits.shape != labels.shape:
        raise ValueError(f"Expected one logit per patient: {labels.shape}; got {logits.shape}.")
    return logits, labels


@torch.no_grad()
def evaluate_model(model, loader, device=None, criterion=None):
    """Return one probability per patient, in loader order, and ranking metrics."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    count = 0
    for batch in loader:
        logits, labels = _forward_batch(model, batch, device)
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced nonfinite logits.")
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        count += labels.size(0)
        all_labels.append(labels.cpu().numpy().ravel())
        all_probs.append(logits.sigmoid().cpu().numpy().ravel())
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    labels, probs = np.concatenate(all_labels), np.concatenate(all_probs)
    both_classes = len(np.unique(labels)) == 2
    return {
        "labels": labels, "probabilities": probs,
        "loss": total_loss / count if criterion is not None else None,
        "auroc": roc_auc_score(labels, probs) if both_classes else float("nan"),
        "auprc": average_precision_score(labels, probs) if both_classes else float("nan"),
    }


def train_model(model, train_loader, val_loader, num_epochs=50, device=None,
                learning_rate=1e-3, weight_decay=1e-4, patience=10,
                criterion=None, checkpoint_path=None):
    """Train with AdamW, reduce LR on plateaus, and restore best validation AUROC.

    A supplied criterion (e.g. focal loss) takes precedence. Otherwise calculate
    weighted BCE from the training patients, never from validation or test labels.
    """
    if num_epochs < 1 or patience < 1:
        raise ValueError("num_epochs and patience must be positive.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    if criterion is None:
        labels = train_loader.dataset.metadata["Label"]
        positives, negatives = int(labels.sum()), int((labels == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("Training data must include both classes.")
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device))
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    best_auroc, best_state, stale_epochs = -float("inf"), None, 0
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            logits, labels = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            count += labels.size(0)
        if count == 0:
            raise ValueError("Cannot train with an empty loader.")
        validation = evaluate_model(model, val_loader, device, criterion)
        auroc = validation["auroc"]
        if not np.isfinite(auroc):
            raise ValueError("Validation AUROC requires both classes.")
        history.append({"epoch": epoch, "train_loss": total_loss / count,
                        "val_loss": validation["loss"], "val_auroc": auroc,
                        "val_auprc": validation["auprc"], "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch} | Train {total_loss / count:.4f} | "
              f"Val {validation['loss']:.4f} | AUROC {auroc:.4f} | "
              f"AUPRC {validation['auprc']:.4f}")
        scheduler.step(auroc)
        if auroc > best_auroc:
            best_auroc, stale_epochs = auroc, 0
            best_state = deepcopy(model.state_dict())
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping; best validation AUROC: {best_auroc:.4f}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history)
