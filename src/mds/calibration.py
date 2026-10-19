"""Temperature scaling and expected calibration error.

A model is "calibrated" if, among cells it labels with 80% confidence, about
80% are right. Deep nets are usually over-confident. Temperature scaling
divides all logits by one number T > 1 (fitted on validation data), which
softens probabilities without changing which class is ranked first, so
accuracy and macro-F1 are unchanged; only the confidence numbers improve.
That matters for the app's "low confidence" banner and the soft blast count.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Find T minimising validation negative log-likelihood (optimised in log T)."""
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |confidence - accuracy| over confidence bins, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, accuracy, count) for a reliability diagram."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / np.maximum(counts, 1)
    acc = np.bincount(idx, weights=correct, minlength=n_bins) / np.maximum(counts, 1)
    return mean_conf, acc, counts
