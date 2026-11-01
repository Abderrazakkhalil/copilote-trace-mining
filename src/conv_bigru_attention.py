"""Fold-local chrono-position model and reproducible training primitives.

Run the complete experiment with ``python -m src.dl_hybrid_pipeline``.
No precomputed vocabularies, scalers, labels from test, or RowId rules are used.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Required by deterministic CUDA matrix multiplication; set before CUDA work.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.parser import RE_TIME, _parse_action_token, _split_glued_tokens
from src.postprocess import apply_browser_mask


@dataclass
class Trace:
    tokens: list[tuple[str, str, str]]
    controllers: list[str]
    deltas: np.ndarray


def read_traces(path: Path, is_train: bool) -> list[Trace]:
    """Distribute each observed tX interval over its preceding pending actions.

    Individual action latencies are unobserved. Trailing, right-censored actions
    receive the existing extractor's fixed 0.5 s convention. No timing is
    recomputed across the head/tail cut. Shared parser handles malformed tokens.
    """
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        tokens, controllers, deltas, pending = [], [], [], []
        current = 0.0
        for event in _split_glued_tokens(parts[2 if is_train else 1:]):
            event = event.strip()
            if not event:
                continue
            marker = RE_TIME.fullmatch(event)
            if marker:
                target = float(marker.group(1))
                deltas.extend([max(0.0, target - current) / len(pending)] * len(pending)
                              if pending else [])
                pending.clear()
                current = target
            else:
                action = _parse_action_token(event)
                ctrl = action["controller"] or "<none>"
                tokens.append((action["base_action"], ctrl, action["config"] or "<none>"))
                controllers.append(ctrl)
                pending.append(event)
        deltas.extend([0.5] * len(pending))
        assert len(tokens) == len(controllers) == len(deltas)
        traces.append(Trace(tokens, controllers, np.asarray(deltas, dtype=np.float32)))
    return traces


class SequencePreprocessor:
    """Fitted only on the current training subset, including controller stats.

    tempo[0] = standardized log1p(dt); tempo[1] = controller-centered
    log1p(dt), centered at its median and divided by its standard deviation.
    Statistics use full original sessions, before deterministic truncation.
    """
    def __init__(self, max_len: int = 256):
        if max_len != 256:
            raise ValueError("The fixed protocol requires max_len=256")
        self.max_len = max_len

    def fit(self, traces: list[Trace]):
        counts = Counter(t for s in traces for t in s.tokens)
        self.token_vocab = {t: i + 2 for i, t in enumerate(sorted(t for t, n in counts.items() if n >= 2))}
        self.ctrl_vocab = {c: i + 2 for i, c in enumerate(sorted({c for s in traces for c in s.controllers}))}
        all_logs = np.concatenate([np.log1p(s.deltas) for s in traces])
        self.mean = float(all_logs.mean()) if len(all_logs) else 0.0
        self.std = max(float(all_logs.std()), 1e-4) if len(all_logs) else 1.0
        self.fallback = (float(np.median(all_logs)) if len(all_logs) else 0.0, self.std)
        grouped = defaultdict(list)
        for s in traces:
            for ctrl, log_dt in zip(s.controllers, np.log1p(s.deltas)):
                grouped[ctrl].append(float(log_dt))
        self.ctrl_stats = {c: (float(np.median(v)), max(float(np.std(v)), 1e-4))
                           if len(v) >= 2 else self.fallback for c, v in grouped.items()}
        return self

    def transform(self, traces: list[Trace]) -> tuple[np.ndarray, ...]:
        n = len(traces)
        tokens = np.zeros((n, 256), np.int64)
        ctrls = np.zeros_like(tokens)
        tempo = np.zeros((n, 256, 2), np.float32)
        pos = np.zeros((n, 256, 4), np.float32)
        for row, s in enumerate(traces):
            length = len(s.tokens)
            if not length:
                # A synthetic unknown step gives empty sessions a finite representation.
                tokens[row, 0] = ctrls[row, 0] = 1
                continue
            indices = (np.r_[np.arange(128), np.arange(length - 128, length)]
                       if length > 256 else np.arange(length))
            for j, original in enumerate(indices):
                ctrl = s.controllers[original]
                tokens[row, j] = self.token_vocab.get(s.tokens[original], 1)
                ctrls[row, j] = self.ctrl_vocab.get(ctrl, 1)
                log_dt = float(np.log1p(s.deltas[original]))
                median, std = self.ctrl_stats.get(ctrl, self.fallback)
                tempo[row, j] = ((log_dt - self.mean) / self.std,
                                 np.clip((log_dt - median) / std, -5, 5))
                pos[row, j] = (original / length, np.log1p(original) / 8,
                               np.log1p(length - original) / 8,
                               float(length > 256 and j == 128))
        return tokens, ctrls, tempo, pos


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d over valid time steps only; padding never enters statistics."""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = x.transpose(1, 2)[mask]
        normalized = super().forward(valid)
        out = torch.zeros_like(x.transpose(1, 2))
        out[mask] = normalized
        return out.transpose(1, 2)


class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.log_scale = nn.Parameter(torch.tensor(math.log(18.0)))

    def forward(self, x):
        return self.log_scale.exp() * F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))


class ConvBiGRUAttentionNet(nn.Module):
    def __init__(self, num_tokens: int, num_ctrls: int, d_behav: int, num_classes: int = 247):
        super().__init__()
        self.emb_token = nn.Embedding(num_tokens, 80, padding_idx=0)
        self.emb_ctrl = nn.Embedding(num_ctrls, 32, padding_idx=0)
        self.proj_tempo = nn.Sequential(nn.Linear(2, 16), nn.GELU())
        self.proj_pos = nn.Sequential(nn.Linear(4, 16), nn.GELU())
        self.spatial_dropout = nn.Dropout1d(0.25)
        self.convs = nn.ModuleList([nn.Conv1d(144, 72, 1), nn.Conv1d(144, 72, 3, padding=1)])
        self.conv_norms = nn.ModuleList([MaskedBatchNorm1d(72), MaskedBatchNorm1d(72)])
        self.stem_norm = nn.LayerNorm(144)
        self.stem_dropout = nn.Dropout(0.20)
        self.gru = nn.GRU(144, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(272, 128), nn.Tanh(), nn.Linear(128, 6))
        self.context_projection = nn.Linear(6 * 256, 256)
        self.behavior = nn.Sequential(nn.Linear(d_behav, 96), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(0.20))
        self.fusion = nn.Sequential(nn.Linear(608, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25))
        self.head = CosineLinear(256, num_classes)

    def forward(self, token_id, ctrl_id, tempo_feat, pos_feat, x_behav):
        mask = token_id.ne(0)
        if self.training:
            token_id = token_id.masked_fill((torch.rand_like(token_id, dtype=torch.float32) < .10) & mask, 1)
        categorical = torch.cat([self.emb_token(token_id), self.emb_ctrl(ctrl_id)], dim=-1)
        categorical = self.spatial_dropout(categorical.transpose(1, 2)).transpose(1, 2)
        positional = self.proj_pos(pos_feat) * mask.unsqueeze(-1)
        x = torch.cat([categorical, self.proj_tempo(tempo_feat), positional], dim=-1) * mask.unsqueeze(-1)
        conv = torch.cat([F.gelu(bn(layer(x.transpose(1, 2)), mask))
                          for layer, bn in zip(self.convs, self.conv_norms)], dim=1).transpose(1, 2)
        x = self.stem_dropout(self.stem_norm(x + conv)) * mask.unsqueeze(-1)
        lengths = mask.sum(1)
        if torch.all(lengths == x.shape[1]):
            hidden, _ = self.gru(x)
        else:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            hidden, _ = self.gru(packed)
            hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=x.shape[1])
        scores = self.attention(torch.cat([hidden, positional], dim=-1)).transpose(1, 2)
        weights = scores.masked_fill(~mask.unsqueeze(1), -torch.inf).softmax(-1)
        context = self.context_projection(torch.bmm(weights, hidden).flatten(1))
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        fused = self.fusion(torch.cat([context, maximum, self.behavior(x_behav)], dim=-1))
        return self.head(fused)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def make_loader(arrays, y=None, batch_size=64, shuffle=False, seed=42):
    tensors = [torch.from_numpy(np.asarray(x)) for x in arrays]
    if y is not None:
        tensors.append(torch.as_tensor(y, dtype=torch.long))
    # Merge a singleton tail into the preceding batch, so dense BN stays defined.
    indices = np.random.default_rng(seed).permutation(len(tensors[0])) if shuffle else np.arange(len(tensors[0]))
    batches = [indices[i:i + batch_size].tolist() for i in range(0, len(indices), batch_size)]
    if shuffle and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return DataLoader(TensorDataset(*tensors), batch_sampler=batches, num_workers=0)


@torch.no_grad()
def predict(model, arrays, device, batch_size=64):
    model.eval()
    return np.concatenate([model(*(x.to(device) for x in batch)).softmax(-1).cpu().numpy()
                           for batch in make_loader(arrays, batch_size=batch_size)])


def average_states(states):
    return {key: torch.stack([s[key].float() for s in states]).mean(0).to(states[0][key].dtype)
            if states[0][key].is_floating_point() else states[0][key].clone() for key in states[0]}


@torch.no_grad()
def recalibrate_batchnorm(model, arrays, device, batch_size):
    """Refresh averaged checkpoint BN statistics on training inputs exclusively."""
    model.eval()  # Disable masking and dropout, but enable BN updates below.
    modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    momenta = [m.momentum for m in modules]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    for batch in make_loader(arrays, batch_size=batch_size, shuffle=True):
        model(*(x.to(device) for x in batch))
    for module, momentum in zip(modules, momenta):
        module.momentum = momentum
    model.eval()


def training_signature(train_arrays, y, eval_arrays, eval_y, classes, config):
    """Bind an epoch checkpoint to the exact fold tensors and training recipe."""
    import json
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for array in (*train_arrays, y, *eval_arrays, eval_y):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(np.asarray(classes).tolist()).encode())
    # Universal newlines keep the same signature after Windows/Linux checkout.
    digest.update(Path(__file__).read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def save_training_checkpoint(payload, path):
    """A failed write leaves the last complete epoch intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_network(train_arrays, y, eval_arrays, eval_y, classes, eval_browsers,
                  browser_map, num_tokens, num_ctrls, epochs=28, batch_size=64,
                  device="cpu", seed=42, chosen_epochs=None, diagnostic_best=False,
                  checkpoint_path=None):
    """Train with no eval-dependent gradient/scheduler/early stopping decisions.

    Inner fits use the prespecified last 3 checkpoints for cross-fitting. Outer
    fits use three epochs selected from INNER score curves, never outer labels.
    Optionally report the requested outer-best-3 average separately as a biased
    diagnostic; it is never used in nested evaluation or in the submission.
    """
    seed_everything(seed)
    model = ConvBiGRUAttentionNet(num_tokens, num_ctrls, train_arrays[-1].shape[1], len(classes)).to(device)
    counts = np.bincount(y, minlength=len(classes))
    if np.any(counts == 0):
        raise ValueError("Each training subset must contain every class")
    weights = counts.astype(np.float32) ** -.20
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights / weights.mean(), device=device), label_smoothing=.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    selected = set(chosen_epochs if chosen_epochs is not None else range(max(1, epochs - 2), epochs + 1))
    retained, best, curve = {}, [], []
    first_epoch, previous_seconds = 1, 0.0
    config = {"epochs": epochs, "batch_size": batch_size, "seed": seed,
              "selected_epochs": sorted(selected), "diagnostic_best": diagnostic_best,
              "num_tokens": num_tokens, "num_ctrls": num_ctrls,
              "eval_browsers": list(eval_browsers), "browser_map": dict(browser_map)}
    signature = training_signature(train_arrays, y, eval_arrays, eval_y, classes, config) if checkpoint_path else None
    if checkpoint_path and Path(checkpoint_path).exists():
        # This file is created by this project, not an untrusted downloaded model.
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 1 or saved.get("signature") != signature:
            raise ValueError("Checkpoint does not match the fold, data or training configuration")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        retained, best, curve = saved["retained"], saved["best"], saved["curve"]
        first_epoch, previous_seconds = saved["epoch"] + 1, saved["seconds"]
        random.setstate(saved["rng_python"])
        np.random.set_state(saved["rng_numpy"])
        torch.set_rng_state(saved["rng_torch"])
        if torch.cuda.is_available() and saved["rng_cuda"] and len(saved["rng_cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        print(f"    resume epoch {first_epoch}/{epochs} from {checkpoint_path} ({saved['device']} -> {device})", flush=True)
    start = time.monotonic()
    for epoch in range(first_epoch, epochs + 1):
        model.train()
        losses = []
        for batch in make_loader(train_arrays, y, batch_size, shuffle=True, seed=seed + epoch):
            inputs, targets = batch[:-1], batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(x.to(device) for x in inputs))
            loss = loss_fn(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        probabilities = apply_browser_mask(predict(model, eval_arrays, device, batch_size), eval_browsers, classes, browser_map)
        score = f1_score(eval_y, probabilities.argmax(1), labels=np.arange(len(classes)), average="macro", zero_division=0)
        curve.append(float(score))
        state = None
        if epoch in selected or diagnostic_best and (len(best) < 3 or score > best[-1][0]):
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch in selected:
            retained[epoch] = state
        if diagnostic_best and state is not None:
            best.append((float(score), epoch, state))
            best.sort(key=lambda item: (-item[0], item[1]))
            best = best[:3]
        elapsed = previous_seconds + time.monotonic() - start
        if checkpoint_path:
            save_training_checkpoint({
                "format_version": 1, "signature": signature, "epoch": epoch,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "retained": retained, "best": best, "curve": curve, "seconds": elapsed,
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "device": str(device), "torch_version": str(torch.__version__),
            }, checkpoint_path)
        print(f"    epoch {epoch:02d}/{epochs} loss={np.mean(losses):.4f} masked_f1={score:.5f} elapsed={elapsed:.0f}s saved={bool(checkpoint_path)}", flush=True)
    diagnostic = None
    if diagnostic_best:
        model.load_state_dict(average_states([x[2] for x in best]))
        recalibrate_batchnorm(model, train_arrays, device, batch_size)
        diagnostic = predict(model, eval_arrays, device, batch_size)
    model.load_state_dict(average_states(list(retained.values())))
    recalibrate_batchnorm(model, train_arrays, device, batch_size)
    return model, {"curve": curve, "chosen_epochs": sorted(selected),
                   "diagnostic_best_epochs": [x[1] for x in best], "diagnostic_probs": diagnostic,
                   "seconds": previous_seconds + time.monotonic() - start}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    from src.dl_hybrid_pipeline import main
    main()
