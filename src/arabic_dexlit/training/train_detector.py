"""Stage-1 training loop.

Design choices that keep a full run inside a Colab session rather than a week:

* **bf16/fp16 autocast** on A100/L4/T4, chosen from the GPU's own capability.
* **Dynamic padding** to the longest sequence in a batch, not a fixed 128 --
  most sentences are short, so this is the single biggest throughput win.
* **Optional bottom-layer freezing.** The lower layers of an Arabic encoder
  already model Arabic; on a T4, freezing them cuts step time substantially for
  a small accuracy cost.
* **Evaluation capped by batch count**, so validation never dominates a run on
  a large corpus.

Checkpoint selection uses a **composite** score, not raw F1:

    score = span_f1 - false_edit_penalty * false_edit_rate

because a model that scores well on F1 while quietly corrupting monolingual
Arabic is not the model we want to ship.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..schema import ID2TAG, IGNORE_INDEX, NUM_TAGS
from ..model.detector import DetectorConfig, SpanDetector
from .dataset import (
    DetectorCollator,
    DetectorDataset,
    balanced_subset,
    read_jsonl,
)
from .metrics import DetectorMetrics
from . import viz


def pick_amp(device: torch.device) -> tuple[bool, torch.dtype]:
    """Choose the fastest safe autocast dtype for this GPU."""
    if device.type != "cuda":
        return False, torch.float32
    major = torch.cuda.get_device_capability(0)[0]
    # Ampere and newer (A100, L4) have real bf16; T4 (7.5) must use fp16.
    return True, torch.bfloat16 if major >= 8 else torch.float16


@torch.no_grad()
def evaluate(
    model: SpanDetector,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
    copy_threshold: float | None = None,
    collect_confusion: bool = False,
) -> tuple[dict, np.ndarray | None]:
    model.eval()
    m = DetectorMetrics()
    preds_all: list[list[int]] = []
    golds_all: list[list[int]] = []
    losses: list[float] = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            out = model(**batch)
        losses.append(float(out["loss"]))

        logits = out["logits"].float()
        pred = logits.argmax(-1)
        if copy_threshold is not None and "copy_logits" in out:
            keep = torch.sigmoid(out["copy_logits"].float()) > copy_threshold
            pred = torch.where(keep, torch.zeros_like(pred), pred)

        p = pred.cpu().tolist()
        g = batch["labels"].cpu().tolist()
        m.update(p, g)
        if collect_confusion:
            preds_all.extend(p)
            golds_all.extend(g)

    res = m.compute()
    res["loss"] = float(np.mean(losses)) if losses else 0.0
    cm = (
        viz.confusion_from_pairs(preds_all, golds_all, NUM_TAGS)
        if collect_confusion and preds_all
        else None
    )
    model.train()
    return res, cm


def train_detector(cfg: dict) -> dict:
    """Run stage-1 training. ``cfg`` is the resolved config dict (see configs/)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, amp_dtype = pick_amp(device)
    torch.manual_seed(cfg.get("seed", 42))

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["encoder_name"], use_fast=True)

    train_rows = balanced_subset(
        read_jsonl(data_dir / "train.jsonl"), cfg.get("max_train_examples")
    )
    val_rows = balanced_subset(
        read_jsonl(data_dir / "validation.jsonl"), cfg.get("max_eval_examples")
    )
    print(f"[data] train={len(train_rows):,} val={len(val_rows):,}")

    max_len = cfg.get("max_length", 128)
    train_ds = DetectorDataset(train_rows, tok, max_len)
    val_ds = DetectorDataset(val_rows, tok, max_len)

    collate = DetectorCollator(tok.pad_token_id or 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("eval_batch_size", 64),
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    dcfg = DetectorConfig(
        encoder_name=cfg["encoder_name"],
        dropout=cfg.get("dropout", 0.1),
        use_script_features=cfg.get("use_script_features", True),
        use_copy_gate=cfg.get("use_copy_gate", True),
        copy_gate_weight=cfg.get("copy_gate_weight", 0.3),
        positive_weight=cfg.get("positive_weight", 2.0),
        freeze_encoder_layers=cfg.get("freeze_encoder_layers", 0),
    )
    model = SpanDetector(dcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M parameters, amp={amp} ({amp_dtype})")

    epochs = cfg.get("epochs", 3)
    accum = cfg.get("grad_accum", 1)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * epochs

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.get("weight_decay", 0.01)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.get("learning_rate", 3e-5),
    )
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * cfg.get("warmup_ratio", 0.06)), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype is torch.float16)

    run = None
    if cfg.get("wandb_project"):
        try:
            import wandb

            run = wandb.init(
                project=cfg["wandb_project"],
                name=cfg.get("run_name"),
                config={**cfg, **asdict(dcfg), "n_params": n_params},
                reinit=True,
            )
        except Exception as e:  # never let logging kill a training run
            print(f"[wandb] disabled: {e}")

    best_score = -1.0
    history: list[dict] = []
    step = 0
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 500)
    penalty = cfg.get("false_edit_penalty", 0.5)
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                out = model(**batch)
                loss = out["loss"] / accum
            scaler.scale(loss).backward()
            running += out["loss"].detach().item()

            if (i + 1) % accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0)
                )
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                if step % log_every == 0:
                    avg = running / (log_every * accum)
                    lr = sched.get_last_lr()[0]
                    sps = step / max(1e-9, time.time() - t0)
                    print(
                        f"[train] epoch {epoch+1} step {step}/{total_steps} "
                        f"loss {avg:.4f} lr {lr:.2e} ({sps:.2f} steps/s)",
                        flush=True,
                    )
                    if run:
                        run.log(
                            {
                                "train/loss": avg,
                                "train/tag_loss": float(out.get("tag_loss", 0.0)),
                                "train/copy_loss": float(out.get("copy_loss", 0.0)),
                                "train/lr": lr,
                                "train/steps_per_sec": sps,
                                "epoch": epoch + (i + 1) / max(1, len(train_loader)),
                            },
                            step=step,
                        )
                    running = 0.0

                if step % eval_every == 0 or step == total_steps:
                    res, cm = evaluate(
                        model, val_loader, device, amp=amp, amp_dtype=amp_dtype,
                        max_batches=cfg.get("eval_max_batches"),
                        collect_confusion=True,
                    )
                    res["step"] = step
                    history.append(res)
                    score = res["span_f1"] - penalty * res["false_edit_rate"]
                    print(
                        f"[eval ] step {step} f1 {res['span_f1']:.4f} "
                        f"pass {res['passthrough_accuracy']:.4f} "
                        f"fer {res['false_edit_rate']:.4f} score {score:.4f}",
                        flush=True,
                    )
                    if run:
                        payload = {f"val/{k}": v for k, v in res.items() if k != "step"}
                        payload["val/score"] = score
                        figs = _figures(model, tok, val_rows, res, cm, history, device)
                        import wandb

                        payload.update(
                            {k: wandb.Image(f) for k, f in figs.items() if f is not None}
                        )
                        run.log(payload, step=step)
                        for f in figs.values():
                            if f is not None:
                                import matplotlib.pyplot as plt

                                plt.close(f)

                    if score > best_score:
                        best_score = score
                        save_detector(model, tok, dcfg, out_dir, res)
                        print(f"[ckpt ] new best score {score:.4f} -> {out_dir}")

    final, _ = evaluate(model, val_loader, device, amp=amp, amp_dtype=amp_dtype)
    print(f"[done ] {time.time()-t0:.0f}s  best score {best_score:.4f}")
    if run:
        run.summary.update({"best_score": best_score, **{f"final/{k}": v for k, v in final.items()}})
        run.finish()
    return {"best_score": best_score, "final": final, "history": history}


def _figures(model, tok, val_rows, res, cm, history, device) -> dict:
    """Build the diagnostic figures logged at each eval."""
    figs: dict = {}
    try:
        figs["viz/per_category_f1"] = viz.plot_category_scores(res)
        figs["viz/safety"] = viz.plot_safety(history)
        if cm is not None:
            figs["viz/confusion"] = viz.plot_confusion(cm, viz.tag_labels(NUM_TAGS))

        # One qualitative example: a sentence with real spans in it.
        row = next((r for r in val_rows if r.get("spans")), None)
        if row is not None:
            words = row["src_tokens"][:40]
            enc = tok(words, is_split_into_words=True, truncation=True,
                      max_length=128, return_tensors="pt")
            wid = enc.word_ids()
            from ..model.detector import script_of

            scripts = torch.tensor(
                [[0 if w is None else script_of(words[w]) for w in wid]], dtype=torch.long
            )
            enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            model.eval()
            with torch.no_grad():
                out = model(**enc, script_ids=scripts.to(device), output_attentions=True)
            model.train()
            pred = out["logits"].argmax(-1)[0].cpu().tolist()

            word_tags, seen = [], set()
            for pos, w in enumerate(wid):
                if w is None or w in seen:
                    continue
                seen.add(w)
                word_tags.append(ID2TAG.get(int(pred[pos]), "O"))
            figs["viz/edit_decisions"] = viz.plot_edit_decisions(
                words, word_tags, row["tags"][: len(word_tags)]
            )
            if out.get("attentions"):
                a = out["attentions"][-1][0].mean(0).float().cpu().numpy()
                pieces = tok.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
                figs["viz/attention"] = viz.plot_attention(pieces[:32], a[:32, :32])
    except Exception as e:  # diagnostics must never break training
        print(f"[viz  ] skipped: {e}")
    return figs


def save_detector(model, tok, dcfg: DetectorConfig, out_dir: Path, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "detector.pt")
    (out_dir / "detector_config.json").write_text(
        json.dumps(dcfg.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "detector_metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items()}, indent=2), encoding="utf-8"
    )
    tok.save_pretrained(out_dir)
