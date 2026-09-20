"""Fine-tune the end-to-end sentence rewriter.

One model, one objective: the full noisy sentence in, the full corrected
sentence out. No detector, no router, no rules -- every decision the previous
pipeline made across five components is made here by cross-attention.

What is measured
----------------
Sentence exact-match is reported, but it is not the number to steer by. The
metric that decides whether the model is usable is the **Unnecessary
Modification Rate**: of the tokens that should have been left alone, how many
were changed. A rewriter that improves conversion while quietly corrupting
ordinary Arabic is worse than no rewriter at all, and exact-match alone cannot
tell the two apart.

Checkpoints are therefore selected on ``sentence_exact - umr_penalty * UMR``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model.seq2seq import Seq2SeqConfig, encode_example, load_model_and_tokenizer
from .dataset import balanced_subset, read_jsonl
from .metrics import EndToEndMetrics


def pick_amp(device: torch.device) -> tuple[bool, torch.dtype]:
    """Fastest safe autocast dtype for this GPU.

    Ampere and newer (A100, L4) have real bf16, which is numerically forgiving;
    a T4 must use fp16 and therefore also needs the gradient scaler.
    """
    if device.type != "cuda":
        return False, torch.float32
    major = torch.cuda.get_device_capability(0)[0]
    return True, torch.bfloat16 if major >= 8 else torch.float16



class Seq2SeqDataset(Dataset):
    """Whole sentences: noisy ASR text -> corrected code-switched text."""

    def __init__(self, rows: list[dict], tokenizer, cfg: Seq2SeqConfig) -> None:
        self.rows = rows
        self.tok = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        enc = encode_example(self.tok, r["src"], r["tgt"], self.cfg)
        enc["row"] = i
        return enc


class Seq2SeqCollator:
    """Pads a batch. Picklable, so ``num_workers > 0`` works under spawn."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tok = tokenizer
        self.pad_to = pad_to_multiple_of

    def __call__(self, batch: list[dict]) -> dict:
        rows = [b.pop("row") for b in batch]
        labels = [b.pop("labels") for b in batch] if "labels" in batch[0] else None

        padded = self.tok.pad(
            batch, padding=True, pad_to_multiple_of=self.pad_to, return_tensors="pt"
        )
        if labels is not None:
            n = max(len(l) for l in labels)
            if self.pad_to:
                n = ((n + self.pad_to - 1) // self.pad_to) * self.pad_to
            # -100 so padded positions contribute nothing to the loss.
            padded["labels"] = torch.tensor(
                [l + [-100] * (n - len(l)) for l in labels], dtype=torch.long
            )
        padded["rows"] = torch.tensor(rows, dtype=torch.long)
        return padded


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    loader: DataLoader,
    rows: list[dict],
    cfg: Seq2SeqConfig,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> tuple[dict, list[tuple[str, str, str]]]:
    model.eval()
    m = EndToEndMetrics()
    losses: list[float] = []
    samples: list[tuple[str, str, str]] = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        idx = batch.pop("rows")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            losses.append(float(model(**batch).loss))

        gen = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=cfg.max_target_length,
            num_beams=cfg.num_beams,
            early_stopping=cfg.num_beams > 1,
        )
        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, pred in enumerate(preds):
            r = rows[int(idx[j])]
            m.update(
                pred_tokens=pred.split(),
                gold_tokens=r["tgt"].split(),
                src_tokens=r["src_tokens"],
                gold_tags=r["tags"],
            )
            if len(samples) < 10:
                samples.append((r["src"], pred, r["tgt"]))

    res = m.compute()
    res["loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return res, samples


def train_seq2seq(cfg: dict) -> dict:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, amp_dtype = pick_amp(device)
    torch.manual_seed(cfg.get("seed", 42))

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg = Seq2SeqConfig.from_dict(cfg)
    model, tok = load_model_and_tokenizer(mcfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mcfg.model_name}: {n_params/1e6:.1f}M parameters")

    if cfg.get("gradient_checkpointing", False):
        # Trades compute for memory: byte-level sequences are long and this is
        # what makes a larger batch fit.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = balanced_subset(
        read_jsonl(data_dir / "train.jsonl"), cfg.get("max_train_examples")
    )
    val_rows = balanced_subset(
        read_jsonl(data_dir / "validation.jsonl"), cfg.get("max_eval_examples")
    )
    print(f"[data] train={len(train_rows):,} val={len(val_rows):,}")

    collate = Seq2SeqCollator(tok)
    train_loader = DataLoader(
        Seq2SeqDataset(train_rows, tok, mcfg),
        batch_size=cfg.get("batch_size", 16), shuffle=True, collate_fn=collate,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        Seq2SeqDataset(val_rows, tok, mcfg),
        batch_size=cfg.get("eval_batch_size", 16), shuffle=False,
        collate_fn=collate, num_workers=cfg.get("num_workers", 4),
    )

    epochs = cfg.get("epochs", 3)
    accum = cfg.get("grad_accum", 1)
    total_steps = max(1, (len(train_loader) // accum) * epochs)

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
        lr=cfg.get("learning_rate", 1e-4),
    )
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * cfg.get("warmup_ratio", 0.03)), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype is torch.float16)

    run = None
    if cfg.get("wandb_project"):
        try:
            import wandb

            run = wandb.init(
                project=cfg["wandb_project"], name=cfg.get("run_name", "seq2seq"),
                config={**cfg, **mcfg.to_dict(), "n_params": n_params}, reinit=True,
            )
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    best = -1e9
    step = 0
    oom_skipped = 0
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 1000)
    penalty = cfg.get("umr_penalty", 1.0)
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch.pop("rows")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            try:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    loss = model(**batch).loss / accum
                scaler.scale(loss).backward()
                running += float(loss) * accum
            except torch.OutOfMemoryError:
                oom_skipped += 1
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom_skipped in (1, 10, 100):
                    print(f"[warn ] skipped {oom_skipped} batch(es) after OOM "
                          f"-- lower batch_size or raise grad_accum", flush=True)
                continue

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
                    sps = step / max(1e-9, time.time() - t0)
                    print(f"[train] epoch {epoch+1} step {step}/{total_steps} "
                          f"loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"({sps:.2f} steps/s)", flush=True)
                    if run:
                        run.log({"train/loss": avg,
                                 "train/lr": sched.get_last_lr()[0]}, step=step)
                    running = 0.0

                if step % eval_every == 0 or step == total_steps:
                    res, samples = evaluate(
                        model, tok, val_loader, val_rows, mcfg, device,
                        amp=amp, amp_dtype=amp_dtype,
                        max_batches=cfg.get("eval_max_batches", 16),
                    )
                    print(f"[eval ] step {step} "
                          f"exact {res['sentence_exact_match']:.4f} "
                          f"UMR {res['unnecessary_modification_rate']:.4f} "
                          f"conv {res['conversion_accuracy']:.4f}", flush=True)
                    for s, p, g in samples[:4]:
                        ok = p.strip() == g.strip()
                        print(f"        {'OK ' if ok else 'BAD'} {p[:88]}")
                        if not ok:
                            print(f"            gold {g[:88]}")
                    if run:
                        run.log({f"val/{k}": v for k, v in res.items()}, step=step)

                    score = res["sentence_exact_match"] - penalty * res[
                        "unnecessary_modification_rate"
                    ]
                    if score > best:
                        best = score
                        save_seq2seq(model, tok, mcfg, out_dir, res)
                        print(f"[ckpt ] new best score {score:.4f}")

    print(f"[done ] {time.time()-t0:.0f}s  best {best:.4f}")
    if run:
        run.summary.update({"best_score": best})
        run.finish()
    return {"best_score": best}


def save_seq2seq(model, tokenizer, cfg: Seq2SeqConfig, out_dir: Path, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "seq2seq")
    tokenizer.save_pretrained(out_dir / "seq2seq")
    (out_dir / "seq2seq_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "seq2seq_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
