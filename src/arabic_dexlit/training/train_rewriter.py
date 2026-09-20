"""Training loop for the dual-conditioned rewriter (stage 2, seq2seq variant).

This is an alternative to the span-routing converter, not a replacement for the
detector: it still consumes the detector's tags. The difference is that it
rewrites the **whole sentence** conditioned on those tags, instead of routing
each span through an isolated model. That lets it resolve relationships between
spans -- a three-word company name stays one name -- which span routing cannot.

Training uses **gold tags with dropout** rather than the detector's predictions.
Gold tags give a clean learning signal; the dropout (``tag_dropout``) stops the
decoder treating tags as infallible, so at inference a detector mistake degrades
the output instead of dictating it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..model.dual_seq2seq import (
    BOS_ID,
    PAD_ID,
    DualConditionedRewriter,
    DualSeq2SeqConfig,
)
from ..model.word_vocab import WordVocab
from ..schema import OUTSIDE, TAG2ID
from .dataset import balanced_subset, read_jsonl
from .metrics import EndToEndMetrics
from .train_detector import pick_amp


class RewriterDataset(Dataset):
    """Whole sentences: (source tokens, tags) -> target tokens."""

    def __init__(
        self,
        rows: list[dict],
        vocab: WordVocab,
        *,
        max_src_len: int = 48,
        max_tgt_len: int = 48,
        num_tags: int = len(TAG2ID),
    ) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.no_tag_id = num_tags

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        src_tokens = r["src_tokens"][: self.max_src_len - 1]
        tags = r["tags"][: len(src_tokens)]
        tgt_tokens = r["tgt"].split()[: self.max_tgt_len - 1]

        src = self.vocab.encode(src_tokens, self.max_src_len)
        tgt = self.vocab.encode(tgt_tokens, self.max_tgt_len)
        # One tag per source id; the trailing EOS position carries "no tag".
        tag_ids = [TAG2ID.get(t, TAG2ID[OUTSIDE]) for t in tags] + [self.no_tag_id]
        tag_ids = tag_ids[: len(src)]
        tag_ids += [self.no_tag_id] * (len(src) - len(tag_ids))

        return {
            "src": src,
            "tags": tag_ids,
            "tgt_in": [BOS_ID] + tgt[:-1],
            "tgt_out": tgt,
            "row": i,
        }


class RewriterCollator:
    """Picklable collator (spawn-safe on Windows)."""

    def __init__(self, no_tag_id: int) -> None:
        self.no_tag_id = no_tag_id

    def __call__(self, batch: list[dict]) -> dict:
        ns = max(len(b["src"]) for b in batch)
        nt = max(len(b["tgt_in"]) for b in batch)

        def pad(key: str, n: int, fill: int) -> torch.Tensor:
            return torch.tensor(
                [b[key] + [fill] * (n - len(b[key])) for b in batch], dtype=torch.long
            )

        return {
            "src": pad("src", ns, PAD_ID),
            "tags": pad("tags", ns, self.no_tag_id),
            "tgt_in": pad("tgt_in", nt, PAD_ID),
            "tgt_out": pad("tgt_out", nt, PAD_ID),
            "rows": torch.tensor([b["row"] for b in batch], dtype=torch.long),
        }


@torch.no_grad()
def evaluate(
    model: DualConditionedRewriter,
    loader: DataLoader,
    rows: list[dict],
    vocab: WordVocab,
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
            out = model(batch["src"], batch["tags"], batch["tgt_in"], batch["tgt_out"])
        losses.append(float(out["loss"]))

        gen, copy_pos = model.generate(
            batch["src"], batch["tags"], return_copy=True
        )
        copy_pos = copy_pos.cpu().tolist()
        for j, row_ids in enumerate(gen.cpu().tolist()):
            r = rows[int(idx[j])]
            pred = vocab.decode(row_ids, r["src_tokens"], copy_pos[j])
            m.update(
                pred_tokens=pred,
                gold_tokens=r["tgt"].split(),
                src_tokens=r["src_tokens"],
                gold_tags=r["tags"],
            )
            if len(samples) < 8:
                samples.append((r["src"], " ".join(pred), r["tgt"]))

    res = m.compute()
    res["loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return res, samples


def train_rewriter(cfg: dict) -> dict:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, amp_dtype = pick_amp(device)
    torch.manual_seed(cfg.get("seed", 42))

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = balanced_subset(
        read_jsonl(data_dir / "train.jsonl"), cfg.get("max_train_examples")
    )
    val_rows = balanced_subset(
        read_jsonl(data_dir / "validation.jsonl"), cfg.get("max_eval_examples")
    )

    # One shared vocabulary: 84% of output tokens are copied from the input, so
    # separate source/target vocabularies would duplicate nearly everything and
    # break the copy mechanism's id alignment.
    vocab = WordVocab.build(
        train_rows,
        max_size=cfg.get("vocab_size", 32000),
        min_count=cfg.get("vocab_min_count", 2),
    )
    cov = vocab.coverage(val_rows[:5000])
    print(
        f"[vocab] {len(vocab):,} words | token coverage {cov['token_coverage']:.1%} "
        f"| of the rest, {cov['oov_recoverable_by_copy']:.1%} recoverable by copy"
    )
    vocab.save(out_dir / "rewriter_vocab.json")

    mcfg = DualSeq2SeqConfig.from_dict({**cfg, "vocab_size": len(vocab)})
    model = DualConditionedRewriter(mcfg).to(device)
    print(f"[model] rewriter {model.num_parameters()/1e6:.2f}M parameters "
          f"(copy={mcfg.use_copy}, tag_dropout={mcfg.tag_dropout})")

    train_ds = RewriterDataset(train_rows, vocab,
                               max_src_len=mcfg.max_src_len, max_tgt_len=mcfg.max_tgt_len)
    val_ds = RewriterDataset(val_rows, vocab,
                             max_src_len=mcfg.max_src_len, max_tgt_len=mcfg.max_tgt_len)
    collate = RewriterCollator(mcfg.num_tags)
    print(f"[data] train={len(train_ds):,} val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 128), shuffle=True,
        collate_fn=collate, num_workers=cfg.get("num_workers", 4),
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("eval_batch_size", 128), shuffle=False,
        collate_fn=collate, num_workers=cfg.get("num_workers", 4),
    )

    epochs = cfg.get("epochs", 6)
    total_steps = max(1, len(train_loader) * epochs)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg.get("learning_rate", 3e-4), total_steps=total_steps,
        pct_start=max(0.01, cfg.get("warmup_ratio", 0.05)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype is torch.float16)

    run = None
    if cfg.get("wandb_project"):
        try:
            import wandb

            run = wandb.init(project=cfg["wandb_project"],
                             name=cfg.get("run_name", "rewriter"),
                             config={**cfg, **mcfg.to_dict()}, reinit=True)
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    best = -1.0
    step = 0
    oom_skipped = 0
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 500)
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in train_loader:
            batch.pop("rows")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            try:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    out = model(batch["src"], batch["tags"],
                                batch["tgt_in"], batch["tgt_out"])
                    loss = out["loss"]
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("max_grad_norm", 1.0)
                )
                scaler.step(optim)
                scaler.update()
            except torch.OutOfMemoryError:
                oom_skipped += 1
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom_skipped in (1, 10, 100):
                    print(f"[warn ] skipped {oom_skipped} batch(es) after OOM", flush=True)
                continue
            sched.step()
            step += 1
            running += loss.detach().item()

            if step % log_every == 0:
                avg = running / log_every
                print(f"[train] epoch {epoch+1} step {step}/{total_steps} "
                      f"loss {avg:.4f}", flush=True)
                if run:
                    run.log({"train/loss": avg,
                             "train/lr": sched.get_last_lr()[0]}, step=step)
                running = 0.0

            if step % eval_every == 0 or step == total_steps:
                res, samples = evaluate(
                    model, val_loader, val_rows, vocab, device,
                    amp=amp, amp_dtype=amp_dtype,
                    max_batches=cfg.get("eval_max_batches", 8),
                )
                print(
                    f"[eval ] step {step} sent-exact {res['sentence_exact_match']:.4f} "
                    f"UMR {res['unnecessary_modification_rate']:.4f} "
                    f"conv-acc {res['conversion_accuracy']:.4f}",
                    flush=True,
                )
                for s, p, g in samples[:4]:
                    print(f"        {'OK ' if p.strip()==g.strip() else 'BAD'} {p[:90]!r}")
                    if p.strip() != g.strip():
                        print(f"            gold {g[:90]!r}")
                if run:
                    run.log({f"val/{k}": v for k, v in res.items()}, step=step)

                # Selection balances doing the job against damaging untouched
                # text: a model that rewrites everything can score well on
                # conversion alone.
                score = res["sentence_exact_match"] - cfg.get(
                    "umr_penalty", 1.0
                ) * res["unnecessary_modification_rate"]
                if score > best:
                    best = score
                    save_rewriter(model, mcfg, vocab, out_dir, res)
                    print(f"[ckpt ] new best score {score:.4f}")

    print(f"[done ] {time.time()-t0:.0f}s  best {best:.4f}")
    if run:
        run.summary.update({"best_score": best})
        run.finish()
    return {"best_score": best}


def save_rewriter(
    model: DualConditionedRewriter,
    cfg: DualSeq2SeqConfig,
    vocab: WordVocab,
    out_dir: Path,
    metrics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "rewriter.pt")
    (out_dir / "rewriter_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "rewriter_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    vocab.save(out_dir / "rewriter_vocab.json")
