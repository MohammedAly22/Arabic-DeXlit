"""Stage-2 training loop.

The converter is small (a few million parameters) and its examples are short
span pairs rather than sentences, so this trains far faster than stage 1 --
typically minutes, not hours. It is a separate script because the two stages
have genuinely different optimisation profiles: the converter is trained from
scratch with a high learning rate, while the detector fine-tunes a pretrained
encoder at a low one.

Selection is on **exact match**, since a partly-correct span is still a wrong
output string.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.converter import (
    CATEGORY_IDS,
    ConverterConfig,
    SpanConverter,
    decode_ids,
)
from ..model.lexicon import OOV_ID, Lexicon
from ..schema import CATEGORIES
from .dataset import (
    ConverterDataset,
    balanced_subset,
    collate_converter,
    read_jsonl,
)
from .metrics import ConverterMetrics
from .train_detector import pick_amp

_ID2CAT = {v: k for k, v in CATEGORY_IDS.items()}


@torch.no_grad()
def evaluate(
    model: SpanConverter,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
    lexicon: "Lexicon | None" = None,
) -> tuple[dict, list[tuple[str, str, str]]]:
    model.eval()
    m = ConverterMetrics()
    losses: list[float] = []
    samples: list[tuple[str, str, str]] = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            out = model(
                batch["src"], batch["category"], batch["tgt_in"],
                batch["tgt_out"], word_ids=batch.get("word_ids"),
            )
        losses.append(float(out["loss"]))

        gen = model.greedy_decode(batch["src"], batch["category"])
        preds = [decode_ids(r) for r in gen.cpu().tolist()]

        # Hybrid: where the word head is confident and does not predict OOV, its
        # answer is a whole dictionary word and therefore cannot be misspelled.
        # Everything else falls back to the characters, keeping the open
        # vocabulary that brand names and URLs need.
        if lexicon is not None and model.word_head is not None:
            widx, wconf = model.predict_words(batch["src"], batch["category"])
            thr = model.cfg.word_confidence
            for j, (wi, wc) in enumerate(zip(widx.cpu().tolist(), wconf.cpu().tolist())):
                if wi != OOV_ID and wc >= thr:
                    preds[j] = lexicon.word(wi)
        golds = [decode_ids(r) for r in batch["tgt_out"].cpu().tolist()]
        cats = [_ID2CAT.get(int(c), "CS") for c in batch["category"].cpu().tolist()]
        m.update(preds, golds, cats)

        if len(samples) < 12:
            srcs = [decode_ids(r) for r in batch["src"].cpu().tolist()]
            samples.extend(list(zip(srcs, preds, golds))[: 12 - len(samples)])

    res = m.compute()
    res["loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return res, samples


def train_converter(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, amp_dtype = pick_amp(device)
    torch.manual_seed(cfg.get("seed", 42))

    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ccfg = ConverterConfig(
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_encoder_layers=cfg.get("num_encoder_layers", 3),
        num_decoder_layers=cfg.get("num_decoder_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 768),
        dropout=cfg.get("dropout", 0.1),
        max_src_len=cfg.get("max_src_len", 48),
        max_tgt_len=cfg.get("max_tgt_len", 40),
    )

    # The caps are counted in *sentences*, matching stage 1, and applied before
    # spans are flattened. Without this a --smoke run looks capped at 400 but
    # actually trains on every span those sentences contain -- ~72K pairs and
    # 9,000 steps, which is not a smoke test.
    train_rows = balanced_subset(
        read_jsonl(data_dir / "train.jsonl"), cfg.get("max_train_examples")
    )
    val_rows = balanced_subset(
        read_jsonl(data_dir / "validation.jsonl"), cfg.get("max_eval_examples")
    )
    train_ds = ConverterDataset(
        train_rows, max_src_len=ccfg.max_src_len, max_tgt_len=ccfg.max_tgt_len
    )
    val_ds = ConverterDataset(
        val_rows, max_src_len=ccfg.max_src_len, max_tgt_len=ccfg.max_tgt_len
    )

    # The lexicon is built from TRAIN targets only -- building it from the whole
    # corpus would leak test vocabulary into the model's output space.
    lexicon = None
    if cfg.get("use_word_head", True):
        lexicon = Lexicon.from_spans(
            train_ds.targets(),
            max_size=cfg.get("lexicon_max_size", 20000),
            min_count=cfg.get("lexicon_min_count", 2),
        )
        ccfg.use_word_head = True
        ccfg.lexicon_size = len(lexicon)
        ccfg.word_confidence = cfg.get("word_confidence", 0.90)
        train_ds.lexicon = lexicon
        val_ds.lexicon = lexicon
        print(
            f"[lex  ] {len(lexicon):,} words | train coverage "
            f"{lexicon.coverage(train_ds.targets()):.1%} | val coverage "
            f"{lexicon.coverage(val_ds.targets()):.1%}"
        )
    else:
        ccfg.use_word_head = False
    print(
        f"[data] sentences: train={len(train_rows):,} val={len(val_rows):,} | "
        f"span pairs: train={len(train_ds):,} val={len(val_ds):,}"
    )
    if not len(train_ds):
        raise SystemExit("no span pairs found -- build the dataset first")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 256),
        shuffle=True,
        collate_fn=collate_converter,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("eval_batch_size", 256),
        shuffle=False,
        collate_fn=collate_converter,
        num_workers=cfg.get("num_workers", 2),
    )

    model = SpanConverter(ccfg).to(device)
    print(f"[model] converter {model.num_parameters()/1e6:.2f}M parameters")

    epochs = cfg.get("epochs", 8)
    total_steps = max(1, len(train_loader) * epochs)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    warmup = int(total_steps * cfg.get("warmup_ratio", 0.05))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=cfg.get("learning_rate", 3e-4),
        total_steps=total_steps,
        pct_start=max(0.01, warmup / total_steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and amp_dtype is torch.float16)

    run = None
    if cfg.get("wandb_project"):
        try:
            import wandb

            run = wandb.init(
                project=cfg["wandb_project"],
                name=cfg.get("run_name", "converter"),
                config={**cfg, **ccfg.to_dict(), "n_params": model.num_parameters()},
                reinit=True,
            )
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    best_em = -1.0
    step = 0
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 500)
    t0 = time.time()

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                out = model(
                    batch["src"], batch["category"], batch["tgt_in"],
                    batch["tgt_out"], word_ids=batch.get("word_ids"),
                )
                loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
            running += loss.detach().item()

            if step % log_every == 0:
                avg = running / log_every
                print(
                    f"[train] epoch {epoch+1} step {step}/{total_steps} loss {avg:.4f}",
                    flush=True,
                )
                if run:
                    run.log(
                        {"train/loss": avg, "train/lr": sched.get_last_lr()[0]}, step=step
                    )
                running = 0.0

            if step % eval_every == 0 or step == total_steps:
                res, samples = evaluate(
                    model, val_loader, device, amp=amp, amp_dtype=amp_dtype,
                    max_batches=cfg.get("eval_max_batches", 20), lexicon=lexicon,
                )
                print(
                    f"[eval ] step {step} exact {res['exact_match']:.4f} "
                    f"char {res['char_accuracy']:.4f}",
                    flush=True,
                )
                for s, p, g in samples[:5]:
                    mark = "OK " if p.strip().lower() == g.strip().lower() else "BAD"
                    print(f"        {mark} {s!r} -> {p!r} (gold {g!r})")
                if run:
                    import wandb

                    payload = {f"val/{k}": v for k, v in res.items()}
                    payload["val/examples"] = wandb.Table(
                        columns=["span", "prediction", "gold"],
                        data=[list(x) for x in samples],
                    )
                    run.log(payload, step=step)

                if res["exact_match"] > best_em:
                    best_em = res["exact_match"]
                    save_converter(model, ccfg, out_dir, res)
                    if lexicon is not None:
                        lexicon.save(out_dir / "lexicon.json")
                    print(f"[ckpt ] new best exact-match {best_em:.4f}")

    print(f"[done ] {time.time()-t0:.0f}s  best exact-match {best_em:.4f}")
    if run:
        run.summary.update({"best_exact_match": best_em})
        run.finish()
    return {"best_exact_match": best_em}


def save_converter(model: SpanConverter, ccfg: ConverterConfig, out_dir: Path, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "converter.pt")
    (out_dir / "converter_config.json").write_text(
        json.dumps(ccfg.to_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "converter_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
