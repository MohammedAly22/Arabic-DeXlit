#!/usr/bin/env python
"""Measure where training time actually goes, on the GPU you are training on.

Run this before tuning anything. It answers four questions with measurements
rather than guesses:

1. **How large a batch actually fits?** Binary-searches real forward+backward
   passes until it OOMs, so the answer accounts for activations, not just
   parameters.
2. **What does gradient checkpointing cost here?** It trades roughly 30-40%
   throughput for memory. On a card with headroom that trade is pure loss.
3. **Is the dataloader starving the GPU?** Times batch production alone against
   a training step; if loading is the larger number, more workers beat a bigger
   batch.
4. **What do sequences actually need?** Padding every batch to a fixed 512 bytes
   when the median is ~111 wastes attention compute quadratically.

Usage
-----
    python scripts/diagnose_gpu.py                       # full sweep
    python scripts/diagnose_gpu.py --model google/byt5-small
    python scripts/diagnose_gpu.py --quick               # skip the batch search
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


def human(n: float) -> str:
    return f"{n/1e9:.1f}GB"


def gpu_report() -> dict:
    if not torch.cuda.is_available():
        print("No CUDA device. This script measures GPU behaviour.")
        raise SystemExit(1)
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    info = {
        "name": p.name,
        "total_memory": p.total_memory,
        "capability": f"{cap[0]}.{cap[1]}",
        "bf16": cap[0] >= 8,
        "sm_count": p.multi_processor_count,
    }
    print("=" * 66)
    print(f"GPU        : {info['name']}")
    print(f"VRAM       : {human(info['total_memory'])}")
    print(f"Capability : {info['capability']}  (bf16 {'yes' if info['bf16'] else 'no -- fp16 + scaler'})")
    print(f"SMs        : {info['sm_count']}")
    print(f"torch      : {torch.__version__}")
    return info


def measure_lengths(data_dir: Path, prefix_len: int, limit: int = 120_000) -> dict:
    """What the data actually needs, versus what the config pads to."""
    path = data_dir / "train.jsonl"
    if not path.exists():
        print(f"\n[lengths] {path} not found -- skipping")
        return {}
    src, tgt = [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            src.append(len(r["src"].encode("utf-8")) + prefix_len)
            tgt.append(len(r["tgt"].encode("utf-8")))
    src.sort()
    tgt.sort()
    n = len(src)
    out = {
        "n": n,
        "src_median": src[n // 2], "src_p95": src[int(0.95 * n)], "src_p99": src[int(0.99 * n)],
        "tgt_median": tgt[n // 2], "tgt_p95": tgt[int(0.95 * n)], "tgt_p99": tgt[int(0.99 * n)],
        "src_max": src[-1], "tgt_max": tgt[-1],
    }
    print("\n" + "=" * 66)
    print("SEQUENCE LENGTHS (bytes, including task prefix)")
    print(f"  source : median {out['src_median']:4d}  p95 {out['src_p95']:4d}  "
          f"p99 {out['src_p99']:4d}  max {out['src_max']}")
    print(f"  target : median {out['tgt_median']:4d}  p95 {out['tgt_p95']:4d}  "
          f"p99 {out['tgt_p99']:4d}  max {out['tgt_max']}")
    print(f"  -> a budget of {out['src_p99']} covers 99% of sentences")
    return out


def _batch(tok, model, bs: int, length: int, device):
    """A synthetic batch of the given shape, on device."""
    ids = torch.randint(3, 250, (bs, length), device=device)
    mask = torch.ones_like(ids)
    labels = torch.randint(3, 250, (bs, max(8, length // 2)), device=device)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def try_batch(model, tok, bs: int, length: int, device, amp_dtype, checkpointing: bool) -> bool:
    """One real forward+backward. True if it fits."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # train() matters: HuggingFace skips gradient checkpointing entirely
        # when the module is in eval mode, so without this both arms of the
        # comparison silently measure the same thing.
        model.train()
        model.gradient_checkpointing_enable() if checkpointing else model.gradient_checkpointing_disable()
        model.config.use_cache = not checkpointing
        batch = _batch(tok, model, bs, length, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = model(**batch).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        return True
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False


def find_max_batch(model, tok, length: int, device, amp_dtype, checkpointing: bool) -> int:
    """Binary-search the largest batch that survives a real step."""
    lo, hi = 1, 4096
    # Grow until failure, so the search starts from a true upper bound.
    probe = 8
    while probe <= hi and try_batch(model, tok, probe, length, device, amp_dtype, checkpointing):
        lo = probe
        probe *= 2
    hi = min(hi, probe)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if try_batch(model, tok, mid, length, device, amp_dtype, checkpointing):
            lo = mid
        else:
            hi = mid
    return lo


def time_steps(model, tok, bs: int, length: int, device, amp_dtype, checkpointing: bool,
               iters: int = 12) -> tuple[float, float]:
    """Median seconds per step and peak memory, for a given configuration."""
    model.train()
    model.gradient_checkpointing_enable() if checkpointing else model.gradient_checkpointing_disable()
    model.config.use_cache = not checkpointing
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    batch = _batch(tok, model, bs, length, device)

    for _ in range(3):  # warm up kernels and the allocator
        with torch.autocast("cuda", dtype=amp_dtype):
            model(**batch).loss.backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=amp_dtype):
            model(**batch).loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], torch.cuda.max_memory_allocated()


def time_dataloader(data_dir: Path, tok, cfg, batch_size: int, workers: int) -> float:
    """Seconds to produce one batch. If this exceeds step time, the GPU starves."""
    from torch.utils.data import DataLoader

    from arabic_dexlit.training.dataset import read_jsonl
    from arabic_dexlit.training.train_seq2seq import Seq2SeqCollator, Seq2SeqDataset

    path = data_dir / "train.jsonl"
    if not path.exists():
        return float("nan")
    rows = read_jsonl(path)[:20000]
    loader = DataLoader(
        Seq2SeqDataset(rows, tok, cfg), batch_size=batch_size, shuffle=True,
        collate_fn=Seq2SeqCollator(tok), num_workers=workers,
        pin_memory=True, drop_last=True,
        persistent_workers=workers > 0,
    )
    it = iter(loader)
    for _ in range(3):
        next(it)
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        next(it)
    return (time.perf_counter() - t0) / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/byt5-base")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--quick", action="store_true", help="skip the batch-size search")
    p.add_argument("--out", default="gpu_diagnosis.json")
    args = p.parse_args()

    info = gpu_report()
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if info["bf16"] else torch.float16
    # Match the trainer, otherwise the measured numbers do not transfer.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from arabic_dexlit.model.seq2seq import Seq2SeqConfig
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = Seq2SeqConfig(model_name=args.model)
    lengths = measure_lengths(Path(args.data_dir), len(cfg.task_prefix))
    # Tune at the length the data actually needs, not the configured ceiling.
    test_len = lengths.get("src_p99", 512) if lengths else 512
    test_len = int(((test_len + 63) // 64) * 64)

    print("\n" + "=" * 66)
    print(f"LOADING {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M parameters, testing at length {test_len}")

    results = {"gpu": info, "model": args.model, "lengths": lengths,
               "test_length": test_len, "configs": []}

    print("\n" + "=" * 66)
    print("THROUGHPUT  (samples/sec is the number that matters)")
    print(f"{'checkpointing':>14}{'batch':>8}{'sec/step':>10}{'samples/s':>12}{'peak mem':>11}")

    for ckpt in (True, False):
        if args.quick:
            candidates = [16, 64] if ckpt else [16, 64]
        else:
            mx = find_max_batch(model, tok, test_len, device, amp_dtype, ckpt)
            # Back off slightly from the true maximum: the real corpus has
            # longer outliers than the synthetic probe.
            candidates = sorted({max(1, mx // 4), max(1, mx // 2), int(mx * 0.8)})
            print(f"  [max batch with checkpointing={ckpt}: {mx}]")

        for bs in candidates:
            try:
                sec, mem = time_steps(model, tok, bs, test_len, device, amp_dtype, ckpt)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            sps = bs / sec
            results["configs"].append(
                {"checkpointing": ckpt, "batch_size": bs, "sec_per_step": sec,
                 "samples_per_sec": sps, "peak_memory": mem}
            )
            print(f"{str(ckpt):>14}{bs:>8}{sec:>10.3f}{sps:>12.1f}{human(mem):>11}")

    # --- dataloader ---
    print("\n" + "=" * 66)
    print("DATALOADER  (if load time > step time, the GPU is starving)")
    best = max(results["configs"], key=lambda c: c["samples_per_sec"]) if results["configs"] else None
    bs = best["batch_size"] if best else 32
    print(f"{'workers':>9}{'sec/batch':>12}{'verdict':>28}")
    dl = {}
    for w in (2, 4, 8, 16):
        try:
            t = time_dataloader(Path(args.data_dir), tok, cfg, bs, w)
        except Exception as e:
            print(f"{w:>9}  failed: {type(e).__name__}")
            continue
        dl[w] = t
        verdict = ""
        if best and t > best["sec_per_step"]:
            verdict = "STARVING the GPU"
        elif best and t > best["sec_per_step"] * 0.5:
            verdict = "marginal"
        else:
            verdict = "ok"
        print(f"{w:>9}{t:>12.4f}{verdict:>28}")
    results["dataloader"] = dl

    # --- recommendation ---
    print("\n" + "=" * 66)
    print("RECOMMENDED CONFIG")
    if best:
        speedup = best["samples_per_sec"] / min(
            c["samples_per_sec"] for c in results["configs"]
        )
        best_workers = min(dl, key=dl.get) if dl else 4
        # Keep the effective batch near 256; large-batch training needs the
        # learning rate scaled with it.
        accum = max(1, round(256 / best["batch_size"]))
        print(f"  batch_size            : {best['batch_size']}")
        print(f"  grad_accum            : {accum}   (effective {best['batch_size']*accum})")
        print(f"  gradient_checkpointing: {str(best['checkpointing']).lower()}")
        print(f"  num_workers           : {best_workers}")
        print(f"  max_source_length     : {test_len}")
        print(f"  peak memory           : {human(best['peak_memory'])} of {human(info['total_memory'])}")
        print(f"  throughput            : {best['samples_per_sec']:.1f} samples/s "
              f"({speedup:.1f}x the slowest configuration tested)")
        results["recommended"] = {
            "batch_size": best["batch_size"], "grad_accum": accum,
            "gradient_checkpointing": best["checkpointing"],
            "num_workers": best_workers, "max_source_length": test_len,
        }

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
