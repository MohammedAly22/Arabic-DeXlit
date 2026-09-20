#!/usr/bin/env python
"""Train ArabicDeXlit.

One entry point for both stages, so the Colab notebook and the command line run
exactly the same code.

Examples
--------
    # stage 1, default recipe
    python scripts/train.py --stage detector --config configs/detector_base.yaml

    # stage 2 (fast -- usually minutes)
    python scripts/train.py --stage converter --config configs/converter_base.yaml

    # both, in order
    python scripts/train.py --stage both --config configs/detector_base.yaml \
        --converter-config configs/converter_base.yaml

    # a quick end-to-end check before committing a GPU session
    python scripts/train.py --stage both --smoke

Any config key may be overridden from the command line, e.g. ``--epochs 1
--batch-size 64 --wandb-project my-project``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_config(path: str | None) -> dict:
    """Read a YAML (or JSON) config. YAML is optional -- JSON always works."""
    if not path:
        return {}
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:  # pragma: no cover
            raise SystemExit("pyyaml is required for YAML configs: pip install pyyaml")
        return yaml.safe_load(text) or {}
    return json.loads(text)


# Overrides accepted on the command line, with the type to coerce to.
_OVERRIDES: dict[str, type] = {
    "data_dir": str,
    "output_dir": str,
    "encoder_name": str,
    "epochs": int,
    "batch_size": int,
    "eval_batch_size": int,
    "grad_accum": int,
    "learning_rate": float,
    "max_length": int,
    "max_train_examples": int,
    "max_eval_examples": int,
    "eval_every": int,
    "log_every": int,
    "freeze_encoder_layers": int,
    "positive_weight": float,
    "copy_gate_weight": float,
    "seed": int,
    "num_workers": int,
    "wandb_project": str,
    "run_name": str,
}

_SMOKE = {
    "epochs": 1,
    "max_train_examples": 400,
    "max_eval_examples": 200,
    "batch_size": 8,
    "eval_batch_size": 8,
    "eval_every": 20,
    "log_every": 10,
    "eval_max_batches": 4,
    "wandb_project": None,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--stage",
        choices=["detector", "converter", "rewriter", "both", "both-rewriter"],
        default="detector",
        help="'converter' routes spans; 'rewriter' rewrites the whole sentence "
             "conditioned on the detector's tags",
    )
    p.add_argument("--config", default="configs/detector_base.yaml")
    p.add_argument("--converter-config", default="configs/converter_base.yaml")
    p.add_argument("--rewriter-config", default="configs/rewriter_base.yaml")
    p.add_argument(
        "--smoke", action="store_true",
        help="tiny run on a few hundred examples, to verify the pipeline works",
    )
    p.add_argument("--no-wandb", action="store_true")
    for key, typ in _OVERRIDES.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=typ, default=None, dest=key)
    args = p.parse_args()

    def resolve(path: str) -> dict:
        cfg = load_config(path)
        for key in _OVERRIDES:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val
        if args.smoke:
            cfg.update(_SMOKE)
        if args.no_wandb:
            cfg["wandb_project"] = None
        return cfg

    results: dict = {}

    if args.stage in ("detector", "both", "both-rewriter"):
        from arabic_dexlit.training.train_detector import train_detector

        cfg = resolve(args.config)
        print("=== stage 1: span detector ===")
        print(json.dumps({k: v for k, v in cfg.items() if v is not None}, indent=2))
        results["detector"] = train_detector(cfg)

    if args.stage in ("rewriter", "both-rewriter"):
        from arabic_dexlit.training.train_rewriter import train_rewriter

        rcfg = resolve(args.rewriter_config)
        if args.stage == "both-rewriter":
            rcfg["output_dir"] = resolve(args.config)["output_dir"]
        print("\n=== stage 2: dual-conditioned rewriter ===")
        print(json.dumps({k: v for k, v in rcfg.items() if v is not None}, indent=2))
        results["rewriter"] = train_rewriter(rcfg)

    if args.stage in ("converter", "both"):
        from arabic_dexlit.training.train_converter import train_converter

        ccfg = resolve(args.converter_config)
        # Keep both stages in one directory so the pipeline loads from a single path.
        if args.stage == "both":
            dcfg = resolve(args.config)
            ccfg["output_dir"] = dcfg["output_dir"]
        print("\n=== stage 2: span converter ===")
        print(json.dumps({k: v for k, v in ccfg.items() if v is not None}, indent=2))
        results["converter"] = train_converter(ccfg)

    print("\n=== results ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
