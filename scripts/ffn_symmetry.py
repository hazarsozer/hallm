"""FFN Jacobian symmetry over trained checkpoints (spec 2026-09-15 §5, extended 2026-09-16 §5).
Inference-only.

LM checkpoints: inputs are `--windows` random WikiText-103 validation windows. DeiT-small (--deit,
kind=vision) is an uncontrolled reference — a model trained by other people, on other data, at other
scale — fetched once:
    curl -L -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar -xzf data/imagenette2-160.tgz -C data/
`--hallm-vit` (kind=vision-ours) is the controlled comparison: ViTs trained with our own recipe over
the same sharing code as the LM ladder, inputs from `--vit-data`'s held-out ImageNet-100 split.
Needs the analysis group:  uv run --group analysis python scripts/ffn_symmetry.py ...

Usage:
  uv run --group analysis python scripts/ffn_symmetry.py \
      --checkpoints 'runs/ladder/L*-A0-s13??/*-s13??.pt' 'runs/ladder/L8-A2-s1337/*.pt' \
      --data data --deit --images data/imagenette2-160/val \
      --hallm-vit 'runs/vision/v8-*/*.pt' --vit-data data/in100
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from hallm.data import load_bin
from hallm.symmetry import hallm_rotation_shares, rotation_share
from hallm.train import build_model_from_checkpoint


def _row(model: str, kind: str, arm: str, per_layer: list[float], per_layer_xhat: list[float]) -> dict:
    """`per_layer`/`mean` are in the u = LN2(x) basis; `per_layer_xhat`/`mean_xhat` in the pre-gain
    x̂ basis (spec 2026-09-15 §5 — the share is basis-dependent). Kept at full precision so the
    L8-A2 sanity check ("0 to float precision") stays falsifiable; round only for display."""
    return {"model": model, "kind": kind, "arm": arm,
            "per_layer": [float(s) for s in per_layer], "mean": float(np.mean(per_layer)),
            "per_layer_xhat": [float(s) for s in per_layer_xhat], "mean_xhat": float(np.mean(per_layer_xhat))}


def lm_rows(patterns: list[str], data_dir: str, n_positions: int, windows: int, device: str) -> list[dict]:
    val = load_bin(Path(data_dir) / "val.bin")
    rows = []
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        if Path(path).stem == "resume":
            continue
        model, cfg = build_model_from_checkpoint(path, map_location=device)
        model.to(device).eval()
        T = cfg.block_size
        starts = np.random.default_rng(0).integers(0, len(val) - T, size=windows)
        idx = torch.from_numpy(np.stack([np.asarray(val[s:s + T], dtype=np.int64) for s in starts])).to(device)
        shares_u = hallm_rotation_shares(model, idx, n_positions, basis="u")
        shares_xhat = hallm_rotation_shares(model, idx, n_positions, basis="xhat")
        rows.append(_row(Path(path).stem, "lm", cfg.arm, shares_u, shares_xhat))
        print(rows[-1]["model"], rows[-1]["mean"])
    return rows


def deit_row(images_dir: str, n_images: int, n_positions: int, device: str,
            model_name: str = "facebook/deit-small-patch16-224") -> dict:
    """`model_name` defaults to the real DeiT-small checkpoint but can be pointed at a local
    directory (a `save_pretrained` model + a DeiT-style `preprocessor_config.json`) for offline
    testing. Uses `DeiTImageProcessorPil` rather than `AutoImageProcessor`: as of transformers
    5.17.0 the Auto class needs torchvision, which this project does not depend on (it pins
    torch and could move the training lock) — the Pil backend needs only Pillow, already a dep."""
    from PIL import Image
    from transformers import AutoModel, DeiTImageProcessorPil

    from hallm.symmetry import vit_rotation_shares

    proc = DeiTImageProcessorPil.from_pretrained(model_name)
    vit = AutoModel.from_pretrained(model_name).to(device).eval()
    files = sorted(Path(images_dir).rglob("*.JPEG"))
    pick = np.random.default_rng(0).choice(len(files), size=min(n_images, len(files)), replace=False)
    images = [Image.open(files[i]).convert("RGB") for i in sorted(pick)]
    pixels = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
    shares_u = vit_rotation_shares(vit, pixels, n_positions, basis="u")
    shares_xhat = vit_rotation_shares(vit, pixels, n_positions, basis="xhat")
    return _row(Path(model_name).name, "vision", "A0", shares_u, shares_xhat)


def hallm_vit_rows(patterns: list[str], data_dir: str, n_positions: int, n_images: int,
                   device: str) -> list[dict]:
    """Rotation shares for ViTs we trained ourselves — the controlled half of the LM/vision
    comparison (spec 2026-09-16 §5). Inputs are held-out ImageNet-100 images, center-cropped."""
    from hallm.data.imagenet import get_image_batch, load_images, load_labels
    from hallm.model.config import VisionConfig
    from hallm.model.vit import ViT

    images = load_images(Path(data_dir) / "val.bin")
    labels = load_labels(Path(data_dir) / "val_labels.npy")
    rows = []
    for path in sorted(p for pat in patterns for p in glob.glob(pat)):
        if Path(path).stem == "resume":
            continue
        ckpt = torch.load(path, map_location=device, weights_only=True)   # dict-of-primitives
        model = ViT(VisionConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        x, _ = get_image_batch(images, labels, n_images, model.cfg.image_size, device, None,
                               train=False)
        shares_u = hallm_rotation_shares(model, x, n_positions, basis="u")
        shares_xhat = hallm_rotation_shares(model, x, n_positions, basis="xhat")
        rows.append(_row(Path(path).stem, "vision-ours", model.cfg.arm, shares_u, shares_xhat))
        print(rows[-1]["model"], rows[-1]["mean"])
    return rows


def random_row(d: int = 256, h: int = 1024, n: int = 64) -> dict:
    g = torch.Generator().manual_seed(0)
    share = rotation_share(torch.randn(h, d, generator=g), torch.randn(d, h, generator=g),
                           torch.randn(n, d, generator=g))
    return _row("random", "reference", "—", [share], [share])   # in_scale=None for both bases


def _fmt(s: float) -> str:
    """3 significant figures, but scientific notation below 1e-3 so the L8-A2 sanity check
    ("0 to float precision") stays legible in the table instead of rounding to "0.000"."""
    return f"{s:.2e}" if abs(s) < 1e-3 else f"{s:.3f}"


def report(rows: list[dict]) -> str:
    L = ["<!-- GENERATED by scripts/ffn_symmetry.py — do not edit -->",
         "# FFN Jacobian rotation share (spec 2026-09-15 §5)", "",
         "‖½(J − Jᵀ)‖² / ‖J‖² of each FFN's input Jacobian at real inputs. 0 = W+Wᵀ (exact), "
         "~0.5 = random matrices, 1 = pure rotation. The share is basis-dependent: `mean` is w.r.t. "
         "u = LN2(x) (includes the LayerNorm gain γ), `mean (x̂)` w.r.t. the pre-gain normalized "
         "input (γ folded into J). `kind=vision` (DeiT-small) is an **uncontrolled reference** "
         "(other recipe, scale and data) — a model trained by other people, not evidence for or "
         "against sharing here. `kind=vision-ours` is the controlled measurement (spec "
         "2026-09-16 §5): ViTs trained with our own recipe over the same sharing code as the LM "
         "ladder (`kind=lm`), so it is directly comparable to it. Cross-model comparisons "
         "otherwise remain uncontrolled: mechanism evidence, not a test.", "",
         "| model | kind | arm | layers | mean | mean (x̂) | per layer |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        per = ", ".join(_fmt(s) for s in r["per_layer"])
        L.append(f"| {r['model']} | {r['kind']} | {r['arm']} | {len(r['per_layer'])} | "
                 f"{_fmt(r['mean'])} | {_fmt(r['mean_xhat'])} | {per} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FFN Jacobian rotation share")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--data", default="data")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--n-positions", type=int, default=256)
    ap.add_argument("--deit", action="store_true")
    ap.add_argument("--deit-model", default="facebook/deit-small-patch16-224")
    ap.add_argument("--images", default="data/imagenette2-160/val")
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--hallm-vit", nargs="*", default=[])
    ap.add_argument("--vit-data", default="data/in100")
    ap.add_argument("--n-images-ours", type=int, default=64)
    ap.add_argument("--out", default="results/analysis/ffn-symmetry.json")
    ap.add_argument("--report", default="results/reports/ffn-symmetry.md")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def write(rows: list[dict]) -> None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report(rows), encoding="utf-8")
        print(f"wrote {args.out} and {args.report}")

    # Written after each row group: a failure in a later group (e.g. a missing/renamed checkpoint,
    # or an offline run with --deit) then cannot lose the already-computed earlier rows.
    rows = lm_rows(args.checkpoints, args.data, args.n_positions, args.windows, device)
    write(rows)
    if args.deit:
        rows.append(deit_row(args.images, args.n_images, args.n_positions, device, args.deit_model))
        write(rows)
    if args.hallm_vit:
        rows += hallm_vit_rows(args.hallm_vit, args.vit_data, args.n_positions, args.n_images_ours, device)
        write(rows)
    rows.append(random_row())
    write(rows)


if __name__ == "__main__":
    main()
