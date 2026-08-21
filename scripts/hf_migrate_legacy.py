"""Fold the legacy HuggingFace layout into the run-addressed `checkpoints/<run-id>/` scheme.

The Experiment 1-2 checkpoints exist ONLY on the Hub — there is no local copy anywhere — so this
script is deliberately paranoid:

  1. COPY  server-side (CommitOperationCopy, no download/upload round trip, no re-hashing risk)
  2. VERIFY every destination exists with an LFS sha256 identical to its source
  3. DELETE the source only for paths that passed step 2

Nothing is deleted in the same commit as its copy, and `--apply` is required for any write at all.
A path whose verification fails is left alone and reported.

Mapping (artifact-layout spec 2026-08-19): the legacy dirs are exactly the unqueued seed-1337
slot-ins of the ladder, so the mapping is total and collision-free.

Usage:
    uv run --with huggingface_hub python scripts/hf_migrate_legacy.py            # dry run
    uv run --with huggingface_hub python scripts/hf_migrate_legacy.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ID = "hallm-thesis/hallm-wikitext103"

# legacy dir -> (run id, legacy config filename)
LEGACY = {
    "A0": ("L8-A0-s1337", "arm0_none.yaml"),
    "A1": ("L8-A1-s1337", "arm1_albert.yaml"),
    "A2": ("L8-A2-s1337", "arm2_halvit.yaml"),
    "A3": ("L8-A3-s1337", "arm3_both.yaml"),
    "A0-deep": ("L16-A0-s1337", "arm0_deep.yaml"),
    "A2-iso": ("L16-A2-s1337", "arm2_iso.yaml"),
}

# Already-migrated duplicates: old path -> run id whose checkpoints/ copy should match it.
DUPES = {
    "runs/ladder/L4-A0-s1337/L4-A0-s1337.pt": "L4-A0-s1337",
    "runs/ladder/L4-A0-s1338/L4-A0-s1338.pt": "L4-A0-s1338",
    "runs/ladder/L4-A2-s1337/L4-A2-s1337.pt": "L4-A2-s1337",
}

# Verified identical to the bins every manifest already records.
DATA_SHA = {
    "train.bin": "c36727c10ace9aaca6e13786863fb9fbb5dfa354d93d604e2cc415d85bada12a",
    "val.bin": "ba1fa893453dd6e475c337fb524923709f08be0548dbe27a1fce438b65eab9d8",
}


def oid_of(sib) -> str | None:
    if not sib.lfs:
        return None
    if isinstance(sib.lfs, dict):
        return sib.lfs.get("sha256")
    return getattr(sib.lfs, "sha256", None)


def snapshot(api, repo: str) -> dict[str, tuple[int | None, str | None]]:
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: (s.size, oid_of(s)) for s in info.siblings}


def retro_manifest(run_id: str) -> dict:
    """Minimal manifest for a pre-manifest-era run, honestly marked."""
    from dataclasses import asdict

    from hallm.experiment import load_experiment

    model_cfg, train_cfg = load_experiment(Path("configs/runs") / f"{run_id}.yaml")
    return {
        "retroactive": True,
        "note": (
            "Backfilled after the fact. This run predates the manifest harness, so environment "
            "fields were never captured and are recorded as unknown rather than guessed. "
            "model_cfg/train_cfg are reconstructed from the committed config for this run ID, "
            "which is protocol-identical to what was trained."
        ),
        "config_path": f"configs/runs/{run_id}.yaml",
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "data_sha256": dict(DATA_SHA),
        "tokenizer": "gpt2",
        "git_commit": "unknown",
        "python": "unknown",
        "torch": "unknown",
        "platform": "unknown",
        "gpu": "NVIDIA GeForce RTX 4070 SUPER",
        "created_utc": "unknown",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--apply", action="store_true", help="perform writes (default: dry run)")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

    api = HfApi()
    before = snapshot(api, args.repo)
    print(f"repo {args.repo}: {len(before)} files\n")

    # ---- plan ----------------------------------------------------------------
    copies, cfg_moves, manifests = [], [], []
    for legacy, (run_id, cfg_name) in LEGACY.items():
        src_pt, dst_pt = f"{legacy}/model.pt", f"checkpoints/{run_id}/model.pt"
        if src_pt not in before:
            print(f"  skip {legacy}: {src_pt} not present")
            continue
        if dst_pt in before:
            print(f"  skip {legacy}: {dst_pt} already exists")
            continue
        copies.append((src_pt, dst_pt))
        cfg_moves.append((f"{legacy}/{cfg_name}", f"checkpoints/{run_id}/config.yaml"))
        manifests.append(run_id)

    print(f"\nplanned: {len(copies)} checkpoint copies, {len(cfg_moves)} configs, "
          f"{len(manifests)} retroactive manifests")
    for s, d in copies:
        print(f"   {s:<24} -> {d}")

    print("\nverified duplicates eligible for cleanup:")
    dupe_ok = []
    for old, run_id in DUPES.items():
        new = f"checkpoints/{run_id}/model.pt"
        o_old, o_new = before.get(old, (None, None))[1], before.get(new, (None, None))[1]
        if o_old and o_new and o_old == o_new:
            dupe_ok.append(old)
            print(f"   {old}  ==  {new}  ({o_old[:12]}...) OK")
        else:
            print(f"   {old}: NOT verified (old={o_old}, new={o_new}) — leaving alone")

    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply.")
        return 0

    # ---- step 1: copy checkpoints server-side --------------------------------
    if copies:
        api.create_commit(
            repo_id=args.repo, repo_type="model",
            operations=[CommitOperationCopy(src_path_in_repo=s, path_in_repo=d) for s, d in copies],
            commit_message="Migrate legacy checkpoints into checkpoints/<run-id>/ (copy)",
        )
        print(f"\ncopied {len(copies)} checkpoints")

    # ---- step 2: configs + retroactive manifests ------------------------------
    for src_cfg, dst_cfg in cfg_moves:
        local = api.hf_hub_download(repo_id=args.repo, filename=src_cfg, repo_type="model")
        api.upload_file(path_or_fileobj=local, path_in_repo=dst_cfg,
                        repo_id=args.repo, repo_type="model")
        print(f"   config {src_cfg} -> {dst_cfg}")
    for run_id in manifests:
        payload = json.dumps(retro_manifest(run_id), indent=2, sort_keys=True).encode()
        api.upload_file(path_or_fileobj=payload,
                        path_in_repo=f"checkpoints/{run_id}/manifest.json",
                        repo_id=args.repo, repo_type="model")
        print(f"   manifest checkpoints/{run_id}/manifest.json (retroactive)")

    # ---- step 3: verify before deleting anything ------------------------------
    after = snapshot(api, args.repo)
    deletable, failed = list(dupe_ok), []
    for src, dst in copies:
        o_src = before[src][1]
        o_dst = after.get(dst, (None, None))[1]
        if o_dst and o_src == o_dst:
            deletable.append(src)
        else:
            failed.append((src, dst, o_src, o_dst))
    for src_cfg, dst_cfg in cfg_moves:
        if dst_cfg in after:
            deletable.append(src_cfg)
        else:
            failed.append((src_cfg, dst_cfg, "reg", None))

    if failed:
        print("\n!! verification FAILED — these sources will NOT be deleted:")
        for src, dst, a, b in failed:
            print(f"   {src} -> {dst}  (src={a}, dst={b})")

    # ---- step 4: delete only verified sources --------------------------------
    if deletable:
        api.create_commit(
            repo_id=args.repo, repo_type="model",
            operations=[CommitOperationDelete(path_in_repo=p) for p in deletable],
            commit_message="Remove legacy paths after hash-verified migration",
        )
        print(f"\ndeleted {len(deletable)} verified-migrated source paths")

    final = snapshot(api, args.repo)
    print(f"\nfinal: {len(final)} files")
    stragglers = [p for p in final if not p.startswith(("checkpoints/", "data/")) and p not in (".gitattributes", "README.md")]
    print("non-canonical paths remaining:", stragglers or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
