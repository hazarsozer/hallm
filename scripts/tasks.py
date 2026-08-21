"""Task and update discovery for collaborators.

Answers one question — "is there anything for me?" — from two sources:

  1. open GitHub issues assigned to you (tasks)
  2. commits on origin/main your checkout has not seen (updates)

DISCOVERY IS READ-ONLY AND SAFE TO RUN AT ANY TIME, INCLUDING MID-TRAINING.
It uses `git fetch`, which only updates remote-tracking refs — it cannot modify your working
tree, your branch, or a run in progress. It never runs `git pull` or `git merge`; those would
change files under a training job and can split a pair across code versions.

There is deliberately no "what I've already seen" state file, because such a file goes stale and
then lies. Instead the repository itself is the watermark:
  - `HEAD..origin/main` is exactly "commits that are new to me"
  - "open and assigned to me" is exactly "tasks I still owe"
Closing an issue or syncing your branch clears the corresponding item with nothing to maintain.

Usage:
    uv run python scripts/tasks.py check            # anything for me?
    uv run python scripts/tasks.py show T-001       # full detail on one task
    uv run python scripts/tasks.py report T-001     # validate deliverables, then post
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TASK_LABEL = "task"
# Paths where a change alters how a collaborator must work. Everything else is our noise,
# not theirs — surfacing every commit would train them to ignore the report.
AFFECTS_COLLABORATOR = (
    "configs/",       # run configs and queues they launch from
    "scripts/",       # the commands they type
    "src/hallm/",     # harness behaviour, manifest and result formats
    "COLLABORATOR.md",  # their standing rules (CLAUDE.md is gitignored/personal)
    "README.md",      # setup and layout
    "docs/superpowers/specs/",  # protocol and program decisions
)


def sh(*args: str, check: bool = True) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def gh_json(*args: str):
    return json.loads(sh("gh", *args) or "[]")


def me() -> str:
    return gh_json("api", "user", "--jq", "{login:.login}")["login"]


def parse_task_block(body: str) -> dict | None:
    """Extract the ```task fenced YAML block from an issue body."""
    m = re.search(r"```task\s*\n(.*?)```", body or "", re.S)
    if not m:
        return None
    import yaml

    try:
        return yaml.safe_load(m.group(1))
    except Exception:
        return None


def my_tasks(login: str) -> list[dict]:
    issues = gh_json("issue", "list", "--state", "open", "--assignee", login,
                     "--json", "number,title,body,labels", "--limit", "50")
    out = []
    for i in issues:
        spec = parse_task_block(i.get("body", "")) or {}
        out.append({"number": i["number"], "title": i["title"], "spec": spec,
                    "body": i.get("body", "")})
    return out


def new_commits() -> list[dict]:
    """Commits on origin/main not in HEAD. Fetch only — never modifies the working tree."""
    sh("git", "fetch", "--quiet", "origin", check=False)
    log = sh("git", "log", "--no-merges", "--format=%h%x1f%s", "HEAD..origin/main", check=False)
    commits = []
    for line in log.splitlines():
        if not line.strip():
            continue
        h, subject = line.split("\x1f", 1)
        files = sh("git", "show", "--name-only", "--format=", h, check=False).splitlines()
        relevant = sorted({f for f in files if f.startswith(AFFECTS_COLLABORATOR)})
        commits.append({"sha": h, "subject": subject, "files": relevant})
    return commits


RULES_FILE = "COLLABORATOR.md"
POINTER = """# HaLLM — collaborator instructions (personal, gitignored)

You are a collaborator on this repository, not its owner. This is a controlled scientific
experiment: silent deviations destroy results, and the damage is usually invisible until
someone tries to publish the number.

## Never (these are the hard ones — inlined so they load unconditionally)

- Never push to `main`. Work on a branch and open a PR.
- Never force-push, hard-reset pushed work, or rebase a shared branch.
- Never edit anything in `configs/`. Config drift invalidates a pair.
- Never edit `results/reports/*.md` — they are generated from `results/runs/*.json`.
- Never start a training run unless told to in that session. Not to "verify", not to "test".
- HuggingFace: only ADD files under `checkpoints/<run-id>/`. Never delete, move, rename or
  overwrite an existing path; never change repo settings or org membership.
- If anything is unclear or reality does not match instructions: STOP and report in plain
  English. Do not improvise a workaround.

## Tasks and updates

When asked whether there is anything to do — or at the start of a session — run:

    uv run python scripts/tasks.py check

Read-only and safe at any time, including mid-training: it uses `git fetch`, never `git pull`.
Discovery never executes. Explain the task to the human in plain language and ask before
starting anything.

## Full rules

`COLLABORATOR.md` in the repo root is the maintained, versioned source. Read it at the start
of a session. The rules above are the subset that must apply even if you never open it.
"""


def rules_banner() -> None:
    """Print the constraints with every report.

    The rules must reach the agent even from a bare clone whose personal CLAUDE.md was
    never written or was lost to a re-clone — so they travel with the thing that dispenses
    work rather than depending on a file `git clone` can silently drop.
    """
    print("-" * 62)
    print(f"Rules: {RULES_FILE} (read it before acting). In short:")
    print("  never push to main · never edit configs/ · never start a run unasked")
    print("  HuggingFace: add only, never delete/rename/overwrite")
    print("  unclear? stop and report in plain English rather than improvising")
    print("-" * 62 + "\n")


def ensure_pointer() -> None:
    """Offer to restore the personal CLAUDE.md pointer if it is missing."""
    root = Path(__file__).resolve().parents[1]
    local = root / "CLAUDE.md"
    if local.exists() or not (root / RULES_FILE).exists():
        return
    print(f"NOTE: no local CLAUDE.md here, so your agent will not auto-load {RULES_FILE}.")
    print(f"      Write the one-line pointer with:  python scripts/tasks.py bootstrap\n")


def cmd_bootstrap(args) -> int:
    root = Path(__file__).resolve().parents[1]
    local = root / "CLAUDE.md"
    if local.exists():
        print(f"{local} already exists — leaving it alone.")
        return 0
    local.write_text(POINTER, encoding="utf-8")
    print(f"wrote {local} (gitignored, personal to this machine)")
    return 0


def cmd_check(args) -> int:
    rules_banner()
    ensure_pointer()
    login = me()
    tasks = my_tasks(login)
    commits = new_commits()
    affecting = [c for c in commits if c["files"]]

    if not tasks and not commits:
        print("Nothing for you. Your checkout is current and no tasks are assigned.")
        return 0

    print(f"{len(tasks)} task(s) assigned to you; "
          f"{len(commits)} commit(s) on main you haven't got.\n")

    if commits:
        print("=" * 62)
        print(f"UPDATE — main is {len(commits)} commit(s) ahead of your checkout")
        if affecting:
            touched = sorted({f.split('/')[0] + '/' if '/' in f else f
                              for c in affecting for f in c["files"]})
            print(f"  Areas that affect how you work: {', '.join(touched)}")
            for c in affecting[:6]:
                print(f"    {c['sha']}  {c['subject'][:66]}")
            if len(affecting) > 6:
                print(f"    ... and {len(affecting)-6} more")
            print("\n  Before your next run, re-branch from the new main. Do NOT pull on top of")
            print("  a branch with a run in progress — that can split a pair across code versions.")
        else:
            print("  None of it touches configs, scripts, harness code or your rules.")
            print("  Safe to keep working; sync whenever convenient.")
        print()

    for t in tasks:
        s = t["spec"]
        print("=" * 62)
        tid = s.get("task_id", f"#{t['number']}")
        print(f"TASK {tid} — {t['title']}  (issue #{t['number']})")
        if s.get("why"):
            print(f"\n  Why: {s['why']}")
        if s.get("what"):
            print(f"  What: {s['what']}")
        runs = s.get("runs") or []
        if runs:
            hrs = sum(r.get("est_hours", 0) for r in runs)
            print(f"  Work: {len(runs)} training run(s)"
                  + (f", ~{hrs}h total on your card" if hrs else ""))
            for r in runs:
                print(f"    - {r['id']}")
            print("  Interruptible: yes — checkpoints every 1000 steps, resumes exactly.")
        if s.get("baseline_for_tax"):
            print(f"  These numbers are compared against: {s['baseline_for_tax']}")
        if s.get("deliverables"):
            print(f"  Deliver: {', '.join(s['deliverables'])}")
        if s.get("branch"):
            print(f"  Branch: {s['branch']}")
        print()

    print("=" * 62)
    print("Nothing has been started. Say the word and I'll begin;")
    print("run `scripts/tasks.py show <task-id>` for the full brief first.")
    return 0


def cmd_show(args) -> int:
    for t in my_tasks(me()):
        s = t["spec"]
        if s.get("task_id") == args.task_id or f"#{t['number']}" == args.task_id:
            print(t["body"])
            return 0
    print(f"No open task {args.task_id} assigned to you.")
    return 1


def cmd_report(args) -> int:
    """Validate the deliverables exist and conform BEFORE claiming the task is done."""
    task = next((t for t in my_tasks(me())
                 if t["spec"].get("task_id") == args.task_id), None)
    if not task:
        print(f"No open task {args.task_id} assigned to you.")
        return 1

    runs = [r["id"] for r in (task["spec"].get("runs") or [])]
    problems, lines = [], []
    for rid in runs:
        res = Path("results/runs") / f"{rid}.json"
        man = Path("results/manifests") / f"{rid}.json"
        if not res.exists():
            problems.append(f"{rid}: missing {res}")
            continue
        if not man.exists():
            problems.append(f"{rid}: missing {man}")
            continue
        row = json.loads(res.read_text())
        lines.append(f"- `{rid}` — test_ppl **{row.get('test_ppl')}**")
        # Protocol conformance: an ablation is only evidence if the recipe did not drift.
        ref = task["spec"].get("protocol_reference")
        if ref and Path(f"results/manifests/{ref}.json").exists():
            a = json.loads(man.read_text())
            b = json.loads(Path(f"results/manifests/{ref}.json").read_text())
            drift = {k: (v, b["train_cfg"].get(k)) for k, v in a["train_cfg"].items()
                     if k not in ("out_dir", "seed") and b["train_cfg"].get(k) != v}
            if drift:
                problems.append(f"{rid}: train_cfg drifted from {ref}: {drift}")
            if a.get("data_sha256") != b.get("data_sha256"):
                problems.append(f"{rid}: data hashes differ from {ref} — wrong corpus")

    if problems:
        print("NOT reporting — validation failed:\n")
        for p in problems:
            print("  ", p)
        print("\nFix these before reporting. A run that drifted from protocol is not evidence.")
        return 1

    body = (f"## {args.task_id} complete\n\n" + "\n".join(lines) +
            "\n\nValidated locally before posting: every declared run has a result and a manifest, "
            "`train_cfg` shows no drift from the protocol reference, and data hashes match.\n")
    if args.dry_run:
        print(body)
        return 0
    sh("gh", "issue", "comment", str(task["number"]), "--body", body)
    print(f"reported on issue #{task['number']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="anything assigned or changed for me?")
    sub.add_parser("bootstrap", help="write the personal CLAUDE.md pointer (gitignored)")
    p_show = sub.add_parser("show", help="full brief for one task")
    p_show.add_argument("task_id")
    p_rep = sub.add_parser("report", help="validate deliverables, then post to the issue")
    p_rep.add_argument("task_id")
    p_rep.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return {"check": cmd_check, "show": cmd_show, "report": cmd_report,
            "bootstrap": cmd_bootstrap}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
