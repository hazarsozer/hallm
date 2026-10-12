"""Data-level shortcut check for the p-hop generator: how well do answers that need no hop do?

No model and no training — it only looks at generated problems. A task measures p-hop reasoning
only if the cheap strategies below sit near chance (1 / vocab_size). Run it before trusting any
accuracy on a p-hop config.

    uv run python scripts/synth_shortcut_check.py --p 4 --vocab 256 --margin 4
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from hallm.synth.tasks import PHopInductionTask


def hop_path(tokens: list[int], p: int) -> list[int]:
    """Positions visited: start at the last token, then p times the token after the previous
    occurrence of the current one."""
    pos = len(tokens) - 1
    path = [pos]
    for _ in range(p):
        v = tokens[pos]
        pos = next(j for j in range(pos - 1, -1, -1) if tokens[j] == v) + 1
        path.append(pos)
    return path


def shortcut_baselines(task: PHopInductionTask, p: int, n: int, seed: int = 123) -> dict[str, float]:
    rng = random.Random(seed)
    answer_pos: Counter[int] = Counter()
    hits: Counter[str] = Counter()
    for _ in range(n):
        problem = task.generate(rng, p)
        tokens, answer = problem.tokens[: problem.answer_start], problem.answer[0]
        path = hop_path(tokens, p)
        answer_pos[path[-1]] += 1
        for k in range(p):  # stop early after k hops instead of p
            hits[f"stop_after_{k}_hops"] += tokens[path[k]] == answer
    best_pos, best_count = answer_pos.most_common(1)[0]
    out = {name: count / n for name, count in sorted(hits.items())}
    out[f"copy_fixed_position_{best_pos}"] = best_count / n
    out["chance"] = 1 / task.vocab_size
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    task = PHopInductionTask(vocab_size=args.vocab, seq_len_margin=args.margin)
    print(f"p={args.p} vocab={args.vocab} seq_len={task.seq_len(args.p)}  ({args.n} problems)")
    for name, acc in shortcut_baselines(task, args.p, args.n).items():
        print(f"  {name:28s} {acc:6.1%}")


if __name__ == "__main__":
    main()
