"""Data-level shortcut check for a p-hop generator: how well do strategies that skip hops score?

No model and no training — it only looks at generated problems. A task measures p-hop reasoning
only if the cheap strategies below sit near chance (1 / vocab_size). Run it before trusting any
accuracy on a p-hop config.

    uv run python scripts/synth_shortcut_check.py --task p_hop --p 8 --vocab 4 --seq-len 64
    uv run python scripts/synth_shortcut_check.py --task p_hop_induction --p 4 --vocab 256 --margin 4
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from hallm.synth.data import make_task
from hallm.synth.tasks import SynthTask


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


def shortcut_baselines(task: SynthTask, p: int, n: int, seed: int = 123) -> dict[str, float]:
    """Accuracy of hop-free (and partial-hop) strategies on `n` generated problems.

    Keys: `chance`; `copy_best_fixed_position` (always output the token at the single best
    position); `max_answer_position_share` (how concentrated the answer's position is);
    `most_frequent_token`; `stop_after_<k>_hops` for k < p; `answer_share_max` (label balance)."""
    rng = random.Random(seed)
    answer_pos: Counter[int] = Counter()
    answers: Counter[int] = Counter()
    hits: Counter[str] = Counter()
    by_position: Counter[int] = Counter()
    for _ in range(n):
        problem = task.generate(rng, p)
        tokens, answer = problem.tokens[: problem.answer_start], problem.answer[0]
        path = hop_path(tokens, p)
        answer_pos[path[-1]] += 1
        answers[answer] += 1
        for k in range(p):
            hits[f"stop_after_{k}_hops"] += tokens[path[k]] == answer
        hits["most_frequent_token"] += Counter(tokens).most_common(1)[0][0] == answer
        for i, tok in enumerate(tokens):
            by_position[i] += tok == answer
    out = {"chance": 1 / task.vocab_size}
    out["copy_best_fixed_position"] = max(by_position.values()) / n
    out["max_answer_position_share"] = max(answer_pos.values()) / n
    out["answer_share_max"] = max(answers.values()) / n
    out["most_frequent_token"] = hits.pop("most_frequent_token") / n
    out.update({name: hits[name] / n for name in (f"stop_after_{k}_hops" for k in range(p))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="p_hop", choices=("p_hop", "p_hop_induction"))
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=64, help="p_hop only")
    ap.add_argument("--margin", type=int, default=4, help="p_hop_induction only")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    if args.task == "p_hop":
        task = make_task("p_hop", vocab_size=args.vocab or 4, seq_len=args.seq_len)
    else:
        task = make_task("p_hop_induction", vocab_size=args.vocab or 256, seq_len_margin=args.margin)
    print(f"{args.task}: p={args.p} vocab={task.vocab_size} seq_len={task.seq_len(args.p)}  ({args.n} problems)")
    for name, acc in shortcut_baselines(task, args.p, args.n).items():
        print(f"  {name:28s} {acc:6.1%}")


if __name__ == "__main__":
    main()
