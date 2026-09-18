"""Track 2 synthetic reasoning tasks (spec 2026-09-14 §5).

Three generated tasks, each with one integer "difficulty" knob, following Saunshi et al.
(ICLR 2025, arXiv 2502.17416): p-hop induction, multi-digit addition, and variable-binding
chains. Each task returns fixed-length token sequences for a given difficulty level, so a
batch of problems at the same difficulty stacks into a single tensor with no padding.

A `SynthProblem` separates the full token sequence into a "prompt" part (everything before
`answer_start`, never supervised) and an "answer" part (`tokens[answer_start:]`, the only
positions a training loss or exact-match check should look at). This mirrors how the model
already computes loss (`GPT.forward` uses `ignore_index=-1`), so a synth training loop just
needs to set every prompt-position target to -1.

Every generator is deterministic given an `random.Random` instance, so a fixed seed (plus a
different seed for train vs. held-out eval, per spec §5's "generated from a separate seed")
gives reproducible, non-overlapping-in-expectation problem streams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SynthProblem:
    """One generated problem. `tokens[:answer_start]` is the prompt (never supervised);
    `tokens[answer_start:]` is the answer (the only positions with a real training target)."""

    tokens: list[int]
    answer_start: int

    @property
    def answer(self) -> list[int]:
        return self.tokens[self.answer_start :]

    def __post_init__(self) -> None:
        if not 0 <= self.answer_start < len(self.tokens):
            raise ValueError(f"answer_start {self.answer_start} out of range for {len(self.tokens)} tokens")


class SynthTask(Protocol):
    """Common interface the training/eval harness drives every task through."""

    name: str
    vocab_size: int

    def generate(self, rng: random.Random, difficulty: int) -> SynthProblem: ...

    def seq_len(self, difficulty: int) -> int: ...


# --- p-hop induction --------------------------------------------------------------------------
#
# "Starting at the last token, p times jump to the token after the previous occurrence of the
# current token; answer = the final token." Constructed backward from the end of the sequence so
# every hop is well-defined by construction (never left to chance): at each hop we place a second
# occurrence of the current value at a fresh earlier position, and record that no filler token in
# between may accidentally equal that value (which would make an earlier occurrence "the previous
# one" instead of the one we planted).


class PHopInductionTask:
    name = "p_hop_induction"

    def __init__(self, vocab_size: int = 64, seq_len_margin: int = 6) -> None:
        self.vocab_size = vocab_size
        # Each hop consumes at least 1 position of "room" to move backward, but we want plenty of
        # slack so hops aren't forced to be adjacent (that would make the task trivially easy).
        self._margin = seq_len_margin

    def seq_len(self, difficulty: int) -> int:
        p = difficulty
        return max(4, self._margin * (p + 1))

    def generate(self, rng: random.Random, difficulty: int) -> SynthProblem:
        p = difficulty
        if p < 0:
            raise ValueError("p-hop difficulty (hops) must be >= 0")
        n = self.seq_len(difficulty)
        seq: list[int | None] = [None] * n
        forbidden: list[tuple[int, int, int]] = []  # (lo, hi, value) inclusive range must avoid value

        pos = n - 1
        seq[pos] = rng.randrange(self.vocab_size)
        for hop_i in range(p):
            hops_remaining_after = p - hop_i - 1
            # A hop needs pos >= 2 (room for q in [0, pos-2]); to still be able to finish the
            # remaining hops afterward (each costs at least 1 position of room in the slowest-
            # shrinking case), new_pos must leave at least `hops_remaining_after + 1` of room.
            needed_min_new_pos = hops_remaining_after + 1 if hops_remaining_after > 0 else 0
            lo = max(0, needed_min_new_pos - 1)
            hi = pos - 2
            if lo > hi:
                raise ValueError(f"seq_len {n} too short for {p} hops (ran out of room at pos {pos})")
            v = seq[pos]

            def _valid(q: int) -> bool:
                # The landmark position q must be free (or already luckily holding v). Its
                # neighbour q+1 (=new_pos, the left edge of the forbidden range this hop is about
                # to record) can collide with the PREVIOUS hop's landmark -- q+1 is exactly that
                # position's maximum possible value -- so if it's already assigned, that value
                # must not equal v (else it would be a closer "previous occurrence" than q,
                # corrupting this hop's own invariant).
                if seq[q] is not None and seq[q] != v:
                    return False
                if seq[q + 1] is not None and seq[q + 1] == v:
                    return False
                return True

            q = rng.randint(lo, hi)
            if not _valid(q):
                for _ in range(20):
                    q = rng.randint(lo, hi)
                    if _valid(q):
                        break
                else:
                    raise RuntimeError("could not place a p-hop landmark after 20 retries")
            seq[q] = v
            forbidden.append((q + 1, pos - 1, v))
            new_pos = q + 1
            if seq[new_pos] is None:
                seq[new_pos] = rng.choice([s for s in range(self.vocab_size) if s != v])
            pos = new_pos

        answer = seq[pos]
        answer_pos = pos

        # Fill every remaining slot with a value that doesn't violate any forbidden-range
        # constraint collected above (each one guards that a planted "previous occurrence" really
        # is the closest one before its landmark position).
        for i in range(n):
            if seq[i] is not None:
                continue
            banned = {v for lo, hi, v in forbidden if lo <= i <= hi}
            candidates = [s for s in range(self.vocab_size) if s not in banned]
            if not candidates:
                raise RuntimeError(f"vocab_size {self.vocab_size} too small to satisfy constraints at pos {i}")
            seq[i] = rng.choice(candidates)

        tokens = [int(t) for t in seq]  # type: ignore[arg-type]
        # The "answer" is the value the walk lands on; we append it as one extra supervised
        # position so the task has the same prompt/answer shape as the other two tasks (predict
        # the token at the final position given everything before it).
        full = tokens + [answer]
        return SynthProblem(tokens=full, answer_start=len(tokens))


def solve_p_hop(tokens: list[int], p: int, start_pos: int) -> int:
    """Reference solver: independently re-derive the p-hop answer from a raw sequence, by
    literally walking 'jump to the token after the previous occurrence' p times. Used only in
    tests, to cross-check `PHopInductionTask.generate` against a from-scratch implementation."""
    pos = start_pos
    for _ in range(p):
        v = tokens[pos]
        prev = None
        for j in range(pos - 1, -1, -1):
            if tokens[j] == v:
                prev = j
                break
        if prev is None:
            raise ValueError(f"no previous occurrence of {v} before position {pos}")
        pos = prev + 1
    return tokens[pos]


# --- multi-digit addition ---------------------------------------------------------------------
#
# "a+b=" -> digits of the sum. Both operands are exactly `digits` digits (leading zeros allowed,
# so the sequence length is fixed for a given difficulty); the sum is always represented with
# `digits + 1` digits (a leading zero when there's no carry-out), so the answer length is fixed
# too. Digit order is left MSB-first (the literal "digits of the sum" reading) -- `reverse_output`
# exists so the sizing pilot can flip to LSB-first (a well-known trick for transformer addition)
# without a redesign, if the MSB-first ordering turns out to be hard to learn.


class AdditionTask:
    name = "addition"
    PLUS = 10
    EQUALS = 11
    vocab_size = 12  # digits 0-9 + PLUS + EQUALS

    def __init__(self, reverse_output: bool = False) -> None:
        self.reverse_output = reverse_output

    def seq_len(self, difficulty: int) -> int:
        d = difficulty
        return d + 1 + d + 1 + (d + 1)  # a, PLUS, b, EQUALS, sum(d+1 digits)

    def generate(self, rng: random.Random, difficulty: int) -> SynthProblem:
        d = difficulty
        if d < 1:
            raise ValueError("addition difficulty (digits) must be >= 1")
        a = rng.randrange(10**d)
        b = rng.randrange(10**d)
        s = a + b

        a_digits = _to_digits(a, d)
        b_digits = _to_digits(b, d)
        s_digits = _to_digits(s, d + 1)
        if self.reverse_output:
            s_digits = list(reversed(s_digits))

        prompt = a_digits + [self.PLUS] + b_digits + [self.EQUALS]
        return SynthProblem(tokens=prompt + s_digits, answer_start=len(prompt))

    def decode_answer(self, answer_tokens: list[int]) -> int:
        digits = list(reversed(answer_tokens)) if self.reverse_output else answer_tokens
        return int("".join(str(t) for t in digits))


def _to_digits(n: int, width: int) -> list[int]:
    s = str(n).rjust(width, "0")
    if len(s) > width:
        raise ValueError(f"{n} does not fit in {width} digits")
    return [int(c) for c in s]


# --- variable-binding chains ("i-GSM-lite") ---------------------------------------------------
#
# "a=3;b=a;c=b;...;x?" -> the value. A chain of `difficulty` variables, the first bound to a
# random constant and each following one aliasing the previous variable; the query asks for the
# last variable's (fully-resolved) value. Variable identities are drawn fresh per problem from a
# pool larger than any chain length, so the model can't shortcut on fixed positional names -- it
# has to actually follow the chain.


class BindingChainTask:
    name = "binding_chain"
    SEMI = 10
    EQ = 11
    QMARK = 12
    _N_SPECIAL = 13  # 0-9 digits + SEMI + EQ + QMARK
    CONST_RANGE = 10  # single-digit constants, token ids 0-9

    def __init__(self, max_vars: int = 64) -> None:
        self.max_vars = max_vars
        self.vocab_size = self._N_SPECIAL + max_vars

    def _var_token(self, slot: int) -> int:
        return self._N_SPECIAL + slot

    def seq_len(self, difficulty: int) -> int:
        k = difficulty
        # Prompt length only (matches PHopInductionTask's convention: answer_start == seq_len()):
        # k steps of [var, EQ, value] + (k-1) SEMI separators between steps + one SEMI before the
        # query + the [var, QMARK] query itself.
        return 3 * k + (k - 1) + 1 + 2

    def generate(self, rng: random.Random, difficulty: int) -> SynthProblem:
        k = difficulty
        if k < 1:
            raise ValueError("binding-chain difficulty (chain length) must be >= 1")
        if k > self.max_vars:
            raise ValueError(f"chain length {k} exceeds max_vars {self.max_vars}")
        var_slots = rng.sample(range(self.max_vars), k)
        const = rng.randrange(self.CONST_RANGE)

        tokens: list[int] = []
        for i, slot in enumerate(var_slots):
            if i > 0:
                tokens.append(self.SEMI)
            bound_value = const if i == 0 else self._var_token(var_slots[i - 1])
            tokens += [self._var_token(slot), self.EQ, bound_value]
        tokens += [self.SEMI, self._var_token(var_slots[-1]), self.QMARK]

        return SynthProblem(tokens=tokens + [const], answer_start=len(tokens))


def solve_binding_chain(tokens: list[int], eq: int, semi: int, qmark: int, n_special: int) -> int:
    """Reference solver: split the `var=value;...;var?` token stream on SEMI into `[var, EQ,
    value]` binding segments plus a final `[var, QMARK]` query segment, resolve each binding
    against the ones seen so far (valid because the chain always defines a variable before
    referencing it), then look up the queried variable. Independent of how
    `BindingChainTask.generate` built the sequence -- used only in tests."""
    segments: list[list[int]] = [[]]
    for t in tokens:
        if t == semi:
            segments.append([])
        else:
            segments[-1].append(t)

    bindings: dict[int, int] = {}
    for var, eq_tok, value in segments[:-1]:
        assert eq_tok == eq
        bindings[var] = value if value < n_special else bindings[value]

    query_var, q = segments[-1]
    assert q == qmark
    return bindings[query_var]
