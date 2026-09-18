"""Track 2 synthetic task generators (spec 2026-09-14 §5): each generator is checked against an
independent reference solver, not just against its own internal bookkeeping -- a generator that
was self-consistently wrong would still pass tests that only re-derive the answer the same way
it was built."""

from __future__ import annotations

import random

import pytest

from hallm.synth.tasks import (
    AdditionTask,
    BindingChainTask,
    PHopInductionTask,
    SynthProblem,
    solve_binding_chain,
    solve_p_hop,
)


# --- SynthProblem ------------------------------------------------------------------------------

def test_synth_problem_rejects_bad_answer_start():
    with pytest.raises(ValueError):
        SynthProblem(tokens=[1, 2, 3], answer_start=3)
    with pytest.raises(ValueError):
        SynthProblem(tokens=[1, 2, 3], answer_start=-1)


def test_synth_problem_answer_property():
    p = SynthProblem(tokens=[1, 2, 3, 4], answer_start=2)
    assert p.answer == [3, 4]


# --- p-hop induction -----------------------------------------------------------------------

@pytest.mark.parametrize("p", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_p_hop_matches_independent_solver(p, seed):
    task = PHopInductionTask(vocab_size=64)
    rng = random.Random(seed)
    problem = task.generate(rng, p)
    prompt = problem.tokens[: problem.answer_start]
    assert solve_p_hop(prompt, p, len(prompt) - 1) == problem.answer[0]


def test_p_hop_tokens_within_vocab():
    task = PHopInductionTask(vocab_size=16)
    rng = random.Random(7)
    for p in range(6):
        problem = task.generate(rng, p)
        assert all(0 <= t < task.vocab_size for t in problem.tokens)


def test_p_hop_deterministic_given_same_rng_state():
    task = PHopInductionTask(vocab_size=64)
    a = task.generate(random.Random(99), 4)
    b = task.generate(random.Random(99), 4)
    assert a.tokens == b.tokens
    assert a.answer_start == b.answer_start


def test_p_hop_answer_start_matches_seq_len():
    task = PHopInductionTask(vocab_size=64)
    rng = random.Random(3)
    for p in (0, 1, 4):
        problem = task.generate(rng, p)
        assert problem.answer_start == task.seq_len(p)
        assert len(problem.tokens) == task.seq_len(p) + 1


def test_p_hop_rejects_negative_difficulty():
    with pytest.raises(ValueError):
        PHopInductionTask().generate(random.Random(0), -1)


def test_p_hop_stress_many_seeds_and_hop_counts():
    """Broad randomized cross-check (the landmark-placement logic had a subtle collision bug that
    only showed up for specific seed/hop-count combinations, so a handful of parametrized cases
    isn't enough coverage on its own)."""
    task = PHopInductionTask(vocab_size=64)
    checked = 0
    for seed in range(300):
        for p in range(0, 10):
            rng = random.Random(seed * 1000 + p)
            try:
                problem = task.generate(rng, p)
            except ValueError:
                continue  # too-short case, expected occasionally at high p with this margin
            prompt = problem.tokens[: problem.answer_start]
            assert solve_p_hop(prompt, p, len(prompt) - 1) == problem.answer[0], (seed, p)
            checked += 1
    assert checked > 2500  # sanity: most combinations should actually generate, not skip


def test_p_hop_rejects_too_short_sequence_for_hop_count():
    task = PHopInductionTask(vocab_size=64, seq_len_margin=1)
    # margin=1 -> seq_len(p) = max(4, 1*(p+1)); force a p large enough that available room runs out.
    with pytest.raises(ValueError):
        task.generate(random.Random(0), 50)


# --- addition --------------------------------------------------------------------------------

@pytest.mark.parametrize("digits", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 42, 999])
def test_addition_sum_is_correct(digits, seed):
    task = AdditionTask()
    rng = random.Random(seed)
    problem = task.generate(rng, digits)
    prompt = problem.tokens[: problem.answer_start]
    a_digits = prompt[:digits]
    b_digits = prompt[digits + 1 : digits + 1 + digits]
    a = int("".join(str(d) for d in a_digits))
    b = int("".join(str(d) for d in b_digits))
    assert task.decode_answer(problem.answer) == a + b


def test_addition_handles_carry_overflow():
    # digits=1: 9+9=18 needs the extra digit slot.
    task = AdditionTask()
    found_overflow = False
    for seed in range(200):
        problem = task.generate(random.Random(seed), 1)
        if task.decode_answer(problem.answer) >= 10:
            found_overflow = True
            break
    assert found_overflow, "expected at least one carry-producing sample in 200 draws"


def test_addition_reverse_output_round_trips():
    task = AdditionTask(reverse_output=True)
    rng = random.Random(5)
    for _ in range(20):
        problem = task.generate(rng, 3)
        prompt = problem.tokens[: problem.answer_start]
        a = int("".join(str(d) for d in prompt[:3]))
        b = int("".join(str(d) for d in prompt[4:7]))
        assert task.decode_answer(problem.answer) == a + b


def test_addition_fixed_length_per_difficulty():
    task = AdditionTask()
    rng = random.Random(1)
    for digits in (1, 2, 5):
        lengths = {len(task.generate(rng, digits).tokens) for _ in range(30)}
        assert lengths == {task.seq_len(digits)}


def test_addition_rejects_zero_digits():
    with pytest.raises(ValueError):
        AdditionTask().generate(random.Random(0), 0)


# --- variable-binding chains -----------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("seed", [0, 1, 42, 777])
def test_binding_chain_matches_independent_solver(k, seed):
    task = BindingChainTask(max_vars=32)
    rng = random.Random(seed)
    problem = task.generate(rng, k)
    prompt = problem.tokens[: problem.answer_start]
    resolved = solve_binding_chain(prompt, task.EQ, task.SEMI, task.QMARK, task._N_SPECIAL)
    assert resolved == problem.answer[0]


def test_binding_chain_uses_distinct_variable_slots():
    task = BindingChainTask(max_vars=32)
    rng = random.Random(2)
    for k in (1, 5, 10):
        problem = task.generate(rng, k)
        prompt = problem.tokens[: problem.answer_start]
        var_tokens = [t for t in prompt if t >= task._N_SPECIAL]
        assert len(set(var_tokens)) == k, "each chain step must use a distinct variable"


def test_binding_chain_seq_len_matches_actual_length():
    task = BindingChainTask(max_vars=32)
    rng = random.Random(4)
    for k in (1, 2, 7):
        problem = task.generate(rng, k)
        assert problem.answer_start == task.seq_len(k)


def test_binding_chain_rejects_chain_longer_than_max_vars():
    task = BindingChainTask(max_vars=4)
    with pytest.raises(ValueError):
        task.generate(random.Random(0), 5)


def test_binding_chain_rejects_zero_length():
    with pytest.raises(ValueError):
        BindingChainTask().generate(random.Random(0), 0)


# --- shared interface --------------------------------------------------------------------------

def test_all_three_tasks_expose_name_and_vocab_size():
    for task in (PHopInductionTask(), AdditionTask(), BindingChainTask()):
        assert isinstance(task.name, str) and task.name
        assert isinstance(task.vocab_size, int) and task.vocab_size > 0
