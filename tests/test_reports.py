"""Derived comparison tables. Every number here must be reproducible from per-run files alone."""

import math

from hallm.reports import hs_verdict, index_runs, mean_se, parse_run_id, regress, tax


def test_parse_run_id_and_reject_malformed():
    assert parse_run_id("L16-A2-s1339") == {"depth": 16, "arm": "A2", "seed": 1339}
    assert parse_run_id("L8-A2ffn-s1337")["arm"] == "A2ffn"
    assert parse_run_id("smoke-A0-s7") is None
    assert parse_run_id("nonsense") is None


def test_index_runs_groups_arms_by_rung_and_seed():
    rows = [{"run": "L8-A0-s1337", "test_ppl": 26.06}, {"run": "L8-A2-s1337", "test_ppl": 29.68},
            {"run": "L8-A2ffn-s1337", "test_ppl": 27.0}]
    idx = index_runs(rows)
    assert set(idx[(8, 1337)]) == {"A0", "A2", "A2ffn"}


def test_tax_matches_the_published_experiment_1_number():
    assert abs(tax(29.68, 26.06) - 13.891) < 0.01


def test_mean_se_single_value_has_no_standard_error():
    m, se = mean_se([13.9])
    assert m == 13.9 and math.isnan(se)


def test_regress_recovers_a_known_negative_slope():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [10.0, 8.0, 6.0, 4.0]
    r = regress(xs, ys)
    assert abs(r["slope"] + 2.0) < 1e-9
    assert r["ci_lo"] < 0 and r["ci_hi"] < 0
    assert hs_verdict(r).startswith("SUPPORTED")


def test_regress_reports_inconclusive_when_ci_spans_zero():
    r = regress([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 4.0, 6.5])
    assert hs_verdict(r).startswith("INCONCLUSIVE")


def test_regress_needs_at_least_three_points():
    assert hs_verdict(regress([1.0, 2.0], [3.0, 4.0])) == "insufficient data"
