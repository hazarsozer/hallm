"""Comparison tables DERIVED from per-run result files.

Nothing here is a source of truth — every table is rebuilt from results/runs/*.json and is always
safe to delete. This is the half of the results design that replaces the append-only ledger: the
ledger tried to be both the record and the view, which is why it could not be regenerated.
"""

from __future__ import annotations

import math

# Two-sided 95% t critical values by degrees of freedom; the campaign never has many seeds, so a
# normal approximation would overstate confidence at the n we actually run.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T95.get(df, 1.96)


def parse_run_id(run_id: str) -> dict | None:
    """`L<depth>-A<arm>-s<seed>` -> parts. Tooling keys on the ID string; the manifest remains the
    authority on configuration (artifact-layout spec, rule 3)."""
    parts = run_id.split("-")
    if len(parts) != 3 or not parts[0].startswith("L") or not parts[2].startswith("s"):
        return None
    try:
        return {"depth": int(parts[0][1:]), "arm": parts[1], "seed": int(parts[2][1:])}
    except ValueError:
        return None


def index_runs(rows: list[dict]) -> dict[tuple[int, int], dict[str, dict]]:
    """(depth, seed) -> {arm_tag: row}."""
    out: dict[tuple[int, int], dict[str, dict]] = {}
    for r in rows:
        p = parse_run_id(r.get("run", ""))
        if not p:
            continue
        out.setdefault((p["depth"], p["seed"]), {})[p["arm"]] = r
    return out


def tax(shared_ppl: float, base_ppl: float) -> float:
    """Relative PPL cost of sharing, in percent. The measurand is always the PAIR difference."""
    return (shared_ppl - base_ppl) / base_ppl * 100.0


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def regress(xs: list[float], ys: list[float]) -> dict:
    """OLS slope of y on x with a 95% CI — the H-S decision statistic (program spec P3).

    Replaces the withdrawn min/max-overlap rule, which could only get harder to satisfy as seeds
    accumulated because a min/max range never shrinks with n.
    """
    n = len(xs)
    if n < 3:
        return {"n": n, "slope": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return {"n": n, "slope": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    df = n - 2
    s2 = sum(r * r for r in resid) / df
    se = math.sqrt(s2 / sxx)
    crit = t95(df) * se
    return {"n": n, "slope": slope, "intercept": intercept, "se": se,
            "ci_lo": slope - crit, "ci_hi": slope + crit, "df": df}


def hs_verdict(reg: dict) -> str:
    """Pre-registered (program spec P3): supported iff the slope's 95% CI excludes zero, negative."""
    lo, hi = reg.get("ci_lo"), reg.get("ci_hi")
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return "insufficient data"
    if hi < 0:
        return "SUPPORTED (tax decays with scale)"
    if lo > 0:
        return "REFUTED (tax grows with scale)"
    return "INCONCLUSIVE (CI spans zero)"
