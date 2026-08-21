"""Task discovery parsing and the collaborator-relevance filter.

These cover the pure logic; the `gh`/`git` calls are integration surface exercised by hand.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tasks import AFFECTS_COLLABORATOR, parse_task_block

BODY = """Some human-readable prose explaining the task.

```task
task_id: T-001
type: run
assignee: alpericon
why: check whether the L8 result holds at L4
runs:
  - id: L4-A2ffn-s1337
    est_hours: 4
  - id: L4-A2attn-s1337
    est_hours: 4
baseline_for_tax: L4-A0-s1337
protocol_reference: L4-A0-s1337
branch: alper/T-001-l4-decomposition
```

More prose afterwards.
"""


def test_parses_the_task_block_out_of_surrounding_prose():
    spec = parse_task_block(BODY)
    assert spec["task_id"] == "T-001"
    assert spec["assignee"] == "alpericon"
    assert [r["id"] for r in spec["runs"]] == ["L4-A2ffn-s1337", "L4-A2attn-s1337"]
    assert spec["protocol_reference"] == "L4-A0-s1337"


def test_issue_without_a_block_is_not_a_task():
    assert parse_task_block("just a normal issue, no machine-readable part") is None


def test_malformed_yaml_does_not_raise():
    """A human editing the block badly must not crash discovery — it degrades to 'no spec'."""
    assert parse_task_block("```task\n key: [unclosed\n```") is None


def test_empty_body_is_safe():
    assert parse_task_block("") is None
    assert parse_task_block(None) is None


def test_relevance_filter_covers_everything_a_collaborator_runs_from():
    for p in ("configs/runs/L4-A0-s1337.yaml", "scripts/run_queue.py",
              "src/hallm/train.py", "COLLABORATOR.md", "README.md",
              "docs/superpowers/specs/2026-08-20-research-program-design.md"):
        assert p.startswith(AFFECTS_COLLABORATOR), p


def test_relevance_filter_excludes_our_own_noise():
    for p in ("results/runs/L8-A0-s1338.json", "results/reports/ladder.md",
              "wiki/roadmap/06-scaling-campaign.md", "docs/superpowers/plans/x.md"):
        assert not p.startswith(AFFECTS_COLLABORATOR), p


def test_personal_claude_md_is_not_watched():
    """CLAUDE.md is gitignored and personal, so it can never appear in a commit —
    watching it would be dead code implying a sync that cannot happen."""
    assert not "CLAUDE.md".startswith(AFFECTS_COLLABORATOR)
