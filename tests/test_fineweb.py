"""FineWeb-Edu writer: EOT-separated documents, exact token budgets, disjoint sequential splits.
Network-free: a fake byte-level batch encoder stands in for tiktoken."""
from __future__ import annotations

import numpy as np

from hallm.data.fineweb import EOT, write_until


def _enc(texts):
    return [[ord(c) for c in t] for t in texts]


def _read(p):
    return np.fromfile(p, dtype=np.uint16).tolist()


def test_documents_are_separated_by_eot(tmp_path):
    assert write_until([["ab", "c"]], _enc, tmp_path / "x.bin", 100) == 5
    assert _read(tmp_path / "x.bin") == [97, 98, EOT, 99, EOT]


def test_stops_exactly_at_the_budget(tmp_path):
    assert write_until([["abcdef"]], _enc, tmp_path / "x.bin", 4) == 4
    assert _read(tmp_path / "x.bin") == [97, 98, 99, 100]


def test_returns_fewer_when_the_corpus_runs_out(tmp_path):
    assert write_until([["ab"]], _enc, tmp_path / "x.bin", 100) == 3


def test_sequential_calls_on_one_stream_are_disjoint(tmp_path):
    stream = iter([["aa"], ["bb"], ["cc"]])
    write_until(stream, _enc, tmp_path / "v.bin", 3)
    write_until(stream, _enc, tmp_path / "t.bin", 3)
    assert _read(tmp_path / "v.bin") == [97, 97, EOT]
    assert _read(tmp_path / "t.bin") == [98, 98, EOT]
