"""Covers the network-free parts of the prepare script: the IPv6-blackhole workaround only, since
`convert`/`main` need a real streaming download (no meaningful offline test)."""

from __future__ import annotations

import socket

from scripts.prepare_imagenet100 import _force_ipv4


def test_force_ipv4_resolver_returns_only_af_inet_entries():
    original = socket.getaddrinfo
    try:
        _force_ipv4()
        results = socket.getaddrinfo("localhost", 80)
        assert results
        assert all(r[0] == socket.AF_INET for r in results)
    finally:
        socket.getaddrinfo = original


def test_force_ipv4_is_not_applied_merely_by_importing_the_module():
    # Opt-in only (decision: never patch unconditionally) — importing `prepare_imagenet100` and
    # calling `_force_ipv4` in the test above (which restores in its `finally`) must not leave the
    # process-wide resolver patched for anything that runs afterwards.
    assert socket.getaddrinfo.__name__ != "_ipv4_only"
