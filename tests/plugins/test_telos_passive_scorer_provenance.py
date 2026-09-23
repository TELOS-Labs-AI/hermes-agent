"""Receipts keep the scorer's provenance and never read a broken scorer as approval.

pa_hash was the sentinel sha256("missing:declared-pa") while pa_hash_source / pa_verification_status were
dropped, so a reader could not tell. None / {} / no-decision scorer output was receipted as `execute`, allowed.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _plugin(monkeypatch, tmp_path, scorer=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_TELOS_PASSIVE_SCORER_TIMEOUT_MS", raising=False)
    sys.modules.pop("plugins.observability.telos_passive_scorer", None)
    p = importlib.import_module("plugins.observability.telos_passive_scorer")
    p.reset_for_tests()
    monkeypatch.setattr(p, "_SCORER", scorer)
    monkeypatch.setattr(p, "_SCORER_LOADED", True)
    return p


def _receipt(p):
    import json
    return json.loads(p._receipt_path().read_text().splitlines()[-1])


def _run(p):
    kw = dict(tool_name="terminal", args={"command": "ls"}, tool_call_id="c1", session_id="s1")
    p.on_pre_tool_call(**kw)
    p.on_post_tool_call(**kw, status="ok", duration_ms=1, result="{}")
    return _receipt(p)


# G19/G20: provenance fields from the scorer survive into the receipt.
def test_receipt_keeps_scorer_provenance(monkeypatch, tmp_path):
    def scorer(**_):
        return {"decision": "execute", "allowed": True, "fidelity": 0.9, "dimensions": {},
                "pa_hash": "p" * 64, "pa_hash_source": "missing", "pa_verification_status": "missing",
                "engine_commit": "c" * 40, "embedder_id": "m@sha256:" + "e" * 64}
    r = _run(_plugin(monkeypatch, tmp_path, scorer))
    blob = repr(r)
    for value in ("pa_hash_source", "pa_verification_status", "c" * 40, "m@sha256:"):
        assert value in blob, value


# G21: malformed scorer output is never receipted as approval.
@pytest.mark.parametrize("raw", [None, {}, {"fidelity": 0.9}, "garbage"], ids=["none", "empty", "no-decision", "string"])
def test_malformed_scorer_output_is_not_execute(monkeypatch, tmp_path, raw):
    p = _plugin(monkeypatch, tmp_path, scorer=lambda **_: raw)
    verdict = p._normalize_verdict(raw)
    assert verdict["decision"] != "execute", verdict
    assert verdict["allowed"] is False, verdict


# G22: the default timeout (env unset) covers an engine cold start (measured ~1.35 s on a reference Mac).
def test_default_timeout_covers_cold_start(monkeypatch, tmp_path):
    p = _plugin(monkeypatch, tmp_path)
    assert p._scorer_timeout_ms() >= 2000
