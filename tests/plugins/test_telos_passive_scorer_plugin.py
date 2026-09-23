"""Tests for the bundled observability/telos_passive_scorer plugin."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "telos_passive_scorer"
DIMENSIONS = {"purpose", "scope", "boundary", "tool", "chain", "risk"}


class _FakeContext:
    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}

    def register_hook(self, name: str, callback) -> None:
        self.hooks[name] = callback


def _fresh_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    for name in list(sys.modules):
        if name == "plugins.observability.telos_passive_scorer":
            sys.modules.pop(name, None)
    plugin = importlib.import_module("plugins.observability.telos_passive_scorer")
    plugin.reset_for_tests()
    return plugin


def _receipt_path(tmp_path: Path) -> Path:
    return tmp_path / "hermes_home" / "telos" / "operational_receipts" / "tool_calls.jsonl"


def _read_receipts(tmp_path: Path) -> list[dict]:
    path = _receipt_path(tmp_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_non_gating_degradation(result, scoring_status: str) -> None:
    """Since 612f40fe78 a degraded scorer returns a non-gating notice instead of None.

    The action is "observed", the resolver's explicit non-gating directive. It was "observe", which
    the resolver silently dropped. It must never be a policy directive (block/approve/modify).
    """
    assert isinstance(result, dict), result
    assert result["action"] == "observed"
    assert result["action"] not in {"block", "approve", "modify"}
    assert result["scoring_status"] == scoring_status


def _assert_tool_call_proceeds(plugin, monkeypatch, **call) -> None:
    """Route the plugin's real return through the host resolver: it must not gate the call."""
    from hermes_cli import plugins as host_plugins

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kw: [plugin.on_pre_tool_call(**kw)] if name == "pre_tool_call" else [],
    )
    tool_name, args = call.pop("tool_name"), call.pop("args")
    directive, message = host_plugins.get_pre_tool_call_directive(tool_name, args, **call)
    # Surfaced as the non-gating "observed" directive (previously dropped as irrelevant), never a gate.
    assert directive == "observed", (directive, message)
    block_message, modified_args = host_plugins._dispatch_pre_tool_call_hooks(tool_name, args, **call)
    assert block_message is None and modified_args is None, (block_message, modified_args)


def test_manifest_declares_tool_call_hooks():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    assert data["name"] == "telos_passive_scorer"
    assert set(data["hooks"]) == {"pre_tool_call", "post_tool_call"}


def test_register_adds_pre_and_post_hooks(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    ctx = _FakeContext()

    plugin.register(ctx)

    assert set(ctx.hooks) == {"pre_tool_call", "post_tool_call"}
    assert ctx.hooks["pre_tool_call"] is plugin.on_pre_tool_call
    assert ctx.hooks["post_tool_call"] is plugin.on_post_tool_call


def test_pre_and_post_write_passive_allowed_receipt_shape(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    # Deterministic in any checkout: no closed-core scorer, as in a clean public clone.
    monkeypatch.setattr(plugin, "_SCORER", None)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)

    _assert_non_gating_degradation(plugin.on_pre_tool_call(
        tool_name="computer_use",
        args={"action": "click", "x": 42, "api_key": "sk-secret"},
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        turn_id="turn-1",
        api_request_id="api-1",
    ), "scorer_missing")
    plugin.on_post_tool_call(
        tool_name="computer_use",
        args={"action": "click", "x": 42},
        result={"ok": True, "screenshot_path": "/Users/example/private/screen.png"},
        task_id="task-1",
        session_id="session-1",
        tool_call_id="call-1",
        turn_id="turn-1",
        api_request_id="api-1",
        duration_ms=12.5,
        status="ok",
    )

    [receipt] = _read_receipts(tmp_path)
    assert receipt["schema_version"] == "telos.operational.tool_call.v2"  # v2 since 612f40fe78
    assert receipt["tool_name"] == "computer_use"
    assert receipt["allowed"] is True
    assert receipt["passive"] is True
    assert receipt["scoring_status"] == "scorer_missing"
    assert receipt["governance_degraded"] is True
    assert receipt["closed_core_mechanism"] == "guarded_optional_import"
    assert set(receipt["dimensions"]) == DIMENSIONS
    # Since 612f40fe78 a missing scorer yields an explicit "unscored" verdict, not a synthetic
    # "execute". The call itself is still allowed: receipt["allowed"] above is the gate, and it is True.
    assert receipt["verdict"]["decision"] == "unscored"
    assert receipt["verdict"]["fidelity"] is None
    assert receipt["classification"]["tool_group"] == "computer_use"
    assert receipt["post"]["status"] == "ok"
    assert receipt["post"]["duration_ms"] == 12.5


def test_optional_scorer_verdict_is_recorded_but_never_blocks(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)

    def scorer(**_kwargs):
        return {
            "decision": "escalate",
            "allowed": False,
            "fidelity": 0.12,
            "dimensions": {
                "purpose": 0.1,
                "scope": 0.2,
                "boundary": 0.9,
                "tool": 0.3,
                "chain": 0.4,
                "risk": 0.95,
            },
        }

    monkeypatch.setattr(plugin, "_SCORER", scorer)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)
    assert plugin.on_pre_tool_call(tool_name="shell", args={"cmd": "rm -rf /"}, tool_call_id="c1") is None
    plugin.on_post_tool_call(tool_name="shell", args={"cmd": "rm -rf /"}, result="done", tool_call_id="c1")

    [receipt] = _read_receipts(tmp_path)
    assert receipt["allowed"] is True
    assert receipt["passive"] is True
    assert receipt["scoring_status"] == "scored"
    assert receipt["verdict"]["decision"] == "escalate"
    assert receipt["verdict"]["allowed"] is False
    assert set(receipt["dimensions"]) == DIMENSIONS


def test_scorer_exception_fails_open_and_writes_unscored_receipt(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)

    def broken_scorer(**_kwargs):
        raise RuntimeError("closed scorer unavailable: token abc123")

    monkeypatch.setattr(plugin, "_SCORER", broken_scorer)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)

    _assert_non_gating_degradation(
        plugin.on_pre_tool_call(tool_name="write_file", args={"path": "/Users/example/secret.txt"}, tool_call_id="c2"),
        "scorer_error",
    )
    plugin.on_post_tool_call(tool_name="write_file", args={}, result={"error": "see /Users/example/secret.txt"}, tool_call_id="c2")

    [receipt] = _read_receipts(tmp_path)
    assert receipt["scoring_status"] == "scorer_error"
    assert receipt["governance_degraded"] is True
    assert receipt["errors"][0]["message"] == "closed scorer unavailable: token [REDACTED]"
    assert receipt["args"]["path"] == "[REDACTED_PATH]"
    # Since the redaction fix, a tool RESULT persists shape only. The old assertion here checked
    # that the result's text was redacted; the contract is now that no result text is stored at
    # all, which is what the absence of the path anywhere in the receipt proves.
    assert "/Users/example/secret.txt" not in _receipt_path(tmp_path).read_text(encoding="utf-8")
    assert receipt["post"]["result_metadata"]["content_persisted"] is False
    assert len(receipt["post"]["result_metadata"]["sha256"]) == 64


def test_slow_scorer_times_out_and_fails_open(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_TELOS_PASSIVE_SCORER_TIMEOUT_MS", "1")

    def slow_scorer(**_kwargs):
        time.sleep(0.05)
        return {"decision": "escalate", "allowed": False}

    monkeypatch.setattr(plugin, "_SCORER", slow_scorer)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)

    start = time.perf_counter()
    _assert_non_gating_degradation(
        plugin.on_pre_tool_call(tool_name="computer_use", args={"action": "drag"}, tool_call_id="slow-1"),
        "scorer_timeout",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    plugin.on_post_tool_call(tool_name="computer_use", args={}, result={"ok": True}, tool_call_id="slow-1")

    [receipt] = _read_receipts(tmp_path)
    assert elapsed_ms < 40
    assert receipt["allowed"] is True
    assert receipt["scoring_status"] == "scorer_timeout"
    assert receipt["governance_degraded"] is True
    assert "timed out" in receipt["errors"][0]["message"]


def test_post_without_pre_still_writes_receipt(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)

    plugin.on_post_tool_call(
        tool_name="read_file",
        args={"path": "/Users/example/ops/private.md"},
        result="contents",
        task_id="task-x",
        session_id="session-x",
        tool_call_id="missing-pre",
        duration_ms=3,
        status="ok",
    )

    [receipt] = _read_receipts(tmp_path)
    assert receipt["correlation_id"] == "missing-pre"
    assert receipt["scoring_status"] == "unscored"
    assert receipt["allowed"] is True
    assert receipt["args"]["path"] == "[REDACTED_PATH]"
    assert receipt["post"]["status"] == "ok"


def test_redacts_nested_secrets_paths_and_result_metadata(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)

    plugin.on_pre_tool_call(
        tool_name="api_call",
        args={
            "headers": {"Authorization": "Bearer secret-token"},
            "files": ["/Users/example/.telos/private.key", "/tmp/ok.txt"],
            "nested": {"access_token": "tok_123"},
        },
        tool_call_id="redact-1",
    )
    plugin.on_post_tool_call(
        tool_name="api_call",
        args={},
        result={
            "output_path": "/private/tmp/report.txt",
            "preview": "saved in /Users/example/Desktop/Project Files/file.txt with token abc123",
        },
        tool_call_id="redact-1",
        error=SimpleNamespace(message="failed for /Users/example/.telos/key"),
    )

    [receipt] = _read_receipts(tmp_path)
    assert receipt["args"]["headers"]["Authorization"] == "[REDACTED]"
    assert receipt["args"]["files"] == ["[REDACTED_PATH]", "[REDACTED_PATH]"]
    assert receipt["args"]["nested"]["access_token"] == "[REDACTED]"
    # The result no longer reaches the receipt in any form: shape only, content hashed.
    raw = _receipt_path(tmp_path).read_text(encoding="utf-8")
    assert "/Users/" not in raw
    assert "Project Files" not in raw
    assert "IP/file.txt" not in raw
    assert "output_path" not in raw
    assert receipt["post"]["result_metadata"]["type"] == "dict"
    assert receipt["post"]["result_metadata"]["content_persisted"] is False
    # Error messages are still recorded as redacted TEXT -- an error is diagnostic, not payload.
    assert receipt["post"]["error"]["message"] == "failed for [REDACTED_PATH]"


def test_degraded_scorer_never_gates_the_tool_call_through_the_host_resolver(monkeypatch, tmp_path):
    """The load-bearing property behind the three degradation tests: observation-only holds end to end."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)

    def broken_scorer(**_kwargs):
        raise RuntimeError("closed scorer unavailable")

    monkeypatch.setattr(plugin, "_SCORER", broken_scorer)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)
    _assert_tool_call_proceeds(plugin, monkeypatch, tool_name="write_file", args={"path": "x"}, tool_call_id="g1")
