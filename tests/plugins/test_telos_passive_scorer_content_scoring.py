"""Content-aware scoring for the bundled telos_passive_scorer plugin.

`_action_text` used to send the scorer only `tool(argKeyNames)`, so `echo hi` and
`git push --force ... customer.key` scored identically. The scorer now gets REDACTED argument values.
Every credential below is a FAKE canary.
"""

from __future__ import annotations

import importlib
import sys

import pytest

FAKE_KEY = "sk-proj-FAKEcanaryCONTENTSCORE0123456789ab"


def _fresh_plugin(monkeypatch, tmp_path, scorer=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    sys.modules.pop("plugins.observability.telos_passive_scorer", None)
    plugin = importlib.import_module("plugins.observability.telos_passive_scorer")
    plugin.reset_for_tests()
    monkeypatch.setattr(plugin, "_SCORER", scorer)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)
    return plugin


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    return _fresh_plugin(monkeypatch, tmp_path)


# G1: argument values reach the action text.
def test_action_text_carries_argument_values(plugin):
    text = plugin._action_text("terminal", {"command": "git push --force origin main"})
    assert "git push --force origin main" in text, text
    assert text.startswith("terminal"), text


# G2: different commands produce different action text (the live defect).
def test_different_commands_give_different_action_text(plugin):
    a = plugin._action_text("terminal", {"command": "echo hello"})
    b = plugin._action_text("terminal", {"command": "git push --force origin main"})
    assert a != b, (a, b)


# G3: secrets in argument values are redacted before they reach the scorer.
@pytest.mark.parametrize(
    "args",
    [
        {"command": f"export OPENAI_API_KEY={FAKE_KEY} && python run.py"},
        {"command": f"curl -H 'Authorization: Bearer {FAKE_KEY}' https://example.invalid"},
        {"api_key": FAKE_KEY},
        {"command": f"MY_SERVICE_KEY={FAKE_KEY} ./deploy.sh"},
        {"command": f"DEPLOY_TOKEN='{FAKE_KEY}' make release"},
    ],
    ids=["export-assignment", "bearer-header", "secret-named-arg", "generic-X_KEY", "quoted-token"],
)
def test_action_text_redacts_secrets(plugin, args):
    text = plugin._action_text("terminal", args)
    assert FAKE_KEY not in text, text


# G4: file paths keep the basename (the signal) and lose the directories.
def test_action_text_keeps_path_basename_not_directories(plugin):
    text = plugin._action_text("read_file", {"path": "/Users/someone/.telos/keys/customer.key"})
    assert "customer.key" in text, text
    assert "/Users/someone" not in text, text
    assert plugin._action_text("read_file", {"path": "/tmp/a/README.md"}) != text


# G5: bounded length, however large the arguments are.
def test_action_text_is_bounded(plugin):
    text = plugin._action_text("write_file", {"content": "x" * 200_000, "path": "/tmp/a.txt"})
    assert len(text) <= plugin.ACTION_TEXT_MAX_CHARS, len(text)


# G6: deterministic regardless of argument insertion order; stable for empty args.
def test_action_text_is_deterministic(plugin):
    a = plugin._action_text("t", {"b": "2", "a": "1"})
    b = plugin._action_text("t", {"a": "1", "b": "2"})
    assert a == b
    assert plugin._action_text("t", {}) == plugin._action_text("t", None) == "t()"


# G7: the scorer is actually CALLED with the content-bearing, redacted text (not only the helper).
def test_scorer_receives_redacted_content(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_scorer(**kwargs):
        seen.update(kwargs)
        return {"decision": "execute", "allowed": True, "fidelity": 0.9,
                "dimensions": {}, "explanation": "fake"}

    plugin = _fresh_plugin(monkeypatch, tmp_path, scorer=fake_scorer)
    plugin.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"export OPENAI_API_KEY={FAKE_KEY} && git push --force origin main"},
        tool_call_id="c1", session_id="s1",
    )
    assert "git push --force origin main" in seen.get("action", ""), seen
    assert FAKE_KEY not in repr(seen), seen
