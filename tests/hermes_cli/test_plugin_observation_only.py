"""Executable pre-build contracts for item 10 against unmodified Hermes HEAD d2483dab."""
from __future__ import annotations

import logging
import threading
import time

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    _get_pre_tool_call_directive_details,
    get_pre_tool_call_block_message,
    get_pre_tool_call_directive,
    resolve_pre_tool_block,
)
from hermes_cli.plugins_manifest import parse_manifest_file


def _manifest(*, observation_only: bool = True, name: str = "observer") -> PluginManifest:
    manifest = PluginManifest(name=name, key=name, source="user")
    # Current HEAD has no constructor field. Setting the future declaration dynamically lets
    # dispatcher REDs reach the behavior they test instead of all stopping at one AttributeError.
    manifest.observation_only = observation_only
    return manifest


def _context(
    manager: PluginManager, *, observation_only: bool = True, name: str = "observer"
) -> PluginContext:
    return PluginContext(_manifest(observation_only=observation_only, name=name), manager)


def _config_override(monkeypatch, value) -> None:
    monkeypatch.setattr(
        "hermes_cli.plugins.load_config_readonly",
        lambda: {"plugins": {"entries": {"observer": {"observation_only": value}}}},
    )


def test_manifest_parses_observation_only_true(tmp_path):
    plugin = tmp_path / "observer"
    plugin.mkdir()
    manifest_file = plugin / "plugin.yaml"
    manifest_file.write_text(
        "name: observer\nmanifest_version: 2\nobservation_only: true\n",
        encoding="utf-8",
    )

    manifest = parse_manifest_file(manifest_file, plugin, "user", "")

    assert manifest is not None
    assert manifest.observation_only is True


def test_manifest_defaults_observation_only_false(tmp_path):
    plugin = tmp_path / "policy"
    plugin.mkdir()
    manifest_file = plugin / "plugin.yaml"
    manifest_file.write_text("name: policy\nmanifest_version: 2\n", encoding="utf-8")

    manifest = parse_manifest_file(manifest_file, plugin, "user", "")

    assert manifest is not None
    assert manifest.observation_only is False


@pytest.mark.parametrize("invalid", ['"yes"', "1"])
def test_manifest_rejects_non_boolean_observation_only(tmp_path, caplog, invalid):
    plugin = tmp_path / "observer"
    plugin.mkdir()
    manifest_file = plugin / "plugin.yaml"
    manifest_file.write_text(
        f"name: observer\nmanifest_version: 2\nobservation_only: {invalid}\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manifest = parse_manifest_file(manifest_file, plugin, "user", "")

    assert manifest is not None
    assert manifest.observation_only is False
    assert "observation_only must be a boolean; ignoring" in caplog.text


def test_config_true_overrides_manifest_false_and_warns(monkeypatch, caplog):
    _config_override(monkeypatch, True)
    manager = PluginManager()
    context = _context(manager, observation_only=False)
    context.register_hook("pre_tool_call", lambda **_kwargs: {"action": "block", "message": "must ignore"})
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        block = resolve_pre_tool_block("read_file", {"path": "README.md"})

    assert "operator marked as observation-only; its directives will be ignored" in caplog.text
    assert "observer" in caplog.text
    assert block is None


def test_config_false_overrides_manifest_true_to_policy(monkeypatch):
    _config_override(monkeypatch, False)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    manager = PluginManager()
    context = _context(manager, observation_only=True)
    release = threading.Event()

    def hung_policy(**_kwargs):
        release.wait(timeout=2.0)

    context.register_hook("pre_tool_call", hung_policy)
    try:
        result = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-policy")
    finally:
        release.set()

    assert result == [{"action": "block", "message": "pre_tool_call plugin callback timed out or is still running"}]


def test_observation_only_block_directive_is_host_coerced_to_observed(monkeypatch, caplog):
    manager = PluginManager()
    context = _context(manager)
    context.register_hook("pre_tool_call", lambda **_kwargs: {"action": "block", "message": "must ignore"})
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        details = _get_pre_tool_call_directive_details("read_file", {"path": "README.md"})

    assert details.action == "observed"
    assert details.message is None
    assert details.modified_args is None
    assert resolve_pre_tool_block("read_file", {"path": "README.md"}) is None
    assert "observation-only" in caplog.text
    assert "directive" in caplog.text


def test_observation_only_modify_directive_cannot_rewrite_args(monkeypatch, caplog):
    manager = PluginManager()
    context = _context(manager)
    context.register_hook(
        "pre_tool_call", lambda **_kwargs: {"action": "modify", "args": {"path": "SAFE.md"}}
    )
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        details = _get_pre_tool_call_directive_details("read_file", {"path": "README.md"})

    assert details.modified_args is None
    assert details.action == "observed"
    warnings = [r for r in caplog.records if "observation-only" in r.getMessage()]
    assert len(warnings) == 1


def test_observation_only_approve_directive_cannot_request_approval(monkeypatch, caplog):
    manager = PluginManager()
    context = _context(manager)
    context.register_hook(
        "pre_tool_call", lambda **_kwargs: {"action": "approve", "message": "must ignore"}
    )
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    def forbidden_approval(*_args, **_kwargs):
        raise AssertionError("observation-only directive reached request_tool_approval")

    monkeypatch.setattr("tools.approval.request_tool_approval", forbidden_approval)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        block = resolve_pre_tool_block("read_file", {"path": "README.md"})
        details = _get_pre_tool_call_directive_details("read_file", {"path": "README.md"})

    assert block is None
    assert details.action == "observed"
    assert details.message is None
    warnings = [r for r in caplog.records if "observation-only" in r.getMessage()]
    assert len(warnings) == 1


def test_observation_only_pre_tool_timeout_returns_no_block_or_message(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    manager = PluginManager()
    context = _context(manager)
    release = threading.Event()

    def hung_observer(**_kwargs):
        release.wait(timeout=2.0)
        return {"action": "observed"}

    context.register_hook("pre_tool_call", hung_observer)
    try:
        result = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-a")
    finally:
        release.set()

    assert result == []


def test_observation_only_timeout_records_structured_hook_timeout(monkeypatch, caplog):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    manager = PluginManager()
    context = _context(manager)
    release = threading.Event()

    def hung_observer(**_kwargs):
        release.wait(timeout=2.0)

    context.register_hook("pre_tool_call", hung_observer)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-a")
    finally:
        release.set()

    events = [record for record in caplog.records if getattr(record, "event", None) == "hook_timeout"]
    assert len(events) == 1
    event = events[0]
    assert getattr(event, "schema", None) == "hermes.observer.v1"
    assert getattr(event, "hook", None) == "pre_tool_call"
    assert getattr(event, "plugin", None) == "observer"
    assert getattr(event, "callback", None) == "hung_observer"
    assert getattr(event, "timeout", None) == 0.01
    assert "timed out after 0.01s — skipping" in caplog.text


def test_observation_only_timeout_dispatches_reserved_event(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    manager = PluginManager()
    context = _context(manager)
    release = threading.Event()
    payloads = []
    manager._subscribe_event("test", "hermes:hook_timeout", lambda **payload: payloads.append(payload))

    def hung_observer(**_kwargs):
        release.wait(timeout=2.0)

    context.register_hook("pre_tool_call", hung_observer)
    try:
        result = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-event")
    finally:
        release.set()
    assert manager._wait_for_event_dispatch(timeout=2.0)

    assert payloads == [{
        "schema": "hermes.observer.v1",
        "hook": "pre_tool_call",
        "plugin": "observer",
        "callback": "hung_observer",
        "timeout": 0.01,
    }]
    assert result == []


def test_observation_only_skips_post_timeout_suppression_after_worker_exits(monkeypatch):
    timeout = {"seconds": 0.01}
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: timeout["seconds"])
    manager = PluginManager()
    context = _context(manager)
    release = threading.Event()
    calls = []

    def observer(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            release.wait(timeout=2.0)
        return {"action": "observed"}

    context.register_hook("pre_tool_call", observer)
    first = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-a")
    release.set()

    deadline = time.monotonic() + 2.0
    worker_released = False
    while time.monotonic() < deadline:
        with manager._hook_timeout_lock:
            worker_released = manager._hook_abandoned == {}
        if worker_released:
            break
        time.sleep(0.005)
    assert worker_released, "timed-out worker did not release its liveness token"

    timeout["seconds"] = 0.2
    second = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-b")

    assert second == [{"action": "observed"}]
    assert calls == [1, 1]
    assert first == []


def test_policy_timeout_still_enters_post_timeout_suppression_window(monkeypatch):
    timeout = {"seconds": 0.01}
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: timeout["seconds"])
    manager = PluginManager()
    context = _context(manager, observation_only=False, name="policy")
    release = threading.Event()
    calls = []

    def policy(**_kwargs):
        calls.append(1)
        if len(calls) == 1:
            release.wait(timeout=2.0)

    context.register_hook("pre_tool_call", policy)
    first = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-a")
    release.set()

    deadline = time.monotonic() + 2.0
    worker_released = False
    while time.monotonic() < deadline:
        with manager._hook_timeout_lock:
            worker_released = manager._hook_abandoned == {}
        if worker_released:
            break
        time.sleep(0.005)
    assert worker_released, "timed-out policy worker did not release its liveness token"

    timeout["seconds"] = 0.2
    second = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-b")
    expected_block = [{
        "action": "block",
        "message": "pre_tool_call plugin callback timed out or is still running",
    }]

    assert first == expected_block
    assert second == expected_block
    assert calls == [1]


def test_live_worker_guard_skips_without_block_or_replacement_thread(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.01)
    manager = PluginManager()
    manager._hook_timeout_suppression_seconds = 0.0  # isolate liveness from time suppression
    context = _context(manager)
    release = threading.Event()
    started = threading.Event()
    starts = []

    def hung_observer(**_kwargs):
        starts.append(threading.current_thread().name)
        started.set()
        release.wait(timeout=2.0)
        return {"action": "observed"}

    context.register_hook("pre_tool_call", hung_observer)
    try:
        first = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-a")
        assert started.wait(timeout=1.0)
        second = manager.invoke_hook("pre_tool_call", tool_name="read_file", tool_call_id="call-b")
        assert len(starts) == 1
        assert second == []
        assert first == []
    finally:
        release.set()


def test_observed_is_an_explicit_noop_directive_and_block_shim_returns_none(monkeypatch):
    manager = PluginManager()
    context = _context(manager)
    context.register_hook("pre_tool_call", lambda **_kwargs: {"action": "observed"})
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    directive, message = get_pre_tool_call_directive("read_file", {"path": "README.md"})

    assert directive == "observed"
    assert message is None
    assert get_pre_tool_call_block_message("read_file", {"path": "README.md"}) is None


def test_observed_does_not_outrank_later_policy_block(monkeypatch):
    manager = PluginManager()
    _context(manager).register_hook("pre_tool_call", lambda **_kwargs: {"action": "observed"})
    _context(manager, observation_only=False, name="policy").register_hook(
        "pre_tool_call", lambda **_kwargs: {"action": "block", "message": "policy denied"}
    )
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    directive, message = get_pre_tool_call_directive("read_file", {"path": "README.md"})

    assert directive == "block"
    assert message == "policy denied"


def test_observed_preserves_policy_modify_accumulation(monkeypatch):
    manager = PluginManager()
    _context(manager).register_hook("pre_tool_call", lambda **_kwargs: {"action": "observed"})
    _context(manager, observation_only=False, name="policy").register_hook(
        "pre_tool_call", lambda **_kwargs: {"action": "modify", "args": {"path": "SAFE.md"}}
    )
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    details = _get_pre_tool_call_directive_details("read_file", {"path": "README.md", "line": 1})

    assert details.action == "observed"
    assert details.message is None
    assert details.modified_args == {"path": "SAFE.md", "line": 1}


def test_mixed_owner_same_callback_remains_policy_capable(monkeypatch):
    manager = PluginManager()

    def shared(**_kwargs):
        return {"action": "block", "message": "policy denied"}

    _context(manager).register_hook("pre_tool_call", shared)
    _context(manager, observation_only=False, name="policy").register_hook("pre_tool_call", shared)
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    assert resolve_pre_tool_block("read_file", {"path": "README.md"}) == "policy denied"


def test_dispose_releases_registration_traits():
    manager = PluginManager()
    handle = _context(manager).register_hook("pre_tool_call", lambda **_kwargs: {"action": "observed"})

    assert manager._hook_registration_traits
    handle.dispose()

    assert manager._hook_registration_traits == {}
