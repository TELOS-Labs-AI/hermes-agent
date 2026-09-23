from types import SimpleNamespace

from agent import conversation_compression as cc


def test_replay_compression_warning_suppresses_duplicate_gateway_text(monkeypatch):
    monkeypatch.setattr(cc, "_REPLAYED_COMPRESSION_WARNINGS", set())
    sent = []

    def callback(kind, message):
        sent.append((kind, message))

    first = SimpleNamespace(_compression_warning="notice", status_callback=callback)
    second = SimpleNamespace(_compression_warning="notice", status_callback=callback)

    cc.replay_compression_warning(first)
    cc.replay_compression_warning(second)

    assert sent == [("lifecycle", "notice")]
    # The process-wide ledger is what suppresses the repeat; the per-agent slot is
    # cleared by the caller (agent/turn_context.py "send once"), not here.
    assert "notice" in cc._REPLAYED_COMPRESSION_WARNINGS


def test_replay_compression_warning_allows_distinct_warning_texts(monkeypatch):
    monkeypatch.setattr(cc, "_REPLAYED_COMPRESSION_WARNINGS", set())
    sent = []
    agent_a = SimpleNamespace(_compression_warning="notice-a", status_callback=lambda kind, msg: sent.append((kind, msg)))
    agent_b = SimpleNamespace(_compression_warning="notice-b", status_callback=lambda kind, msg: sent.append((kind, msg)))

    cc.replay_compression_warning(agent_a)
    cc.replay_compression_warning(agent_b)

    assert sent == [("lifecycle", "notice-a"), ("lifecycle", "notice-b")]
