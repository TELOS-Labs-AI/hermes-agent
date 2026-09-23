"""Redaction and at-rest hygiene controls for the bundled telos_passive_scorer plugin.

Every credential-shaped string below is a FAKE canary invented for this file. Nothing here is a
real secret, and each canary is distinctive enough that a `not in` assertion over the whole
serialized receipt is a meaningful control rather than a coincidence.

Covers receipt-redaction findings from internal security review.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# --- FAKE canaries -------------------------------------------------------------------
FAKE_EXPORT_VALUE = "FAKEcanaryEXPORTvalue1234567890"
FAKE_EXPORT_LINE = f"export OPENAI_API_KEY={FAKE_EXPORT_VALUE}"
FAKE_DB_USER = "fakecanaryuser"
FAKE_DB_PASSWORD = "FAKEcanaryDBPASSWORD9911"
FAKE_DB_URL = f"postgres://{FAKE_DB_USER}:{FAKE_DB_PASSWORD}@db.example.invalid:5432/app"
FAKE_GITHUB_PAT = "ghp_FAKEcanary0123456789abcdefghijklmnop"  # gitleaks:allow (fake canary)
FAKE_OPENAI_KEY = "sk-proj-FAKEcanary0123456789abcdefghijkl"
FAKE_ANTHROPIC_KEY = "sk-ant-FAKEcanary0123456789abcdefghijkl"  # gitleaks:allow (fake canary)
FAKE_AWS_KEY_ID = "AKIAFAKECANARY000000"  # gitleaks:allow (fake canary)
FAKE_SLACK_TOKEN = "xoxb-000000000000-000000000000-FAKEcanarySlackTok00"
FAKE_TELEGRAM_TOKEN = "1234567890:FAKEcanaryTelegramBotToken000000000"
FAKE_PEM_BODY_1 = "FAKEcanaryPEMBODYLINEONE0000000000"
FAKE_PEM_BODY_2 = "FAKEcanaryPEMBODYLINETWO0000000000"
FAKE_PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"  # gitleaks:allow (fake canary, no key material)
    f"{FAKE_PEM_BODY_1}\n{FAKE_PEM_BODY_2}\n"
    "-----END RSA PRIVATE KEY-----"
)
FAKE_ENV_RESULT = (
    f"OPENAI_API_KEY={FAKE_EXPORT_VALUE}\n"
    f"DB_PASSWORD={FAKE_DB_PASSWORD}\n"
    f"GITHUB_TOKEN={FAKE_GITHUB_PAT}\n"
)

ORDINARY_TEXT = "review the risk-based task-list on desk-2 before the whisk-broom audit"


class _FakeContext:
    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}

    def register_hook(self, name: str, callback) -> None:
        self.hooks[name] = callback


def _fresh_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    sys.modules.pop("plugins.observability.telos_passive_scorer", None)
    plugin = importlib.import_module("plugins.observability.telos_passive_scorer")
    plugin.reset_for_tests()
    monkeypatch.setattr(plugin, "_SCORER", None)
    monkeypatch.setattr(plugin, "_SCORER_LOADED", True)
    return plugin


def _receipt_path(tmp_path: Path) -> Path:
    return tmp_path / "hermes_home" / "telos" / "operational_receipts" / "tool_calls.jsonl"


def _receipt_lines(tmp_path: Path) -> list[str]:
    return [l for l in _receipt_path(tmp_path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_call(plugin, *, tool_name="api_call", args=None, result=None, error=None, call_id="c1"):
    plugin.on_pre_tool_call(tool_name=tool_name, args=args or {}, tool_call_id=call_id)
    plugin.on_post_tool_call(
        tool_name=tool_name,
        args=args or {},
        result=result,
        error=error,
        tool_call_id=call_id,
        status="ok",
        duration_ms=1.0,
    )


def _receipt(tmp_path: Path) -> tuple[dict, str]:
    """Return the single receipt as a dict AND as its raw serialized line.

    The raw line is what the control asserts over: a secret that escaped into any field, however
    deeply nested or renamed, still shows up in the bytes that were written to disk.
    """
    [line] = _receipt_lines(tmp_path)
    return json.loads(line), line


# --- G1 ------------------------------------------------------------------------------
def test_g1_export_env_assignment_is_redacted(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, tool_name="run_shell", args={"command": FAKE_EXPORT_LINE})

    _, line = _receipt(tmp_path)
    assert FAKE_EXPORT_VALUE not in line, "export KEY=<value> survived redaction"
    assert "[REDACTED]" in line


# --- G2 ------------------------------------------------------------------------------
def test_g2_url_userinfo_is_redacted(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"dsn": FAKE_DB_URL, "note": f"connect via {FAKE_DB_URL}"})

    _, line = _receipt(tmp_path)
    assert FAKE_DB_PASSWORD not in line, "URL userinfo password survived redaction"
    assert FAKE_DB_USER not in line, "URL userinfo username survived redaction"
    assert "db.example.invalid" in line, "host was destroyed; only the userinfo should be removed"


# --- G3 ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "canary",
    [
        pytest.param(FAKE_GITHUB_PAT, id="github_pat"),
        pytest.param(FAKE_OPENAI_KEY, id="openai_sk_proj"),
        pytest.param(FAKE_ANTHROPIC_KEY, id="anthropic_sk_ant"),
        pytest.param(FAKE_AWS_KEY_ID, id="aws_access_key_id"),
        pytest.param(FAKE_SLACK_TOKEN, id="slack_xoxb"),
        pytest.param(FAKE_TELEGRAM_TOKEN, id="telegram_bot_token"),
    ],
)
def test_g3_bare_vendor_tokens_are_redacted(monkeypatch, tmp_path, canary):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    # Bare: no `token=` / `key:` keyword anywhere, so only a shape-based pattern can catch it.
    _run_call(plugin, args={"body": f"here it is {canary} end", "items": [{"v": canary}]})

    _, line = _receipt(tmp_path)
    assert canary not in line, f"bare vendor token {canary[:6]}… survived redaction"


# --- G4 ------------------------------------------------------------------------------
def test_g4_pem_private_key_block_is_redacted(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"content": f"writing:\n{FAKE_PEM_BLOCK}\ndone"})

    _, line = _receipt(tmp_path)
    assert FAKE_PEM_BODY_1 not in line, "PEM body line 1 survived redaction"
    assert FAKE_PEM_BODY_2 not in line, "PEM body line 2 survived redaction"
    assert "BEGIN RSA PRIVATE KEY" not in line, "PEM header survived redaction"


# --- G5 ------------------------------------------------------------------------------
def test_g5_marker_does_not_preserve_length(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    short = plugin._sanitize_text("password=abcd")
    long = plugin._sanitize_text("password=" + "z" * 400)

    assert short == long, "redacted output differs by secret length; the marker leaks length"
    assert "z" * 20 not in long


# --- G6 ------------------------------------------------------------------------------
def test_g6_secrets_redacted_in_nested_args_and_error_message(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(
        plugin,
        args={"outer": {"inner": [{"deep": FAKE_EXPORT_LINE}, FAKE_GITHUB_PAT]}},
        error=SimpleNamespace(message=f"auth failed using {FAKE_DB_URL} and {FAKE_GITHUB_PAT}"),
    )

    receipt, line = _receipt(tmp_path)
    assert FAKE_EXPORT_VALUE not in line, "nested arg secret survived redaction"
    assert FAKE_GITHUB_PAT not in line, "nested / error-message token survived redaction"
    assert FAKE_DB_PASSWORD not in line, "error-message URL password survived redaction"
    assert receipt["post"]["error"] is not None, "the error itself must still be recorded"


# --- G7 ------------------------------------------------------------------------------
def test_g7_result_persists_shape_not_content(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, tool_name="read_file", args={"path": "/tmp/fake.env"}, result=FAKE_ENV_RESULT)

    receipt, line = _receipt(tmp_path)
    meta = receipt["post"]["result_metadata"]
    assert FAKE_EXPORT_VALUE not in line, ".env result value reached the receipt"
    assert FAKE_DB_PASSWORD not in line, ".env result value reached the receipt"
    assert FAKE_GITHUB_PAT not in line, ".env result value reached the receipt"
    assert meta["sha256"] == hashlib.sha256(FAKE_ENV_RESULT.encode("utf-8")).hexdigest()
    assert meta["length"] == len(FAKE_ENV_RESULT)
    assert meta["line_count"] == FAKE_ENV_RESULT.count("\n") + 1
    assert meta["type"] == "str"
    assert meta["content_persisted"] is False


def test_g7b_dict_result_persists_shape_not_content(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={}, result={"stdout": FAKE_ENV_RESULT, "ok": True})

    receipt, line = _receipt(tmp_path)
    meta = receipt["post"]["result_metadata"]
    assert FAKE_EXPORT_VALUE not in line
    assert "stdout" not in line, "result key names still expose result structure verbatim"
    assert meta["type"] == "dict"
    assert len(meta["sha256"]) == 64


# --- G8 ------------------------------------------------------------------------------
def test_g8_new_receipt_file_is_0600(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"a": 1})

    mode = stat.S_IMODE(_receipt_path(tmp_path).stat().st_mode)
    assert mode == 0o600, f"new receipt file created {oct(mode)}, expected 0o600"


# --- G9 ------------------------------------------------------------------------------
def test_g9_ordinary_text_is_not_mangled(monkeypatch, tmp_path):
    """The over-fire trap: `risk-based`, `task-list`, `desk-2` and `whisk-broom` all embed `sk-`."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"note": ORDINARY_TEXT, "nested": {"note": ORDINARY_TEXT}})

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["note"] == ORDINARY_TEXT
    assert receipt["args"]["nested"]["note"] == ORDINARY_TEXT


def test_g9b_plain_urls_and_identifiers_are_not_mangled(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    plain = "https://example.invalid/api/v2?page=2"
    _run_call(plugin, args={"url": plain, "id": "desk-2", "label": "risk-based"})

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["url"] == plain
    assert receipt["args"]["id"] == "desk-2"
    assert receipt["args"]["label"] == "risk-based"


# --- G10 (local guard; the full regression set runs as three suites) ------------------
def test_g10_degraded_scorer_still_returns_the_non_gating_observed_directive(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    ctx = _FakeContext()
    plugin.register(ctx)

    out = plugin.on_pre_tool_call(tool_name="run_shell", args={"command": FAKE_EXPORT_LINE}, tool_call_id="obs-1")
    assert out["action"] == "observed"
    assert out["action"] not in {"block", "approve", "modify"}
    assert out["scoring_status"] == "scorer_missing"


# =====================================================================================
# Review round 1
# F1: assignments whose NAME ends in KEY (or equals KEY), plus TOKEN/SECRET/PASSWORD/
#     PASSWD/CREDENTIAL suffixes, with and without `export`, in EVERY text channel.
# F3 (design decision): absolute-path masking is DELIBERATE and is pinned below; the
#     "paths survive" requirement means RELATIVE paths.
# =====================================================================================

FAKE_EXPORT_XKEY_VALUE = "FAKEEXPORTXKEY000000000"
FAKE_BARE_KEY_VALUE = "FAKEBAREKEYVALUE000000000"


def _channels(canary_line: str) -> dict:
    """The same credential-bearing line placed in every text channel a receipt can carry."""
    return {
        "direct": canary_line,
        "nested": {"list": [canary_line], "dict": {"deep": canary_line}},
        "json_encoded": json.dumps({"payload": canary_line}),
    }


@pytest.mark.parametrize(
    "line, value",
    [
        pytest.param(f"export X_KEY={FAKE_EXPORT_XKEY_VALUE}", FAKE_EXPORT_XKEY_VALUE, id="export_X_KEY"),
        pytest.param(f"KEY={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="bare_KEY"),
        pytest.param(f"export KEY={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="export_bare_KEY"),
        pytest.param(f"X_KEY={FAKE_EXPORT_XKEY_VALUE}", FAKE_EXPORT_XKEY_VALUE, id="X_KEY_no_export"),
        pytest.param(f"MY-KEY={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="hyphenated_MY_KEY"),
        pytest.param(f"SVC_TOKEN={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="suffix_TOKEN"),
        pytest.param(f"SVC_SECRET={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="suffix_SECRET"),
        pytest.param(f"SVC_PASSWORD={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="suffix_PASSWORD"),
        pytest.param(f"SVC_PASSWD={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="suffix_PASSWD"),
        pytest.param(f"SVC_CREDENTIAL={FAKE_BARE_KEY_VALUE}", FAKE_BARE_KEY_VALUE, id="suffix_CREDENTIAL"),
        pytest.param(f'"KEY": "{FAKE_BARE_KEY_VALUE}"', FAKE_BARE_KEY_VALUE, id="json_style_KEY_colon"),
    ],
)
def test_f1_key_suffix_assignments_are_redacted_in_every_channel(monkeypatch, tmp_path, line, value):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(
        plugin,
        args=_channels(line),
        error=SimpleNamespace(message=f"startup failed: {line}"),
    )

    _, receipt_line = _receipt(tmp_path)
    assert value not in receipt_line, f"{line.split('=')[0]} assignment survived redaction"


def test_f1_key_suffix_rule_does_not_over_fire_on_ordinary_identifiers(monkeypatch, tmp_path):
    """`monkey`, `keyboard`, `turkey_soup` all contain `key` but none END in it after a separator.

    Without the `[_-]`-or-start boundary on the suffix rule, every one of these would be redacted.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    benign = {
        "a": "monkey=3",
        "b": "keyboard=qwerty",
        "c": "turkey_soup=ok",
        "d": "whiskey-sour=2",
        "e": "keystone=arch",
    }
    _run_call(plugin, args=benign)

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"] == benign


# --- F3 decision: absolute paths are masked ON PURPOSE; relative paths survive ----------
def test_f3_absolute_paths_are_masked_deliberately(monkeypatch, tmp_path):
    """Design decision: absolute-path masking is intended behaviour, not over-masking.

    Pinned here so a later reviewer reading a 'paths must survive' requirement cannot mistake
    this for a defect, and so nobody silently removes it.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={
        "bare": "/Users/fake/project/src/scorer.py",
        "trailing": "see /Users/fake/project/src/scorer.py for detail",
    })

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["bare"] == "[REDACTED_PATH]"
    assert "/Users/fake" not in receipt["args"]["trailing"]
    assert "[REDACTED_PATH]" in receipt["args"]["trailing"]
    # Pre-existing and deliberate: _ABS_PATH_WITH_SPACES_RE exists to cover paths that CONTAIN
    # spaces (e.g. "Project Files"), so it consumes trailing words up to a terminator. "for detail"
    # is therefore eaten too. Lossy, pre-dates this branch, pinned here so the behaviour is a
    # recorded decision rather than a surprise at the next review.
    assert receipt["args"]["trailing"] == "see [REDACTED_PATH]"


def test_f3_relative_paths_and_opaque_identifiers_survive_unchanged(monkeypatch, tmp_path):
    """The other half of the decision: only ABSOLUTE paths are masked."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    benign = {
        "rel": "src/plugins/observability/scorer.py",
        "dotrel": "./tests/plugins/test_x.py",
        "digest": "a" * 64,
        "uuid": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "git_sha": "a5f4588c1cb5dd7a7e3664956f9d189680204ddb",
        "short_sha": "06065b7868",
    }
    _run_call(plugin, args=benign)

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"] == benign


FAKE_MIDNAME_VALUE = "FAKEMIDNAMEKEYVALUE00000"


def test_f1b_keyword_inside_a_longer_name_is_redacted(monkeypatch, tmp_path):
    """`MY_API_KEY_OLD` ends in OLD, so the suffix rule cannot see it.

    Only the enumerated-word rule's identifier-aware boundary catches this, which keeps mutation
    M1 (reverting that boundary to `\\b`) a meaningful sabotage rather than one the newer
    assignment rule silently covers for.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"cmd": f"export MY_API_KEY_OLD={FAKE_MIDNAME_VALUE}"})

    _, line = _receipt(tmp_path)
    assert FAKE_MIDNAME_VALUE not in line


# =====================================================================================
# Review round 2
# R2-F1: a quoted value containing whitespace leaked its second word.
# R2-F2 (design decision): credential words match ONLY as whole name segments, split on
#        `_`, `-`, or a case boundary. Plural `keys` is not a secret. No outcome may
#        depend on the LENGTH of the value.
# Scorer fidelity now rides on the no-over-mask half: content-aware scoring
# reads these values, so a wrongly masked value is a governance defect, not cosmetics.
# =====================================================================================

FAKE_QUOTED_A = "FAKEQUOTEDPARTA"
FAKE_QUOTED_B = "FAKEQUOTEDPARTB"


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(f'KEY="{FAKE_QUOTED_A} {FAKE_QUOTED_B}"', id="double_quoted_with_space"),
        pytest.param(f"KEY='{FAKE_QUOTED_A} {FAKE_QUOTED_B}'", id="single_quoted_with_space"),
        pytest.param(f'API_KEY="{FAKE_QUOTED_A} {FAKE_QUOTED_B}"', id="api_key_double_quoted"),
        pytest.param(f'export KEY="{FAKE_QUOTED_A} {FAKE_QUOTED_B}"', id="export_double_quoted"),
        pytest.param(f'PASSWORD="{FAKE_QUOTED_A} {FAKE_QUOTED_B}"', id="password_double_quoted"),
        pytest.param(
            f'KEY="{FAKE_QUOTED_A} {FAKE_QUOTED_B}" NEXT=ok', id="quoted_then_more_text"),
    ],
)
def test_r2f1_quoted_value_containing_whitespace_is_fully_redacted(monkeypatch, tmp_path, line):
    """The value must be consumed THROUGH its matching closing quote, not up to the first space."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(
        plugin,
        args=_channels(line),
        error=SimpleNamespace(message=f"startup failed: {line}"),
    )

    _, receipt_line = _receipt(tmp_path)
    assert FAKE_QUOTED_A not in receipt_line, "first word of the quoted value survived"
    assert FAKE_QUOTED_B not in receipt_line, "second word of the quoted value survived"


def test_r2f1_text_after_a_quoted_secret_is_not_swallowed(monkeypatch, tmp_path):
    """Consuming through the closing quote must stop THERE; the rest of the line survives."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"cmd": f'KEY="{FAKE_QUOTED_A} {FAKE_QUOTED_B}" TOKENIZER=bpe done'})

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["cmd"] == 'KEY=[REDACTED] TOKENIZER=bpe done'


# --- R2-F2: whole-segment matching, both directions ----------------------------------
_MUST_SURVIVE = {
    "tokenizer_long": "TOKENIZER=bpe",
    "tokenizer_short": "TOKENIZER=bp",
    "secretary_long": "SECRETARY=xyz",
    "secretary_short": "SECRETARY=x",
    "keys_plural": "keys=3",
    "monkey": "MONKEY=1",
    "keyboard_layout": "KEYBOARD_LAYOUT=us",
    "tokenizer_lower": "tokenizer=bpe",
    "keystone": "keystone=arch",
    "whiskey": "whiskey-sour=2",  # gitleaks:allow (their generic rule matches `key` INSIDE `whiskey` -- the exact over-fire this round fixes)
}

_MUST_MASK = {
    "api_key": ("API_KEY=", "FAKEAPIKEYVALUE001"),
    "x_token": ("X_TOKEN=", "FAKEXTOKENVALUE001"),
    "db_password": ("DB_PASSWORD=", "FAKEDBPASSWORDVAL1"),
    "bare_secret": ("SECRET=", "FAKEBARESECRETVAL1"),
    "camel_api_key": ("apiKey=", "FAKECAMELAPIKEYVA1"),
    "camel_my_token": ("myAuthToken=", "FAKECAMELTOKENVAL1"),
    "short_value": ("API_KEY=", "zZq"),  # short, and not hex (a hex canary collides with receipt hashes)
}


@pytest.mark.parametrize("label", sorted(_MUST_SURVIVE))
def test_r2f2_ordinary_assignments_survive_unchanged(monkeypatch, tmp_path, label):
    """`TOKENIZER`, `SECRETARY`, `keys`, `MONKEY`, `KEYBOARD_LAYOUT` are NOT credential names.

    Masking them destroys the value the content-aware scorer reads, so each of these is a
    governance defect, not a cosmetic one.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    text = _MUST_SURVIVE[label]
    _run_call(plugin, args={"cmd": text, "nested": {"deep": [text]}})

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["cmd"] == text
    assert receipt["args"]["nested"]["deep"] == [text]


@pytest.mark.parametrize("label", sorted(_MUST_MASK))
def test_r2f2_credential_name_segments_are_masked(monkeypatch, tmp_path, label):
    prefix, value = _MUST_MASK[label]
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"cmd": f"{prefix}{value}", "nested": {"deep": [f"{prefix}{value}"]}})

    _, line = _receipt(tmp_path)
    assert value not in line, f"{prefix} value survived redaction"


def test_r2f2_outcome_does_not_depend_on_value_length(monkeypatch, tmp_path):
    """`TOKENIZER=bp` survived while `TOKENIZER=bpe` was masked — a 3-char minimum leaking through.

    Whether a name is a credential name cannot depend on how long its value happens to be.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    ordinary = {f"v{n}": f"TOKENIZER={'b' * n}" for n in range(1, 9)}
    secret = {f"s{n}": f"API_KEY={'q' * n}" for n in range(1, 9)}
    _run_call(plugin, args={**ordinary, **secret})

    receipt, line = _receipt(tmp_path)
    for n in range(1, 9):
        assert receipt["args"][f"v{n}"] == f"TOKENIZER={'b' * n}", "ordinary name masked at some length"
        assert receipt["args"][f"s{n}"] == "API_KEY=[REDACTED]", "credential name missed at some length"
    assert "qqq" not in line


def test_r2f2_dict_keys_use_the_same_segment_rule(monkeypatch, tmp_path):
    """The dict-key rule was a bare substring search, so a `tokenizer` ARG KEY masked its value."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"tokenizer": "bpe", "secretary": "alice", "keys": 3, "api_key": "FAKEDICTKEYVAL001"})

    receipt, line = _receipt(tmp_path)
    assert receipt["args"]["tokenizer"] == "bpe"
    assert receipt["args"]["secretary"] == "alice"
    assert receipt["args"]["keys"] == 3
    assert receipt["args"]["api_key"] == "[REDACTED]"
    assert "FAKEDICTKEYVAL001" not in line


# =====================================================================================
# Review round 3 — plural credential names, superseding "plural keys is not a secret":
#   Plural credential segments (keys, tokens, secrets, passwords, credentials) are MASKED
#   wherever they appear in a COMPOUND name. The ONLY exemption is a name that is a single
#   bare plural segment AND a purely numeric value. `keys=3` survives; `keys=abc` does not.
# =====================================================================================

_PLURAL_MUST_MASK = {
    # The motivating leak example. The value is deliberately NOT sk-shaped in the second case, so
    # the control proves the NAME rule rather than being rescued by the bare-vendor-token rule.
    "openai_api_keys_sk": ("OPENAI_API_KEYS=", ["sk-FAKEpluralkeys0123456789ab"]),
    "api_keys_comma_list": ("API_KEYS=", ["FAKEPLURALONE001", "FAKEPLURALTWO002"]),
    "db_passwords_list": ("DB_PASSWORDS=", ["FAKEPWDONE001", "FAKEPWDTWO002"]),
    "service_tokens": ("SERVICE_TOKENS=", ["FAKESVCTOKEN001"]),
    "secrets_compound": ("APP_SECRETS=", ["FAKEAPPSECRET001", "FAKEAPPSECRET002"]),
    "credentials_compound": ("USER_CREDENTIALS=", ["FAKEUSERCRED001"]),
    "bare_plural_nonnumeric": ("keys=", ["FAKEBAREPLURALABC1"]),
    "bare_plural_tokens_nonnumeric": ("tokens=", ["FAKEBAREPLURALABC2"]),
}


@pytest.mark.parametrize("label", sorted(_PLURAL_MUST_MASK))
def test_r3_plural_credential_names_are_masked(monkeypatch, tmp_path, label):
    prefix, values = _PLURAL_MUST_MASK[label]
    line = prefix + ",".join(values)
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args=_channels(line), error=SimpleNamespace(message=f"boot: {line}"))

    _, receipt_line = _receipt(tmp_path)
    for value in values:
        assert value not in receipt_line, f"{prefix} value {value!r} survived redaction"


def test_r3_comma_separated_plural_value_is_masked_whole(monkeypatch, tmp_path):
    """The motivating leak: a list value must not be truncated at the first comma.

    The value scanner stops at `,` for ordinary assignments, so `API_KEYS=a,b` masked only `a`
    and the SECOND credential reached the receipt.
    """
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={"cmd": "API_KEYS=FAKELISTONE001,FAKELISTTWO002,FAKELISTTHREE03 next=ok"})

    receipt, line = _receipt(tmp_path)
    assert "FAKELISTONE001" not in line
    assert "FAKELISTTWO002" not in line
    assert "FAKELISTTHREE03" not in line
    # and the scan still stops at whitespace, so the rest of the line survives
    assert receipt["args"]["cmd"] == "API_KEYS=[REDACTED] next=ok"


_PLURAL_MUST_SURVIVE = {
    "keys_count": "keys=3",
    "tokens_count": "tokens=12",
    "tokens_upper": "TOKENS=12",
    "secrets_count": "secrets=0",
    "passwords_count": "PASSWORDS=42",
    "credentials_count": "credentials=7",
}


@pytest.mark.parametrize("label", sorted(_PLURAL_MUST_SURVIVE))
def test_r3_bare_plural_with_numeric_value_survives(monkeypatch, tmp_path, label):
    """The one exemption: a single bare plural segment carrying a count, not a credential."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    text = _PLURAL_MUST_SURVIVE[label]
    _run_call(plugin, args={"cmd": text, "nested": {"deep": [text]}})

    receipt, _ = _receipt(tmp_path)
    assert receipt["args"]["cmd"] == text
    assert receipt["args"]["nested"]["deep"] == [text]


def test_r3_exemption_is_narrow_in_both_directions(monkeypatch, tmp_path):
    """Bare plural + numeric survives; bare plural + anything else does not; compound never does."""
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={
        "survives": "keys=3",
        "nonnumeric": "keys=FAKENONNUMERICKEY1",
        "compound_numeric": "API_KEYS=3",
        "singular_numeric": "API_KEY=3",
    })

    receipt, line = _receipt(tmp_path)
    assert receipt["args"]["survives"] == "keys=3"
    assert "FAKENONNUMERICKEY1" not in line
    # A COMPOUND plural name is masked even when the value looks like a count: the exemption is
    # for a bare plural only.
    assert receipt["args"]["compound_numeric"] == "API_KEYS=[REDACTED]"
    assert receipt["args"]["singular_numeric"] == "API_KEY=[REDACTED]"


def test_r3_dict_key_plural_exemption_matches_the_text_rule(monkeypatch, tmp_path):
    plugin = _fresh_plugin(monkeypatch, tmp_path)
    _run_call(plugin, args={
        "keys": 3,
        "tokens": "12",
        "secrets": "FAKEDICTPLURALVAL1",
        "api_keys": "FAKEDICTCOMPOUND01",
    })

    receipt, line = _receipt(tmp_path)
    assert receipt["args"]["keys"] == 3
    assert receipt["args"]["tokens"] == "12"
    assert receipt["args"]["secrets"] == "[REDACTED]"
    assert receipt["args"]["api_keys"] == "[REDACTED]"
    assert "FAKEDICTPLURALVAL1" not in line
    assert "FAKEDICTCOMPOUND01" not in line
