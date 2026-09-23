"""Passive TELOS PA-scoring receipts for Hermes tool calls.

This public plugin deliberately does not hard-import closed TELOS governance
math.  It uses a guarded optional import adapter and records "unscored"
operational receipts when no scorer is installed or the scorer fails.
"""

from __future__ import annotations

import importlib
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "telos.operational.tool_call.v2"
CLOSED_CORE_MECHANISM = "guarded_optional_import"
DIMENSION_NAMES = ("purpose", "scope", "boundary", "tool", "chain", "risk")
DEFAULT_RECEIPT_RELATIVE_PATH = Path("telos") / "operational_receipts" / "tool_calls.jsonl"

_STATE_LOCK = threading.RLock()
_PENDING: dict[str, dict[str, Any]] = {}
_SCORER: Callable[..., Any] | None = None
_SCORER_LOADED = False
_SCORER_ERROR: str = ""

REDACTED_MARKER = "[REDACTED]"
REDACTED_PATH_MARKER = "[REDACTED_PATH]"
REDACTED_KEY_MARKER = "[REDACTED_PRIVATE_KEY]"

# A credential word counts ONLY as a WHOLE name segment,
# where segments are split on `_`, `-`, or a case boundary. `API_KEY`, `X_TOKEN`, `DB_PASSWORD`
# and `SECRET` are credential names; `TOKENIZER`, `SECRETARY`, `MONKEY` and `KEYBOARD_LAYOUT` are
# not. Plural `keys` is not a secret.
#
# This replaced two regexes that matched credential words as SUBSTRINGS of a longer identifier.
# That was not merely untidy: `TOKENIZER=bpe` was masked while `TOKENIZER=bp` survived, because
# the old rule also carried a three-character minimum on the VALUE. Whether a name is a credential
# name must not depend on how long its value happens to be. Over-masking is now a governance
# defect in its own right, because the content-aware scorer reads these values.
_NAME_SEGMENT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

_SECRET_NAME_SEGMENTS = frozenset(
    {
        "key",
        "apikey",
        "privatekey",
        "token",
        "tokens",
        "secret",
        "secrets",
        "password",
        "passwords",
        "passwd",
        "authorization",
        "credential",
        "credentials",
        "keys",
    }
)

# Plural credential names: "plural `keys` is not a secret" alone would be too
# broad, because `OPENAI_API_KEYS=sk-a,sk-b` leaks. Plural credential segments are MASKED wherever
# they appear in a COMPOUND name. The ONLY exemption is a name that is a single bare plural
# segment carrying a purely numeric value -- `keys=3` is a count, `keys=abc` is a credential.
_PLURAL_SECRET_SEGMENTS = frozenset({"keys", "tokens", "secrets", "passwords", "credentials"})

_NUMERIC_VALUE_RE = re.compile(r"^\d+(?:\.\d+)?$")

# name, then `:`/`=` (tolerating the quotes and backslashes of one JSON encoding) or bare
# whitespace, then an optional `bearer`. Only spaces and tabs, never a newline: a name at the end
# of one line must not capture the first word of the next.
_ASSIGN_HEAD_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Za-z][A-Za-z0-9_-]*)"
    r"(?:[\\'\"]{0,3}[ \t]*[:=][ \t]*|[ \t]+)"
    r"(?:[Bb]earer[ \t]+)?"
)

_VALUE_STOP_CHARS = frozenset(" \t\r\n,;")
# A plural credential name carries a LIST, so `,` and `;` are part of the value rather than
# terminators. `API_KEYS=a,b` previously masked only `a` and wrote `b` to the receipt.
_LIST_VALUE_STOP_CHARS = frozenset(" \t\r\n")


def _name_segments(name: str) -> list[str]:
    """Lowercased whole segments of an identifier, split on `_`, `-`, and case boundaries."""
    segments: list[str] = []
    for part in re.split(r"[_-]+", name):
        segments.extend(segment.lower() for segment in _NAME_SEGMENT_RE.findall(part))
    return segments


def _is_secret_name(name: str) -> bool:
    """True when any WHOLE segment of `name` is a credential word."""
    for segment in _name_segments(name):
        if segment in _SECRET_NAME_SEGMENTS:
            return True
    return False


def _has_plural_secret_segment(name: str) -> bool:
    return any(segment in _PLURAL_SECRET_SEGMENTS for segment in _name_segments(name))


def _is_bare_plural_name(name: str) -> bool:
    """True only when the WHOLE name is one plural credential segment (`keys`, `TOKENS`)."""
    segments = _name_segments(name)
    return len(segments) == 1 and segments[0] in _PLURAL_SECRET_SEGMENTS


def _is_count_value(text: str) -> bool:
    """A purely numeric value -- the only thing a bare plural name is allowed to keep."""
    return bool(_NUMERIC_VALUE_RE.match(text.strip("\\\"'")))


def _is_secret_assignment(name: str, value_text: str) -> bool:
    """Whether NAME=VALUE is a credential assignment.

    The value participates in exactly one narrow case: a bare plural name with a count. Everywhere
    else this is a decision about the NAME alone, so no outcome depends on the value's LENGTH.
    """
    if not _is_secret_name(name):
        return False
    if _is_bare_plural_name(name) and _is_count_value(value_text):
        return False
    return True


def _assignment_value_end(text: str, start: int, *, list_value: bool = False) -> int:
    """Index just past the value beginning at `start`.

    An earlier rule stopped the value at the first whitespace, so
    `KEY="alpha beta"` became `KEY=[REDACTED] beta"` and leaked the second word. A quoted value is
    now consumed THROUGH its matching closing quote -- and stops there, so text after the quote
    survives.
    """
    length = len(text)
    cursor = start
    while cursor < length and text[cursor] == "\\":
        cursor += 1
    if cursor < length and text[cursor] in "\"'":
        quote = text[cursor]
        scan = cursor + 1
        while scan < length and text[scan] != "\n":
            if text[scan] == quote:
                return scan + 1
            scan += 1
        # Unterminated quote: consume to end of line. Leak-safe by choice -- a truncated secret
        # is still a secret, and over-masking here is recoverable where disclosure is not.
        newline = text.find("\n", start)
        return length if newline == -1 else newline
    stops = _LIST_VALUE_STOP_CHARS if list_value else _VALUE_STOP_CHARS
    scan = start
    while scan < length and text[scan] not in stops and text[scan] not in "\"'":
        scan += 1
    return scan


def _redact_assignments(text: str) -> str:
    out: list[str] = []
    pos = 0
    for match in _ASSIGN_HEAD_RE.finditer(text):
        name = match.group(1)
        if match.start() < pos or not _is_secret_name(name):
            continue
        value_start = match.end()
        value_end = _assignment_value_end(
            text, value_start, list_value=_has_plural_secret_segment(name)
        )
        if value_end <= value_start:
            continue
        if not _is_secret_assignment(name, text[value_start:value_end]):
            continue
        out.append(text[pos:value_start])
        out.append(REDACTED_MARKER)
        pos = value_end
    out.append(text[pos:])
    return "".join(out)


# scheme://user:password@host -> scheme://[REDACTED]@host. The host is kept; only userinfo goes.
_URL_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+:[^/\s@]*@")

# A PEM private key, header and body together. The non-greedy body stops at the first END line;
# a truncated block with no END line is caught by the second pattern.
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_PEM_HEADER_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*")

# Bare vendor tokens: no assignment and no keyword, so only the token's own shape can catch them.
# Every pattern refuses an identifier character immediately before it. That lookbehind is what
# keeps `risk-based`, `task-list` and `desk-2` from matching the `sk-` rule.
_BARE_TOKEN_RES = (
    re.compile(r"(?<![A-Za-z0-9_-])gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-|ant-(?:api\d{2}-)?)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_-])(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprse]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_-])\d{8,10}:[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
)
_ABS_PATH_WITH_SPACES_RE = re.compile(
    r"(?<![\w])(?:/(?:Users|private|tmp|var|etc|opt|Volumes|home)/.*?)"
    r"(?=(?:\s+(?:with|using|and|token|api[_-]?key|secret|password|authorization)\b)|[,;:'\")\]}]|$)",
    re.IGNORECASE,
)
_ABS_PATH_RE = re.compile(r"(?<![\w])(?:/(?:Users|private|tmp|var|etc|opt|Volumes|home)/[^\s,;:'\")\]}]+)")


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)


def on_pre_tool_call(**kwargs: Any) -> dict[str, Any] | None:
    """Classify and score a tool call, but never block execution."""
    try:
        correlation_id = _correlation_id(kwargs)
        started_at = time.time()
        classification = _classify_tool_call(kwargs.get("tool_name"), kwargs.get("args"))
        registry_record = _ensure_registry_record(kwargs, classification)
        verdict, scoring_status, score_errors, score_latency_ms = _score_tool_call(
            kwargs=kwargs,
            classification=classification,
        )
        state = {
            "started_at": started_at,
            "pre_kwargs": dict(kwargs),
            "classification": classification,
            "verdict": verdict,
            "scoring_status": scoring_status,
            "scoring_latency_ms": score_latency_ms,
            "errors": score_errors,
            "registry_record": registry_record,
        }
        with _STATE_LOCK:
            _PENDING[correlation_id] = state
        if scoring_status != "scored":
            return {
                # "observed" is the host resolver's explicit non-gating directive
                # (hermes_cli/plugins.py). "observe" was silently dropped as irrelevant.
                "action": "observed",
                "scoring_status": scoring_status,
                "message": "TELOS passive scorer degraded; receipt integrity marked unscored.",
            }
    except Exception as exc:
        logger.debug("TELOS passive scorer pre hook failed open: %s", exc, exc_info=True)
    return None


def on_post_tool_call(**kwargs: Any) -> None:
    """Finalize an operational receipt. Errors are debug-only and fail open."""
    try:
        correlation_id = _correlation_id(kwargs)
        with _STATE_LOCK:
            state = _PENDING.pop(correlation_id, None)

        if state is None:
            classification = _classify_tool_call(kwargs.get("tool_name"), kwargs.get("args"))
            verdict = _default_verdict()
            scoring_status = "unscored"
            score_latency_ms = 0.0
            errors: list[dict[str, str]] = []
            pre_kwargs: dict[str, Any] = {}
            started_at = None
            registry_record: dict[str, Any] = {}
        else:
            classification = state["classification"]
            verdict = state["verdict"]
            scoring_status = state["scoring_status"]
            score_latency_ms = state["scoring_latency_ms"]
            errors = list(state.get("errors") or [])
            pre_kwargs = state.get("pre_kwargs") or {}
            started_at = state.get("started_at")
            registry_record = state.get("registry_record") or {}

        receipt = _build_receipt(
            correlation_id=correlation_id,
            pre_kwargs=pre_kwargs,
            post_kwargs=kwargs,
            classification=classification,
            verdict=verdict,
            scoring_status=scoring_status,
            scoring_latency_ms=score_latency_ms,
            started_at=started_at,
            errors=errors,
            registry_record=registry_record,
        )
        _write_receipt(receipt)
    except Exception as exc:
        logger.debug("TELOS passive scorer post hook failed open: %s", exc, exc_info=True)
    return None


def _score_tool_call(*, kwargs: dict[str, Any], classification: dict[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, str]], float]:
    scorer = _get_scorer()
    errors: list[dict[str, str]] = []
    if scorer is None:
        if _SCORER_ERROR:
            errors.append({"source": "scorer", "message": _sanitize_text(_SCORER_ERROR)})
        return _default_verdict("No closed-core scorer available."), "scorer_missing", errors, 0.0

    start = time.perf_counter()
    try:
        raw = _call_scorer_with_timeout(
            scorer,
            {
                "tool_name": str(kwargs.get("tool_name") or ""),
                "action": _action_text(kwargs.get("tool_name"), kwargs.get("args")),
                "tool_args": _redact(kwargs.get("args") or {}),
                "classification": classification,
                "task_id": str(kwargs.get("task_id") or ""),
                "session_id": str(kwargs.get("session_id") or ""),
                "tool_call_id": str(kwargs.get("tool_call_id") or ""),
            },
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return _normalize_verdict(raw), "scored", errors, latency_ms
    except TimeoutError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        errors.append({"source": "scorer", "message": _sanitize_text(str(exc))})
        return _default_verdict(str(exc)), "scorer_timeout", errors, latency_ms
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        errors.append({"source": "scorer", "message": _sanitize_text(str(exc))})
        return _default_verdict(str(exc)), "scorer_error", errors, latency_ms


def _call_scorer_with_timeout(scorer: Callable[..., Any], call_kwargs: dict[str, Any]) -> Any:
    timeout_ms = _scorer_timeout_ms()
    result: dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = scorer(**call_kwargs)
        except Exception as exc:
            result["error"] = exc

    worker = threading.Thread(target=_target, name="telos-passive-scorer", daemon=True)
    worker.start()
    worker.join(timeout_ms / 1000)
    if worker.is_alive():
        raise TimeoutError(f"scorer timed out after {timeout_ms}ms")
    if "error" in result:
        raise result["error"]
    return result.get("value")


# The engine's cold start (ONNX load + first score) measured ~1.35 s on a reference Mac; 250 ms timed out every first call.
_DEFAULT_SCORER_TIMEOUT_MS = 2000


def _scorer_timeout_ms() -> int:
    raw = os.getenv("HERMES_TELOS_PASSIVE_SCORER_TIMEOUT_MS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 5000))
        except ValueError:
            return _DEFAULT_SCORER_TIMEOUT_MS
    return _DEFAULT_SCORER_TIMEOUT_MS


def _get_scorer() -> Callable[..., Any] | None:
    global _SCORER, _SCORER_LOADED, _SCORER_ERROR
    if _SCORER_LOADED:
        return _SCORER

    _SCORER_LOADED = True
    candidates = []
    env_ref = os.getenv("HERMES_TELOS_PASSIVE_SCORER", "").strip()
    if env_ref:
        candidates.append(env_ref)
    candidates.extend(
        [
            "telos_passive_scorer_core:score_tool_call",
            "telos_passive_scorer_core:score_action",
        ]
    )

    errors: list[str] = []
    for ref in candidates:
        try:
            module_name, func_name = _split_scorer_ref(ref)
            module = importlib.import_module(module_name)
            scorer = getattr(module, func_name)
            if callable(scorer):
                _SCORER = scorer
                _SCORER_ERROR = ""
                return _SCORER
            errors.append(f"{ref} is not callable")
        except Exception as exc:
            errors.append(f"{ref}: {exc}")

    _SCORER_ERROR = "; ".join(errors[:3])
    _SCORER = None
    return None


def _split_scorer_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        module_name, func_name = ref.split(":", 1)
    else:
        module_name, _, func_name = ref.rpartition(".")
    if not module_name or not func_name:
        raise ValueError(f"invalid scorer reference {ref!r}")
    return module_name, func_name


def _classify_tool_call(tool_name: Any, args: Any) -> dict[str, Any]:
    name = str(tool_name or "unknown")
    lowered = name.lower()
    args_dict = args if isinstance(args, dict) else {}
    if "computer" in lowered:
        group = "computer_use"
        risk = "high"
    elif any(part in lowered for part in ("write", "edit", "patch", "exec", "shell", "command")):
        group = "mutation"
        risk = "medium"
    elif any(part in lowered for part in ("read", "search", "list", "find")):
        group = "read"
        risk = "low"
    else:
        group = lowered.split("_", 1)[0] or "tool"
        risk = "medium" if args_dict else "low"
    return {
        "action_text": _action_text(name, args_dict),
        "tool_group": group,
        "risk_tier": risk,
        "is_cross_group": False,
    }


ACTION_TEXT_MAX_CHARS = 1000
_ACTION_VALUE_MAX_CHARS = 400


def _action_text(tool_name: Any, args: Any) -> str:
    """What the scorer embeds: the tool plus its REDACTED argument values.

    Key names alone made every call to a tool look identical to the scorer. Values pass through the
    same credential redaction as receipts; file paths keep only their basename, which carries the
    signal (README.md vs customer.key) without the directory layout.
    """
    name = tool_name or "unknown"
    if not isinstance(args, dict) or not args:
        return f"{name}()"
    parts = []
    for key in sorted(args, key=str)[:12]:
        value = _scoring_value(str(key), args[key])
        parts.append(f"{key}={value[:_ACTION_VALUE_MAX_CHARS]}")
    return f"{name}({', '.join(parts)})"[:ACTION_TEXT_MAX_CHARS]


def _scoring_value(key: str, value: Any) -> str:
    if _is_secret_assignment(key, _jsonable_scalar_text(value)):  # same predicate as receipts
        return REDACTED_MARKER
    if isinstance(value, str):
        if "path" in key.lower() or "file" in key.lower() or _looks_like_path(value):
            return _sanitize_text(re.split(r"[\\/]", value.rstrip("/\\"))[-1])
        return _sanitize_text(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_scoring_value(key, item) for item in value)
    if isinstance(value, dict):
        return _canonical_json(_redact(value))
    return str(value)


def _default_verdict(explanation: str | None = None) -> dict[str, Any]:
    return {
        "decision": "unscored",
        "allowed": False,
        "fidelity": None,
        "dimensions": {name: None for name in DIMENSION_NAMES},
        "explanation": _sanitize_text(
            explanation or "No closed-core scorer available; passive receipt emitted unscored."
        ),
    }


# Scorer provenance carried into every receipt so a reader can tell a placeholder PA hash or a fallback embedder.
_PROVENANCE_FIELDS = ("pa_hash_source", "pa_verification_status", "engine_commit", "embedder_id")


def _normalize_verdict(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        raw = raw.to_dict()
    elif hasattr(raw, "__dict__") and not isinstance(raw, dict):
        raw = vars(raw)
    if not isinstance(raw, dict):
        raw = {}

    raw_dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    dimensions = {name: _coerce_number(raw_dimensions.get(name)) for name in DIMENSION_NAMES}
    field_map = {
        "purpose": "purpose_fidelity",
        "scope": "scope_fidelity",
        "boundary": "boundary_violation",
        "tool": "tool_fidelity",
        "chain": "chain_continuity",
        "risk": "risk",
    }
    for name, field in field_map.items():
        if dimensions[name] is None and field in raw:
            dimensions[name] = _coerce_number(raw.get(field))

    decision = raw.get("decision") or raw.get("verdict")
    return {
        # A scorer that returned no decision did not approve anything.
        "decision": str(decision).lower() if decision else "unscored",
        "allowed": bool(raw.get("allowed", False)) if decision else False,
        "fidelity": _coerce_number(raw.get("fidelity")),
        "dimensions": dimensions,
        "explanation": _sanitize_text(str(raw.get("explanation") or "")),
        "pa_hash": _string_or_empty(raw.get("pa_hash")),
        "config_hash": _string_or_empty(raw.get("config_hash")),
        "scorer_version": _string_or_empty(raw.get("scorer_version")),
        "scorer_model_hash": _string_or_empty(raw.get("scorer_model_hash")),
        **{key: _string_or_empty(raw.get(key)) for key in _PROVENANCE_FIELDS},
    }


def _build_receipt(
    *,
    correlation_id: str,
    pre_kwargs: dict[str, Any],
    post_kwargs: dict[str, Any],
    classification: dict[str, Any],
    verdict: dict[str, Any],
    scoring_status: str,
    scoring_latency_ms: float,
    started_at: float | None,
    errors: list[dict[str, str]],
    registry_record: dict[str, Any],
) -> dict[str, Any]:
    merged = {**pre_kwargs, **post_kwargs}
    if pre_kwargs.get("args") and not post_kwargs.get("args"):
        merged["args"] = pre_kwargs.get("args")
    post_duration = post_kwargs.get("duration_ms")
    observed_latency_ms = None
    if started_at is not None:
        observed_latency_ms = round((time.time() - started_at) * 1000, 3)
    post_status = _string_or_empty(post_kwargs.get("status"))
    execution_allowed = post_status != "blocked"
    verdict_allowed = bool(verdict.get("allowed", False)) if scoring_status == "scored" else False
    binding = _receipt_binding(verdict)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "version": 2,
        "receipt_id": f"telos-tool-{uuid.uuid4()}",
        "correlation_id": correlation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile_id": os.getenv("HERMES_PROFILE", "") or os.getenv("HERMES_ACTIVE_PROFILE", "") or "default",
        "session_id": _string_or_empty(merged.get("session_id")),
        "task_id": _string_or_empty(merged.get("task_id")),
        "turn_id": _string_or_empty(merged.get("turn_id")),
        "api_request_id": _string_or_empty(merged.get("api_request_id")),
        "tool_call_id": _string_or_empty(merged.get("tool_call_id")),
        "tool_name": _string_or_empty(merged.get("tool_name")) or "unknown",
        "args": _redact(merged.get("args") or {}),
        "classification": _redact(classification),
        "pa_hash": binding["pa_hash"],
        "config_hash": binding["config_hash"],
        "scorer_version": verdict.get("scorer_version") or binding["scorer_version"],
        "scorer_model_hash": verdict.get("scorer_model_hash") or binding["scorer_model_hash"],
        **{key: binding[key] for key in _PROVENANCE_FIELDS},
        "canonical_input_hash": "",
        "verdict_hash": "",
        "receipt_hash": "",
        "previous_receipt_hash": _previous_receipt_hash(),
        "registry_event_hash": _string_or_empty(registry_record.get("registry_event_hash")),
        "dimensions": verdict.get("dimensions") or {name: None for name in DIMENSION_NAMES},
        "verdict": {
            "decision": verdict.get("decision") or "execute",
            "allowed": verdict_allowed,
            "fidelity": verdict.get("fidelity"),
            "explanation": _sanitize_text(str(verdict.get("explanation") or "")),
        },
        "execution_allowed": execution_allowed,
        "verdict_allowed": verdict_allowed,
        "allowed": execution_allowed,
        "allowed_semantics": "execution_allowed",
        "allowed_divergence": _allowed_divergence(
            execution_allowed=execution_allowed,
            verdict_allowed=verdict_allowed,
            scoring_status=scoring_status,
        ),
        "passive": True,
        "scoring_status": scoring_status,
        "score_available": scoring_status == "scored",
        "governance_degraded": scoring_status != "scored",
        "fail_policy": _fail_policy(),
        "fail_policy_scope": "receipt_integrity",
        "drift_flags": _drift_flags(
            scoring_status=scoring_status,
            execution_allowed=execution_allowed,
            verdict_allowed=verdict_allowed,
            registry_record=registry_record,
        ),
        "closed_core_mechanism": CLOSED_CORE_MECHANISM,
        "latency": {
            "scoring_ms": scoring_latency_ms,
            "observed_tool_ms": observed_latency_ms,
        },
        "post": {
            "status": post_status,
            "duration_ms": post_duration,
            "result_metadata": _result_metadata(post_kwargs.get("result")),
            "error": _error_metadata(post_kwargs.get("error")),
        },
        "errors": _redact(errors),
        "signature_available": False,
    }
    receipt["canonical_input_hash"] = _sha256_json(_receipt_input_payload(receipt))
    receipt["verdict_hash"] = _sha256_json(receipt["verdict"])
    receipt["receipt_hash"] = _receipt_hash(receipt)
    receipt["governance_proof"] = {
        "type": "telos.operational.tool_call.v2.hash_chain",
        "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'));sha256",
        "signature_available": False,
        "receipt_hash": receipt["receipt_hash"],
        "previous_receipt_hash": receipt["previous_receipt_hash"],
    }
    return receipt


def reverify_receipt(
    receipt: dict[str, Any],
    public_key: bytes | str | None = None,
    hmac_key: bytes | str | None = None,
) -> dict[str, Any]:
    """Recompute passive receipt hashes and optional signatures."""
    expected_input_hash = _sha256_json(_receipt_input_payload(receipt))
    expected_verdict_hash = _sha256_json(receipt.get("verdict") or {})
    expected_receipt_hash = _receipt_hash(receipt)
    checks = {
        "canonical_input_hash": receipt.get("canonical_input_hash") == expected_input_hash,
        "verdict_hash": receipt.get("verdict_hash") == expected_verdict_hash,
        "receipt_hash": receipt.get("receipt_hash") == expected_receipt_hash,
    }
    signature_valid = None
    if public_key is not None and receipt.get("signature"):
        signature_valid = _verify_ed25519_signature(receipt, public_key)
        checks["signature"] = bool(signature_valid)
    if hmac_key is not None and receipt.get("hmac"):
        hmac_valid = _verify_hmac(receipt, hmac_key)
        checks["hmac"] = bool(hmac_valid)
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "canonical_input_hash": expected_input_hash,
        "verdict_hash": expected_verdict_hash,
        "receipt_hash": expected_receipt_hash,
    }


def _ensure_registry_record(kwargs: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_cli import capability_registry

        return capability_registry.ensure_tool_registered(
            str(kwargs.get("tool_name") or "unknown"),
            classification=classification,
            description=classification.get("action_text", ""),
        )
    except Exception as exc:
        logger.debug("TELOS capability registry tool update failed: %s", exc, exc_info=True)
        return {}


def _receipt_binding(verdict: dict[str, Any]) -> dict[str, str]:
    try:
        from hermes_cli import capability_registry

        manifest = capability_registry.get_manifest()
    except Exception:
        manifest = {}
    pa_hash = verdict.get("pa_hash") or manifest.get("pa_hash") or _fallback_hash("pa")
    config_hash = (
        verdict.get("config_hash") or manifest.get("config_hash") or _fallback_hash("config")
    )
    scorer_version = verdict.get("scorer_version") or os.getenv(
        "HERMES_TELOS_PASSIVE_SCORER_VERSION", "unknown"
    )
    scorer_model_hash = verdict.get("scorer_model_hash") or os.getenv(
        "HERMES_TELOS_PASSIVE_SCORER_MODEL_HASH", ""
    )
    if not scorer_model_hash:
        scorer_model_hash = hashlib.sha256(str(scorer_version).encode("utf-8")).hexdigest()
    return {
        "pa_hash": _string_or_empty(pa_hash),
        "config_hash": _string_or_empty(config_hash),
        "scorer_version": _string_or_empty(scorer_version),
        "scorer_model_hash": _string_or_empty(scorer_model_hash),
        **{key: _string_or_empty(verdict.get(key)) or "missing" for key in _PROVENANCE_FIELDS},
    }


def _fallback_hash(label: str) -> str:
    return hashlib.sha256(f"missing:{label}".encode("utf-8")).hexdigest()


def _fail_policy() -> str:
    raw = os.getenv("HERMES_TELOS_PASSIVE_FAIL_POLICY", "").strip().lower()
    if raw in {"open", "observe", "permissive"}:
        return raw
    return "closed"


def _allowed_divergence(
    *,
    execution_allowed: bool,
    verdict_allowed: bool,
    scoring_status: str,
) -> str:
    if execution_allowed == verdict_allowed:
        return ""
    if scoring_status != "scored":
        return "scorer_degraded"
    return "passive_observation"


def _drift_flags(
    *,
    scoring_status: str,
    execution_allowed: bool,
    verdict_allowed: bool,
    registry_record: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if scoring_status != "scored":
        flags.append("scorer_degraded")
    if execution_allowed != verdict_allowed:
        flags.append("allowed_divergence")
    if registry_record.get("binding_status") == "stale":
        flags.append("capability_binding_stale")
    if not registry_record:
        flags.append("registry_unavailable")
    return flags


def _receipt_input_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": receipt.get("schema_version"),
        "profile_id": receipt.get("profile_id"),
        "session_id": receipt.get("session_id"),
        "task_id": receipt.get("task_id"),
        "turn_id": receipt.get("turn_id"),
        "api_request_id": receipt.get("api_request_id"),
        "tool_call_id": receipt.get("tool_call_id"),
        "tool_name": receipt.get("tool_name"),
        "args": receipt.get("args"),
        "classification": receipt.get("classification"),
        "pa_hash": receipt.get("pa_hash"),
        "config_hash": receipt.get("config_hash"),
    }


def _receipt_hash(receipt: dict[str, Any]) -> str:
    return _sha256_json(_receipt_payload_for_hash(receipt))


def _receipt_payload_for_hash(receipt: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "receipt_hash",
        "signature",
        "hmac",
        "governance_proof",
        "transport_metadata",
    }
    return {key: value for key, value in receipt.items() if key not in excluded}


def _previous_receipt_hash() -> str:
    path = _receipt_path()
    if not path.exists():
        return ""
    try:
        last = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return ""
        parsed = json.loads(last)
        return _string_or_empty(parsed.get("receipt_hash"))
    except (OSError, ValueError):
        return ""


def _verify_ed25519_signature(receipt: dict[str, Any], public_key: bytes | str) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key_bytes = bytes.fromhex(public_key) if isinstance(public_key, str) else public_key
        signature = receipt.get("signature")
        sig_bytes = bytes.fromhex(signature) if isinstance(signature, str) else bytes(signature)
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(
            sig_bytes,
            _canonical_json(_receipt_payload_for_hash(receipt)).encode("utf-8"),
        )
        return True
    except Exception:
        return False


def _verify_hmac(receipt: dict[str, Any], hmac_key: bytes | str) -> bool:
    key = hmac_key.encode("utf-8") if isinstance(hmac_key, str) else hmac_key
    expected = hmac.new(
        key,
        _canonical_json(_receipt_payload_for_hash(receipt)).encode("utf-8"),
        hashlib.sha512,
    ).hexdigest()
    return hmac.compare_digest(str(receipt.get("hmac") or ""), expected)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _result_metadata(result: Any) -> dict[str, Any] | None:
    """Persist the SHAPE of a tool result, never its content.

    A `read_file` of a `.env` used to land in the receipt verbatim.
    Redacting it is not enough: redaction is pattern-based and a result is arbitrary text, so the
    only defensible contract is that no result value is written at all. What remains is the type,
    the size, the line count, and a sha256 over the exact content -- enough to prove later that a
    given result is the one that was measured, and enough for drift analysis, while carrying none
    of the bytes.
    """
    if result is None:
        return None
    value = _jsonable(result)
    text = value if isinstance(value, str) else _canonical_json(value)
    return {
        "type": type(result).__name__,
        "length": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_persisted": False,
    }


def _error_metadata(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, dict):
        return _redact(error)
    message = getattr(error, "message", None) or str(error)
    return {"message": _sanitize_text(str(message))}


def _write_receipt(receipt: dict[str, Any]) -> None:
    try:
        path = _receipt_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # Create owner-only. An existing file is deliberately NOT re-permissioned: changing the
        # mode of a live receipt store is an operator decision, not this plugin's.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:
        logger.debug("TELOS passive scorer receipt write failed open: %s", exc, exc_info=True)


def _receipt_path() -> Path:
    override = os.getenv("HERMES_TELOS_PASSIVE_RECEIPTS_PATH", "").strip()
    if override:
        return Path(override)
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / DEFAULT_RECEIPT_RELATIVE_PATH
    except Exception:
        return Path.home() / ".hermes" / DEFAULT_RECEIPT_RELATIVE_PATH


def _correlation_id(kwargs: dict[str, Any]) -> str:
    for key in ("tool_call_id", "api_request_id"):
        value = kwargs.get(key)
        if value:
            return str(value)
    return "|".join(
        [
            str(kwargs.get("session_id") or ""),
            str(kwargs.get("task_id") or ""),
            str(kwargs.get("turn_id") or ""),
            str(kwargs.get("tool_name") or "unknown"),
        ]
    )


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_assignment(key_text, _jsonable_scalar_text(item)):
                clean[key_text] = "[REDACTED]"
            elif "path" in key_text.lower() or "file" in key_text.lower():
                clean[key_text] = _redact_path_value(item)
            else:
                clean[key_text] = _redact(item)
        return clean
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _jsonable_scalar_text(value: Any) -> str:
    """The text a dict VALUE would carry, for the bare-plural count exemption only."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def _redact_path_value(value: Any) -> Any:
    if isinstance(value, str):
        return "[REDACTED_PATH]" if _looks_like_path(value) else _sanitize_text(value)
    if isinstance(value, list):
        return [_redact_path_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_path_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_path_value(item) for key, item in value.items()}
    return value


def _sanitize_text(text: str) -> str:
    """Remove every credential shape from a string.

    Order is load-bearing. PEM blocks are multi-line and go first. URL userinfo goes before the
    assignment rule so that `https://user:pw@host` is not first mangled as `user:pw`. Bare
    vendor tokens go last, after the assignment rule has already collapsed `key=<token>`.

    Every marker is a fixed string, so the redacted output carries no information about the
    length of what it replaced.
    """
    text = _PEM_BLOCK_RE.sub(REDACTED_KEY_MARKER, text)
    text = _PEM_HEADER_RE.sub(REDACTED_KEY_MARKER, text)
    text = _URL_USERINFO_RE.sub(lambda match: f"{match.group(1)}{REDACTED_MARKER}@", text)
    text = _ABS_PATH_WITH_SPACES_RE.sub(REDACTED_PATH_MARKER, text)
    text = _ABS_PATH_RE.sub(REDACTED_PATH_MARKER, text)
    text = _redact_assignments(text)
    for pattern in _BARE_TOKEN_RES:
        text = pattern.sub(REDACTED_MARKER, text)
    return text


def _looks_like_path(text: str) -> bool:
    return bool(text.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", text))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return _sanitize_text(str(value))


def reset_for_tests() -> None:
    global _SCORER, _SCORER_LOADED, _SCORER_ERROR
    with _STATE_LOCK:
        _PENDING.clear()
    _SCORER = None
    _SCORER_LOADED = False
    _SCORER_ERROR = ""
