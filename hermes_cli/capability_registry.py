"""Append-only TELOS capability registry for Hermes-managed surfaces.

The registry is a declaration ledger, not an execution permission gate. Skill,
agent, and tool lifecycle events are appended to NDJSON, while the current
manifest is atomically snapshotted for fast reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home

try:  # POSIX file locking; tests and deployments run on macOS/Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


MANIFEST_SCHEMA_VERSION = "telos.capability_manifest.v1"
EVENT_SCHEMA_VERSION = "telos.capability_registry_event.v1"
DEFAULT_MANIFEST_RELATIVE_PATH = Path("telos") / "capability_manifest.json"
DEFAULT_EVENTS_RELATIVE_PATH = Path("telos") / "capability_events.jsonl"

_PROCESS_LOCK = threading.RLock()


class CapabilityRegistryError(RuntimeError):
    """Raised when a registry write cannot be made verifiable."""


def register_capability(
    capability_id: str,
    *,
    kind: str,
    source: str,
    lifecycle: str = "registered",
    metadata: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Register or refresh a governed capability in the current manifest."""
    return _upsert_capability(
        "capability_registered",
        capability_id,
        kind=kind,
        source=source,
        lifecycle=lifecycle,
        metadata=metadata,
        content_hash=content_hash,
    )


def update_capability(
    capability_id: str,
    *,
    kind: str,
    source: str,
    lifecycle: str = "updated",
    metadata: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Record a lifecycle update for a governed capability."""
    return _upsert_capability(
        "capability_updated",
        capability_id,
        kind=kind,
        source=source,
        lifecycle=lifecycle,
        metadata=metadata,
        content_hash=content_hash,
    )


def remove_capability(
    capability_id: str,
    *,
    kind: str,
    source: str,
    lifecycle: str = "removed",
    metadata: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Mark a capability terminal without deleting event history."""
    return _upsert_capability(
        "capability_unregistered",
        capability_id,
        kind=kind,
        source=source,
        lifecycle=lifecycle,
        metadata=metadata,
        content_hash=content_hash,
    )


def ensure_tool_registered(
    tool_name: str,
    *,
    source: str = "passive_scorer",
    classification: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Register a tool capability on first use or refresh its last-seen time."""
    name = str(tool_name or "unknown").strip() or "unknown"
    metadata = {
        "tool_name": name,
        "description": description,
        "classification": _jsonable(classification or {}),
    }
    capability_id = f"tool:{name}"
    if capability_id in get_manifest().get("capabilities", {}):
        return update_capability(
            capability_id,
            kind="tool",
            source=source,
            lifecycle="observed",
            metadata=metadata,
        )
    return register_capability(
        capability_id,
        kind="tool",
        source=source,
        lifecycle="observed",
        metadata=metadata,
    )


def get_manifest() -> dict[str, Any]:
    """Return the current manifest, marking stale bindings in-memory."""
    ctx = _binding_context()
    with _PROCESS_LOCK:
        manifest = _load_manifest(ctx)
        return _jsonable(manifest)


def reset_for_tests() -> None:
    """Compatibility hook for tests that reset module-level state elsewhere."""
    return None


def _upsert_capability(
    event_name: str,
    capability_id: str,
    *,
    kind: str,
    source: str,
    lifecycle: str,
    metadata: dict[str, Any] | None,
    content_hash: str | None,
) -> dict[str, Any]:
    clean_id = str(capability_id or "").strip()
    if not clean_id:
        raise CapabilityRegistryError("capability_id is required")

    ctx = _binding_context()
    now = _now()
    clean_metadata = _jsonable(metadata or {})
    digest = content_hash or _sha256_json(clean_metadata)

    with _PROCESS_LOCK:
        manifest_path = _manifest_path()
        events_path = _events_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.parent.mkdir(parents=True, exist_ok=True)

        with _file_lock(manifest_path):
            manifest = _load_manifest(ctx)
            capabilities = manifest.setdefault("capabilities", {})
            previous = capabilities.get(clean_id)
            registered_at = (
                previous.get("registered_at")
                if isinstance(previous, dict) and previous.get("registered_at")
                else now
            )
            record = {
                "kind": str(kind or "unknown"),
                "lifecycle": str(lifecycle or "updated"),
                "source": str(source or "unknown"),
                "pa_hash": ctx["pa_hash"],
                "config_hash": ctx["config_hash"],
                "content_hash": digest,
                "metadata_hash": _sha256_json(clean_metadata),
                "registered_at": registered_at,
                "last_seen_at": now,
                "binding_status": "current",
                "drift_reasons": [],
            }
            if clean_metadata:
                record["metadata"] = clean_metadata

            event = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": event_name,
                "capability_id": clean_id,
                "profile_id": ctx["profile_id"],
                "pa_hash": ctx["pa_hash"],
                "config_hash": ctx["config_hash"],
                "timestamp": now,
                "record": record,
                "previous_event_hash": _last_event_hash(events_path),
            }
            event["event_hash"] = _sha256_json(event)

            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

            record["last_event_hash"] = event["event_hash"]
            capabilities[clean_id] = record
            manifest.update(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "profile_id": ctx["profile_id"],
                    "pa_hash": ctx["pa_hash"],
                    "config_hash": ctx["config_hash"],
                    "generated_at": now,
                }
            )
            _atomic_write_json(manifest_path, manifest)

    out = dict(record)
    out["capability_id"] = clean_id
    out["registry_event_hash"] = event["event_hash"]
    return out


def _load_manifest(ctx: dict[str, str]) -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return _empty_manifest(ctx)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CapabilityRegistryError(f"capability manifest is not readable JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CapabilityRegistryError("capability manifest root must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest.setdefault("profile_id", ctx["profile_id"])
    manifest.setdefault("pa_hash", ctx["pa_hash"])
    manifest.setdefault("config_hash", ctx["config_hash"])
    capabilities = manifest.setdefault("capabilities", {})
    if not isinstance(capabilities, dict):
        raise CapabilityRegistryError("capability manifest capabilities must be an object")
    for record in capabilities.values():
        if not isinstance(record, dict):
            continue
        reasons: list[str] = []
        if record.get("pa_hash") != ctx["pa_hash"]:
            reasons.append("pa_hash_changed")
        if record.get("config_hash") != ctx["config_hash"]:
            reasons.append("config_hash_changed")
        if reasons:
            record["binding_status"] = "stale"
            record["drift_reasons"] = reasons
        else:
            record.setdefault("binding_status", "current")
            record.setdefault("drift_reasons", [])
    return manifest


def _empty_manifest(ctx: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_id": ctx["profile_id"],
        "pa_hash": ctx["pa_hash"],
        "config_hash": ctx["config_hash"],
        "generated_at": _now(),
        "capabilities": {},
    }


def _binding_context() -> dict[str, str]:
    profile = (
        os.getenv("HERMES_TELOS_PROFILE_ID", "").strip()
        or os.getenv("HERMES_PROFILE", "").strip()
        or os.getenv("HERMES_ACTIVE_PROFILE", "").strip()
        or _profile_from_home()
        or "default"
    )
    return {
        "profile_id": profile,
        "pa_hash": _configured_hash(
            env_name="HERMES_TELOS_PA_HASH",
            candidates=("PA.yaml", "PA.yml", "pa.yaml", "pa.yml"),
            missing_label=f"pa:{profile}",
        ),
        "config_hash": _configured_hash(
            env_name="HERMES_TELOS_CONFIG_HASH",
            candidates=("config.yaml", "config.yml"),
            missing_label=f"config:{profile}",
        ),
    }


def _configured_hash(*, env_name: str, candidates: tuple[str, ...], missing_label: str) -> str:
    override = os.getenv(env_name, "").strip()
    if override:
        return override
    home = get_hermes_home()
    for name in candidates:
        path = home / name
        if path.is_file():
            return _sha256_file(path)
    return hashlib.sha256(f"missing:{missing_label}".encode("utf-8")).hexdigest()


def _profile_from_home() -> str:
    try:
        home = get_hermes_home()
    except Exception:
        return ""
    if home.parent.name == "profiles" and home.name:
        return home.name
    return ""


def _manifest_path() -> Path:
    override = os.getenv("HERMES_TELOS_CAPABILITY_MANIFEST_PATH", "").strip()
    if override:
        return Path(override)
    return get_hermes_home() / DEFAULT_MANIFEST_RELATIVE_PATH


def _events_path() -> Path:
    override = os.getenv("HERMES_TELOS_CAPABILITY_EVENTS_PATH", "").strip()
    if override:
        return Path(override)
    return get_hermes_home() / DEFAULT_EVENTS_RELATIVE_PATH


@contextmanager
def _file_lock(manifest_path: Path) -> Iterator[None]:
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        tmp_name = handle.name
    os.replace(tmp_name, path)


def _last_event_hash(path: Path) -> str:
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
        return str(parsed.get("event_hash") or "")
    except (OSError, ValueError):
        return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
