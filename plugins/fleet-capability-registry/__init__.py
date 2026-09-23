"""Fleet capability registry loader.

Deployments name this plugin in config. The implementation intentionally
delegates all storage behavior to hermes_cli.capability_registry so lifecycle
callers have one idempotent API and one manifest/event format.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes_cli import capability_registry

logger = logging.getLogger(__name__)


def _warm_registry(**_: Any) -> None:
    try:
        capability_registry.get_manifest()
    except Exception as exc:
        logger.debug("TELOS capability registry warmup failed: %s", exc, exc_info=True)


def register(ctx) -> None:
    _warm_registry()
    ctx.register_hook("on_session_start", _warm_registry)
