"""Panneau technique saga super-admin (issue-019) — charge utile alignée sur ``database``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.engine import Engine

from src.database import (
    SUPER_ADMIN_SAGA_DLQ_PREVIEW_LIMIT,
    SUPER_ADMIN_SAGA_RECENT_OPS_LIMIT,
    SUPER_ADMIN_SAGA_RETRY_QUEUE_PREVIEW_LIMIT,
    DeprovisionOp,
    count_deprovision_ops_by_state,
    list_quarantined_deprovision_ops,
    list_recent_deprovision_ops,
    list_retryable_deprovision_ops_for_super_admin,
)

_STREAMLIT_STYLE_METRIC_STATES = ("pending", "provider_done", "failed", "quarantined")


def counts_in_recent_window(ops: Sequence[DeprovisionOp]) -> dict[str, int]:
    """Compte les états des cartes métriques Streamlit sur une fenêtre d'opérations récentes."""
    out: dict[str, int] = {k: 0 for k in _STREAMLIT_STYLE_METRIC_STATES}
    for op in ops:
        if op.state in out:
            out[op.state] += 1
    return out


def serialize_deprovision_op(op: DeprovisionOp) -> dict[str, Any]:
    """Représentation JSON stable pour le client webapp."""
    next_at = (op.next_retry_at or "").strip()
    q_at = (op.quarantined_at or "").strip()
    return {
        "operationId": op.operation_id,
        "targetUserId": op.target_user_id,
        "actorUserId": op.actor_user_id,
        "state": op.state,
        "retryCount": op.retry_count,
        "lastError": op.last_error,
        "nextRetryAt": next_at or None,
        "quarantinedAt": q_at or None,
    }


def build_deprovision_telemetry_payload(engine: Engine, actor_user_id: str) -> dict[str, Any]:
    """Assemble télémétrie saga, file et DLQ via les helpers ``database`` (garde super-admin)."""
    recent = list_recent_deprovision_ops(
        engine, actor_user_id, limit=SUPER_ADMIN_SAGA_RECENT_OPS_LIMIT
    )
    dlq = list_quarantined_deprovision_ops(
        engine, actor_user_id, limit=SUPER_ADMIN_SAGA_DLQ_PREVIEW_LIMIT
    )
    queue = list_retryable_deprovision_ops_for_super_admin(
        engine, actor_user_id, limit=SUPER_ADMIN_SAGA_RETRY_QUEUE_PREVIEW_LIMIT
    )
    totals = count_deprovision_ops_by_state(engine, actor_user_id)
    return {
        "recentOpsLimit": SUPER_ADMIN_SAGA_RECENT_OPS_LIMIT,
        "dlqPreviewLimit": SUPER_ADMIN_SAGA_DLQ_PREVIEW_LIMIT,
        "retryQueuePreviewLimit": SUPER_ADMIN_SAGA_RETRY_QUEUE_PREVIEW_LIMIT,
        "stateCountsInRecentWindow": counts_in_recent_window(recent),
        "totalsByState": totals,
        "recentOps": [serialize_deprovision_op(o) for o in recent],
        "dlqOps": [serialize_deprovision_op(o) for o in dlq],
        "retryQueueOps": [serialize_deprovision_op(o) for o in queue],
    }
