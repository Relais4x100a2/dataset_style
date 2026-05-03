"""Worker planifié pour reprise des sagas de suppression/révocation comptes."""

from __future__ import annotations

import logging
import os

from src.auth import revoke_account_with_saga
from src.config import initialize_runtime_config
from src.database import (
    create_db_engine,
    ensure_schema,
    list_retryable_deprovision_ops,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _database_url() -> str:
    raw = (os.environ.get("DATABASE_URL") or "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL manquant.")
    return raw


def _max_retries() -> int:
    raw = (os.environ.get("ACCOUNT_SAGA_MAX_RETRIES") or "5").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 5
    return max(1, min(value, 20))


def _batch_size() -> int:
    raw = (os.environ.get("ACCOUNT_RETRY_BATCH_SIZE") or "50").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 50
    return max(1, min(value, 500))


def main() -> int:
    """Point d'entrée worker."""
    initialize_runtime_config()
    engine = create_db_engine(_database_url())
    ensure_schema(engine)
    max_retries = _max_retries()
    rows = list_retryable_deprovision_ops(engine, limit=_batch_size())
    if not rows:
        logger.info("Aucune opération deprovision à reprendre.")
        return 0
    success = 0
    failed = 0
    for row in rows:
        try:
            state = revoke_account_with_saga(
                engine,
                actor_user_id=row.actor_user_id,
                target_user_id=row.target_user_id,
                operation_id=row.operation_id,
                max_retries=max_retries,
                detach_memberships=False,
            )
            logger.info("op=%s state=%s", row.operation_id, state)
            success += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("op=%s error=%s", row.operation_id, exc)
            failed += 1
    logger.info("Retry worker terminé: success=%s failed=%s", success, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
