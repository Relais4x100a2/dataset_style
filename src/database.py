"""
Accès PostgreSQL multi-tenant.

Modèle:
- users
- projects
- project_settings
- entries (rattachées à project_id)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

STATUT_VALIDE = "Fait et validé"

PROJECT_ROLES = ("admin", "collaborator", "viewer")

CACHE_COLUMNS = [
    "_ratio",
    "_ttr",
    "_long_phrases",
    "_signature_json",
    "_coherence_score",
    "_trigrams_json",
    "_lexical_density",
    "_weak_verb_ratio",
    "_syntax_contrast",
    "_nb_sentences",
    "_punct_exp",
    "_stop_ratio_out",
]

ENTRY_COLUMNS = [
    "id",
    "project_id",
    "date",
    "type",
    "structure",
    "ton",
    "format",
    "public",
    "input",
    "output",
    "statut",
    "notes",
    *CACHE_COLUMNS,
]


@dataclass
class UserRecord:
    user_id: str
    email: str
    display_name: str
    is_super_admin: bool = False
    disabled_at: str = ""
    last_login_at: str = ""


@dataclass
class ProjectRecord:
    project_id: str
    name: str
    role: str


@dataclass
class ProjectSettings:
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout_seconds: str = ""
    languagetool_base_url: str = ""
    active_preset_key: str = "roman"
    custom_presets_json: str = ""
    dimensions_override_json: str = ""


@dataclass
class AccountAdminRow:
    user_id: str
    email: str
    is_super_admin: bool
    project_count: int
    last_login_at: str
    entries_total: int
    entries_validated: int


@dataclass
class DeprovisionOp:
    operation_id: str
    target_user_id: str
    actor_user_id: str
    state: str
    retry_count: int
    last_error: str
    next_retry_at: str
    quarantined_at: str


DEPROVISION_STATES = ("pending", "provider_done", "db_done", "completed", "failed", "quarantined")


def create_db_engine(database_url: str) -> Engine:
    """Crée un moteur SQLAlchemy compatible psycopg v3."""
    url = database_url.strip()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True)


def ensure_schema(engine: Engine) -> None:
    """Crée le schéma multi-tenant s'il n'existe pas."""
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        su_user_id TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        is_super_admin BOOLEAN NOT NULL DEFAULT FALSE,
        disabled_at TIMESTAMPTZ NULL,
        last_login_at TIMESTAMPTZ NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        created_by TEXT NOT NULL REFERENCES users(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        archived BOOLEAN NOT NULL DEFAULT FALSE
    );

    CREATE TABLE IF NOT EXISTS project_memberships (
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (project_id, user_id)
    );

    CREATE TABLE IF NOT EXISTS project_settings (
        project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
        llm_base_url TEXT NOT NULL DEFAULT '',
        llm_model TEXT NOT NULL DEFAULT '',
        llm_api_key TEXT NOT NULL DEFAULT '',
        llm_timeout_seconds TEXT NOT NULL DEFAULT '',
        languagetool_base_url TEXT NOT NULL DEFAULT '',
        active_preset_key TEXT NOT NULL DEFAULT 'roman',
        custom_presets_json TEXT NOT NULL DEFAULT '',
        dimensions_override_json TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS entries (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        date TEXT NOT NULL DEFAULT '',
        type TEXT NOT NULL DEFAULT '',
        forme TEXT NOT NULL DEFAULT '',
        structure TEXT NOT NULL DEFAULT '',
        ton TEXT NOT NULL DEFAULT '',
        support TEXT NOT NULL DEFAULT '',
        format TEXT NOT NULL DEFAULT '',
        public TEXT NOT NULL DEFAULT '',
        input TEXT NOT NULL DEFAULT '',
        output TEXT NOT NULL DEFAULT '',
        statut TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        _ratio TEXT NOT NULL DEFAULT '',
        _ttr TEXT NOT NULL DEFAULT '',
        _long_phrases TEXT NOT NULL DEFAULT '',
        _signature_json TEXT NOT NULL DEFAULT '',
        _coherence_score TEXT NOT NULL DEFAULT '',
        _trigrams_json TEXT NOT NULL DEFAULT '',
        _lexical_density TEXT NOT NULL DEFAULT '',
        _weak_verb_ratio TEXT NOT NULL DEFAULT '',
        _syntax_contrast TEXT NOT NULL DEFAULT '',
        _nb_sentences TEXT NOT NULL DEFAULT '',
        _punct_exp TEXT NOT NULL DEFAULT '',
        _stop_ratio_out TEXT NOT NULL DEFAULT ''
    );

    CREATE INDEX IF NOT EXISTS idx_memberships_user_project ON project_memberships(user_id, project_id);
    CREATE INDEX IF NOT EXISTS idx_projects_created_by ON projects(created_by);
    CREATE INDEX IF NOT EXISTS idx_users_last_login_at ON users(last_login_at);
    CREATE INDEX IF NOT EXISTS idx_users_disabled_at ON users(disabled_at);
    CREATE INDEX IF NOT EXISTS idx_users_email_ci ON users((lower(email)));

    CREATE TABLE IF NOT EXISTS user_deprovision_ops (
        operation_id TEXT PRIMARY KEY,
        target_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        actor_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        state TEXT NOT NULL,
        retry_count INTEGER NOT NULL DEFAULT 0,
        last_error TEXT NOT NULL DEFAULT '',
        next_retry_at TIMESTAMPTZ NULL,
        quarantined_at TIMESTAMPTZ NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (target_user_id, state)
    );
    CREATE INDEX IF NOT EXISTS idx_deprovision_state_updated ON user_deprovision_ops(state, updated_at);
    CREATE INDEX IF NOT EXISTS idx_deprovision_next_retry ON user_deprovision_ops(next_retry_at);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(
            text(
                "ALTER TABLE project_settings ADD COLUMN IF NOT EXISTS active_preset_key TEXT NOT NULL DEFAULT 'roman';"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE project_settings ADD COLUMN IF NOT EXISTS custom_presets_json TEXT NOT NULL DEFAULT '';"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE project_settings ADD COLUMN IF NOT EXISTS dimensions_override_json TEXT NOT NULL DEFAULT '';"
            )
        )
        conn.execute(text("ALTER TABLE entries ADD COLUMN IF NOT EXISTS project_id TEXT;"))
        conn.execute(
            text("ALTER TABLE entries ADD COLUMN IF NOT EXISTS structure TEXT NOT NULL DEFAULT '';")
        )
        conn.execute(
            text("ALTER TABLE entries ADD COLUMN IF NOT EXISTS format TEXT NOT NULL DEFAULT '';")
        )
        conn.execute(
            text("ALTER TABLE entries ADD COLUMN IF NOT EXISTS public TEXT NOT NULL DEFAULT '';")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_entries_project_statut ON entries(project_id, statut);"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_entries_project_id ON entries(project_id, id);")
        )
        conn.execute(
            text("UPDATE entries SET structure = forme WHERE structure = '' AND forme <> '';")
        )
        conn.execute(
            text("UPDATE entries SET format = support WHERE format = '' AND support <> '';")
        )
        conn.execute(
            text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_super_admin BOOLEAN NOT NULL DEFAULT FALSE;"
            )
        )
        conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ NULL;")
        )
        conn.execute(
            text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ NULL;")
        )
        conn.execute(
            text(
                "ALTER TABLE user_deprovision_ops ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ NULL;"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE user_deprovision_ops ADD COLUMN IF NOT EXISTS quarantined_at TIMESTAMPTZ NULL;"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE user_deprovision_ops DROP CONSTRAINT IF EXISTS chk_deprovision_state;"
            )
        )
        conn.execute(
            text(
                """
                ALTER TABLE user_deprovision_ops
                ADD CONSTRAINT chk_deprovision_state
                CHECK (state IN ('pending', 'provider_done', 'db_done', 'completed', 'failed', 'quarantined'));
                """
            )
        )
        conn.execute(
            text(
                "UPDATE user_deprovision_ops SET next_retry_at = NOW() WHERE next_retry_at IS NULL;"
            )
        )


def upsert_user_from_su(
    engine: Engine,
    su_user_id: str,
    email: str,
    display_name: str,
) -> UserRecord:
    """Crée/met à jour un utilisateur local depuis SuperTokens."""
    ensure_schema(engine)
    if not su_user_id.strip():
        raise ValueError("su_user_id requis.")
    user_id = f"u_{uuid.uuid4().hex[:12]}"
    normalized_email = email.strip().lower()
    display_name_clean = display_name.strip() or normalized_email.split("@")[0]
    with engine.begin() as conn:
        existing = (
            conn.execute(
                text(
                    """
                    SELECT id, disabled_at
                    FROM users
                    WHERE su_user_id = :su_user_id OR lower(email) = lower(:email)
                    LIMIT 1;
                    """
                ),
                {"su_user_id": su_user_id, "email": normalized_email},
            )
            .mappings()
            .first()
        )
        if existing and existing.get("disabled_at") is not None:
            raise PermissionError("Compte révoqué. Contacte un administrateur.")
    sql = """
    INSERT INTO users(id, su_user_id, email, display_name, disabled_at)
    VALUES (:id, :su_user_id, :email, :display_name, NULL)
    ON CONFLICT (su_user_id)
    DO UPDATE SET email = EXCLUDED.email, display_name = EXCLUDED.display_name
    RETURNING id, email, display_name, is_super_admin, disabled_at, last_login_at;
    """
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(sql),
                {
                    "id": user_id,
                    "su_user_id": su_user_id,
                    "email": normalized_email,
                    "display_name": display_name_clean,
                },
            )
            .mappings()
            .first()
        )
    return UserRecord(
        user_id=str(row["id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        is_super_admin=bool(row.get("is_super_admin") or False),
        disabled_at="" if row.get("disabled_at") is None else str(row["disabled_at"]),
        last_login_at="" if row.get("last_login_at") is None else str(row["last_login_at"]),
    )


def mark_user_login(engine: Engine, user_id: str) -> None:
    """Met à jour la date de dernière connexion."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET last_login_at = NOW() WHERE id = :uid"),
            {"uid": user_id},
        )


def is_user_super_admin(engine: Engine, user_id: str) -> bool:
    """Retourne True si l'utilisateur est super admin."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                text("SELECT is_super_admin FROM users WHERE id = :uid"),
                {"uid": user_id},
            )
            .mappings()
            .first()
        )
    return bool(row and row.get("is_super_admin"))


def get_su_user_id_by_user_id(engine: Engine, user_id: str) -> str:
    """Retourne l'identifiant provider (SuperTokens) d'un utilisateur."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                text("SELECT su_user_id FROM users WHERE id = :uid"),
                {"uid": user_id},
            )
            .mappings()
            .first()
        )
    if not row:
        raise ValueError("Utilisateur introuvable.")
    return str(row["su_user_id"])


def require_super_admin(engine: Engine, actor_user_id: str) -> None:
    """Vérifie qu'un utilisateur est super admin global."""
    if not is_user_super_admin(engine, actor_user_id):
        raise PermissionError("Droits super admin requis.")


def grant_super_admin_by_email(engine: Engine, email: str) -> bool:
    """Promeut un compte en super admin par email normalisé."""
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE users
                SET is_super_admin = TRUE
                WHERE lower(email) = lower(:email)
                """
            ),
            {"email": email.strip().lower()},
        )
    return bool(result.rowcount)


def count_active_memberships(engine: Engine, user_id: str) -> int:
    """Compte les memberships actives d'un utilisateur."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                text("SELECT COUNT(*) AS c FROM project_memberships WHERE user_id = :uid"),
                {"uid": user_id},
            )
            .mappings()
            .first()
        )
    return int((row or {}).get("c", 0))


def count_owned_projects(engine: Engine, user_id: str) -> int:
    """Compte les projets possédés (non archivés) d'un utilisateur."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS c
                    FROM projects
                    WHERE created_by = :uid
                      AND archived = FALSE
                    """
                ),
                {"uid": user_id},
            )
            .mappings()
            .first()
        )
    return int((row or {}).get("c", 0))


def detach_memberships_as_super_admin(
    engine: Engine, actor_user_id: str, target_user_id: str
) -> int:
    """Retire toutes les memberships d'un utilisateur (action auditée côté appelant)."""
    require_super_admin(engine, actor_user_id)
    with engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM project_memberships WHERE user_id = :uid"),
            {"uid": target_user_id},
        )
    return int(result.rowcount or 0)


def count_users_for_admin(engine: Engine) -> int:
    """Compte les utilisateurs actifs pour la pagination super admin."""
    with engine.begin() as conn:
        row = (
            conn.execute(text("SELECT COUNT(*) AS c FROM users WHERE disabled_at IS NULL"))
            .mappings()
            .first()
        )
    return int((row or {}).get("c", 0))


def list_accounts_for_super_admin(
    engine: Engine,
    actor_user_id: str,
    *,
    limit: int,
    offset: int,
) -> list[AccountAdminRow]:
    """Liste paginée des comptes avec métriques globales."""
    require_super_admin(engine, actor_user_id)
    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    sql = """
    WITH project_counts AS (
        SELECT created_by AS user_id, COUNT(*) AS project_count
        FROM projects
        WHERE archived = FALSE
        GROUP BY created_by
    ),
    entry_counts AS (
        SELECT
            p.created_by AS user_id,
            COUNT(e.id) AS entries_total,
            COUNT(*) FILTER (WHERE e.statut = :valid_status) AS entries_validated
        FROM projects p
        LEFT JOIN entries e ON e.project_id = p.id
        WHERE p.archived = FALSE
        GROUP BY p.created_by
    )
    SELECT
        u.id,
        u.email,
        u.is_super_admin,
        COALESCE(pc.project_count, 0) AS project_count,
        u.last_login_at,
        COALESCE(ec.entries_total, 0) AS entries_total,
        COALESCE(ec.entries_validated, 0) AS entries_validated
    FROM users u
    LEFT JOIN project_counts pc ON pc.user_id = u.id
    LEFT JOIN entry_counts ec ON ec.user_id = u.id
    WHERE u.disabled_at IS NULL
    ORDER BY u.created_at DESC
    LIMIT :limit OFFSET :offset;
    """
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(sql),
                {"valid_status": STATUT_VALIDE, "limit": safe_limit, "offset": safe_offset},
            )
            .mappings()
            .all()
        )
    out: list[AccountAdminRow] = []
    for row in rows:
        out.append(
            AccountAdminRow(
                user_id=str(row["id"]),
                email=str(row["email"]),
                is_super_admin=bool(row["is_super_admin"]),
                project_count=int(row["project_count"]),
                last_login_at="" if row["last_login_at"] is None else str(row["last_login_at"]),
                entries_total=int(row["entries_total"]),
                entries_validated=int(row["entries_validated"]),
            )
        )
    return out


def create_deprovision_operation(
    engine: Engine,
    *,
    operation_id: str,
    actor_user_id: str,
    target_user_id: str,
) -> DeprovisionOp:
    """Crée (ou retourne) une opération de révocation idempotente."""
    ensure_schema(engine)
    if not operation_id.strip():
        raise ValueError("operation_id requis.")
    if actor_user_id != target_user_id:
        require_super_admin(engine, actor_user_id)
    with engine.begin() as conn:
        existing = (
            conn.execute(
                text("SELECT * FROM user_deprovision_ops WHERE operation_id = :oid"),
                {"oid": operation_id},
            )
            .mappings()
            .first()
        )
        if existing:
            if (
                str(existing["target_user_id"]) != target_user_id
                or str(existing["actor_user_id"]) != actor_user_id
            ):
                raise ValueError("operation_id déjà utilisé pour une autre opération.")
            return DeprovisionOp(
                operation_id=str(existing["operation_id"]),
                target_user_id=str(existing["target_user_id"]),
                actor_user_id=str(existing["actor_user_id"]),
                state=str(existing["state"]),
                retry_count=int(existing["retry_count"]),
                last_error=str(existing["last_error"] or ""),
                next_retry_at=""
                if existing.get("next_retry_at") is None
                else str(existing["next_retry_at"]),
                quarantined_at=""
                if existing.get("quarantined_at") is None
                else str(existing["quarantined_at"]),
            )
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:target_uid))"),
            {"target_uid": target_user_id},
        )
        active = (
            conn.execute(
                text(
                    """
                    SELECT operation_id
                    FROM user_deprovision_ops
                    WHERE target_user_id = :target_uid
                      AND state IN ('pending', 'provider_done', 'failed')
                    LIMIT 1
                    """
                ),
                {"target_uid": target_user_id},
            )
            .mappings()
            .first()
        )
        if active:
            raise RuntimeError(
                "Une opération active existe déjà pour cet utilisateur. Termine-la avant d'en créer une nouvelle."
            )
        conn.execute(
            text(
                """
                INSERT INTO user_deprovision_ops(operation_id, target_user_id, actor_user_id, state, next_retry_at)
                VALUES (:oid, :target_uid, :actor_uid, 'pending', NOW())
                """
            ),
            {"oid": operation_id, "target_uid": target_user_id, "actor_uid": actor_user_id},
        )
    return DeprovisionOp(
        operation_id=operation_id,
        target_user_id=target_user_id,
        actor_user_id=actor_user_id,
        state="pending",
        retry_count=0,
        last_error="",
        next_retry_at="",
        quarantined_at="",
    )


def get_deprovision_operation(engine: Engine, operation_id: str) -> DeprovisionOp | None:
    """Récupère une opération de révocation par identifiant."""
    with engine.begin() as conn:
        row = (
            conn.execute(
                text("SELECT * FROM user_deprovision_ops WHERE operation_id = :oid"),
                {"oid": operation_id},
            )
            .mappings()
            .first()
        )
    if not row:
        return None
    return DeprovisionOp(
        operation_id=str(row["operation_id"]),
        target_user_id=str(row["target_user_id"]),
        actor_user_id=str(row["actor_user_id"]),
        state=str(row["state"]),
        retry_count=int(row["retry_count"]),
        last_error=str(row["last_error"] or ""),
        next_retry_at="" if row.get("next_retry_at") is None else str(row["next_retry_at"]),
        quarantined_at="" if row.get("quarantined_at") is None else str(row["quarantined_at"]),
    )


def transition_deprovision_operation(
    engine: Engine,
    *,
    operation_id: str,
    expected_state: str,
    next_state: str,
    error_message: str = "",
    increment_retry: bool = False,
) -> None:
    """Transitionne l'état d'une saga de révocation de manière conditionnelle."""
    if expected_state not in DEPROVISION_STATES or next_state not in DEPROVISION_STATES:
        raise ValueError("État de saga invalide.")
    retry_delta = 1 if increment_retry else 0
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                UPDATE user_deprovision_ops
                SET
                    state = :next_state,
                    retry_count = retry_count + :retry_delta,
                    last_error = :last_error,
                    next_retry_at = CASE
                        WHEN :next_state IN ('completed', 'quarantined') THEN NULL
                        ELSE next_retry_at
                    END,
                    quarantined_at = CASE
                        WHEN :next_state = 'quarantined' THEN NOW()
                        ELSE quarantined_at
                    END,
                    updated_at = NOW()
                WHERE operation_id = :oid
                  AND state = :expected_state
                """
            ),
            {
                "next_state": next_state,
                "retry_delta": retry_delta,
                "last_error": error_message.strip(),
                "oid": operation_id,
                "expected_state": expected_state,
            },
        )
    if result.rowcount == 0:
        raise RuntimeError("Transition saga refusée (concurrence ou état inattendu).")


def mark_user_disabled(engine: Engine, user_id: str) -> None:
    """Désactive localement un utilisateur."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET disabled_at = COALESCE(disabled_at, NOW()) WHERE id = :uid"),
            {"uid": user_id},
        )


def record_deprovision_failure(
    engine: Engine,
    *,
    operation_id: str,
    expected_state: str,
    error_message: str,
    max_retries: int,
    backoff_seconds: int,
) -> DeprovisionOp:
    """
    Incrémente retry_count, programme le prochain retry, ou place en quarantaine.
    """
    if max_retries < 1:
        raise ValueError("max_retries doit être >= 1.")
    safe_backoff = max(30, min(backoff_seconds, 24 * 3600))
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT state, retry_count
                    FROM user_deprovision_ops
                    WHERE operation_id = :oid
                    FOR UPDATE
                    """
                ),
                {"oid": operation_id},
            )
            .mappings()
            .first()
        )
        if not row:
            raise ValueError("Opération deprovision introuvable.")
        current_state = str(row["state"])
        if current_state != expected_state:
            raise RuntimeError("Transition saga refusée (concurrence ou état inattendu).")
        next_retry_count = int(row["retry_count"]) + 1
        if next_retry_count >= max_retries:
            conn.execute(
                text(
                    """
                    UPDATE user_deprovision_ops
                    SET
                        state = 'quarantined',
                        retry_count = :retry_count,
                        last_error = :last_error,
                        next_retry_at = NULL,
                        quarantined_at = NOW(),
                        updated_at = NOW()
                    WHERE operation_id = :oid
                    """
                ),
                {
                    "retry_count": next_retry_count,
                    "last_error": error_message.strip(),
                    "oid": operation_id,
                },
            )
        else:
            conn.execute(
                text(
                    """
                    UPDATE user_deprovision_ops
                    SET
                        retry_count = :retry_count,
                        last_error = :last_error,
                        next_retry_at = NOW() + make_interval(secs => :backoff_seconds),
                        updated_at = NOW()
                    WHERE operation_id = :oid
                    """
                ),
                {
                    "retry_count": next_retry_count,
                    "last_error": error_message.strip(),
                    "backoff_seconds": safe_backoff,
                    "oid": operation_id,
                },
            )
    op = get_deprovision_operation(engine, operation_id)
    if op is None:
        raise ValueError("Opération deprovision introuvable après mise à jour.")
    return op


def list_retryable_deprovision_ops(engine: Engine, *, limit: int = 50) -> list[DeprovisionOp]:
    """Liste les opérations à reprendre par le worker planifié."""
    safe_limit = max(1, min(limit, 500))
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT *
                FROM user_deprovision_ops
                WHERE state IN ('pending', 'provider_done', 'failed')
                  AND (next_retry_at IS NULL OR next_retry_at <= NOW())
                ORDER BY updated_at ASC
                LIMIT :limit
                """
                ),
                {"limit": safe_limit},
            )
            .mappings()
            .all()
        )
    out: list[DeprovisionOp] = []
    for row in rows:
        out.append(
            DeprovisionOp(
                operation_id=str(row["operation_id"]),
                target_user_id=str(row["target_user_id"]),
                actor_user_id=str(row["actor_user_id"]),
                state=str(row["state"]),
                retry_count=int(row["retry_count"]),
                last_error=str(row["last_error"] or ""),
                next_retry_at="" if row.get("next_retry_at") is None else str(row["next_retry_at"]),
                quarantined_at=""
                if row.get("quarantined_at") is None
                else str(row["quarantined_at"]),
            )
        )
    return out


def list_recent_deprovision_ops(
    engine: Engine, actor_user_id: str, *, limit: int = 100
) -> list[DeprovisionOp]:
    """Retourne les opérations de saga récentes pour monitoring super admin."""
    require_super_admin(engine, actor_user_id)
    safe_limit = max(1, min(limit, 500))
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT *
                FROM user_deprovision_ops
                ORDER BY updated_at DESC
                LIMIT :limit
                """
                ),
                {"limit": safe_limit},
            )
            .mappings()
            .all()
        )
    out: list[DeprovisionOp] = []
    for row in rows:
        out.append(
            DeprovisionOp(
                operation_id=str(row["operation_id"]),
                target_user_id=str(row["target_user_id"]),
                actor_user_id=str(row["actor_user_id"]),
                state=str(row["state"]),
                retry_count=int(row["retry_count"]),
                last_error=str(row["last_error"] or ""),
                next_retry_at="" if row.get("next_retry_at") is None else str(row["next_retry_at"]),
                quarantined_at=""
                if row.get("quarantined_at") is None
                else str(row["quarantined_at"]),
            )
        )
    return out


def list_quarantined_deprovision_ops(
    engine: Engine, actor_user_id: str, *, limit: int = 50
) -> list[DeprovisionOp]:
    """Liste les opérations en quarantaine (DLQ)."""
    require_super_admin(engine, actor_user_id)
    safe_limit = max(1, min(limit, 200))
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT *
                FROM user_deprovision_ops
                WHERE state = 'quarantined'
                ORDER BY updated_at DESC
                LIMIT :limit
                """
                ),
                {"limit": safe_limit},
            )
            .mappings()
            .all()
        )
    out: list[DeprovisionOp] = []
    for row in rows:
        out.append(
            DeprovisionOp(
                operation_id=str(row["operation_id"]),
                target_user_id=str(row["target_user_id"]),
                actor_user_id=str(row["actor_user_id"]),
                state=str(row["state"]),
                retry_count=int(row["retry_count"]),
                last_error=str(row["last_error"] or ""),
                next_retry_at="" if row.get("next_retry_at") is None else str(row["next_retry_at"]),
                quarantined_at=""
                if row.get("quarantined_at") is None
                else str(row["quarantined_at"]),
            )
        )
    return out


def replay_quarantined_operation(engine: Engine, actor_user_id: str, operation_id: str) -> None:
    """Remet une opération DLQ en file d'attente."""
    require_super_admin(engine, actor_user_id)
    with engine.begin() as conn:
        op = (
            conn.execute(
                text(
                    """
                    SELECT operation_id, target_user_id
                    FROM user_deprovision_ops
                    WHERE operation_id = :oid
                      AND state = 'quarantined'
                    FOR UPDATE
                    """
                ),
                {"oid": operation_id},
            )
            .mappings()
            .first()
        )
        if not op:
            raise RuntimeError("Opération non rejouable (introuvable ou non quarantined).")
        target_uid = str(op["target_user_id"])
        conn.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:target_uid))"),
            {"target_uid": target_uid},
        )
        conflicting = (
            conn.execute(
                text(
                    """
                    SELECT operation_id
                    FROM user_deprovision_ops
                    WHERE target_user_id = :target_uid
                      AND state IN ('pending', 'provider_done', 'failed')
                      AND operation_id <> :oid
                    LIMIT 1
                    """
                ),
                {"target_uid": target_uid, "oid": operation_id},
            )
            .mappings()
            .first()
        )
        if conflicting:
            raise RuntimeError(
                "Replay impossible: une autre opération active existe déjà pour cet utilisateur."
            )
        result = conn.execute(
            text(
                """
                UPDATE user_deprovision_ops
                SET
                    state = 'pending',
                    last_error = '',
                    next_retry_at = NOW(),
                    quarantined_at = NULL,
                    updated_at = NOW()
                WHERE operation_id = :oid
                  AND state = 'quarantined'
                """
            ),
            {"oid": operation_id},
        )
        if result.rowcount == 0:
            raise RuntimeError("Opération non rejouable (introuvable ou non quarantined).")


def reactivate_user(engine: Engine, user_id: str) -> None:
    """Réactive un utilisateur local (compensation)."""
    with engine.begin() as conn:
        conn.execute(text("UPDATE users SET disabled_at = NULL WHERE id = :uid"), {"uid": user_id})


def delete_user_if_detached(engine: Engine, user_id: str) -> None:
    """Désactive définitivement un utilisateur local sans projets ni memberships."""
    if count_owned_projects(engine, user_id) > 0:
        raise ValueError("Suppression refusée: l'utilisateur possède des projets.")
    if count_active_memberships(engine, user_id) > 0:
        raise ValueError("Suppression refusée: memberships actives détectées.")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE users
                SET
                    disabled_at = COALESCE(disabled_at, NOW()),
                    is_super_admin = FALSE,
                    email = CONCAT('deleted+', id, '@local.invalid'),
                    display_name = 'deleted-user'
                WHERE id = :uid
                """
            ),
            {"uid": user_id},
        )


def delete_own_account_if_allowed(engine: Engine, user_id: str) -> None:
    """Supprime le compte courant si aucun projet ni membership n'existe."""
    delete_user_if_detached(engine, user_id)


def list_projects_for_user(engine: Engine, user_id: str) -> list[ProjectRecord]:
    ensure_schema(engine)
    sql = """
    SELECT p.id, p.name, 'admin' AS role
    FROM projects p
    WHERE p.created_by = :user_id AND p.archived = FALSE
    ORDER BY p.updated_at DESC, p.created_at DESC;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"user_id": user_id}).mappings().all()
    return [
        ProjectRecord(project_id=str(r["id"]), name=str(r["name"]), role=str(r["role"]))
        for r in rows
    ]


def create_project(engine: Engine, user_id: str, name: str, description: str = "") -> str:
    ensure_schema(engine)
    pid = f"p_{uuid.uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO projects(id, name, description, created_by)
                VALUES (:id, :name, :description, :created_by);
                """
            ),
            {
                "id": pid,
                "name": name.strip(),
                "description": description.strip(),
                "created_by": user_id,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO project_settings(project_id)
                VALUES (:project_id)
                ON CONFLICT (project_id) DO NOTHING;
                """
            ),
            {"project_id": pid},
        )
    return pid


def delete_project(engine: Engine, project_id: str, user_id: str) -> None:
    require_admin(engine, project_id, user_id)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM projects WHERE id = :pid"), {"pid": project_id})


def get_role(engine: Engine, project_id: str, user_id: str) -> str | None:
    sql = "SELECT 1 FROM projects WHERE id = :pid AND created_by = :uid AND archived = FALSE"
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"pid": project_id, "uid": user_id}).first()
    return "admin" if row else None


def require_role(engine: Engine, project_id: str, user_id: str, allowed: tuple[str, ...]) -> str:
    role = get_role(engine, project_id, user_id)
    if role is None or role not in allowed:
        logger.warning(
            "Action refusée: user_id=%s project_id=%s allowed=%s role=%s",
            user_id,
            project_id,
            allowed,
            role,
        )
        raise PermissionError("Droits insuffisants pour cette action.")
    return role


def require_admin(engine: Engine, project_id: str, user_id: str) -> None:
    """Vérifie qu'un utilisateur est admin du projet."""
    require_role(engine, project_id, user_id, ("admin",))


def list_members(engine: Engine, project_id: str) -> pd.DataFrame:
    sql = """
    SELECT u.email, u.display_name, m.role
    FROM project_memberships m
    JOIN users u ON u.id = m.user_id
    WHERE m.project_id = :pid
    ORDER BY m.added_at ASC;
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"pid": project_id})
    if df.empty:
        return pd.DataFrame(columns=["email", "display_name", "role"])
    return df.fillna("")


def add_or_update_member(engine: Engine, project_id: str, email: str, role: str) -> bool:
    if role not in PROJECT_ROLES:
        raise ValueError("Rôle invalide.")
    with engine.begin() as conn:
        user = conn.execute(
            text("SELECT id FROM users WHERE lower(email) = lower(:email)"),
            {"email": email.strip()},
        ).first()
        if not user:
            return False
        conn.execute(
            text(
                """
                INSERT INTO project_memberships(project_id, user_id, role)
                VALUES (:pid, :uid, :role)
                ON CONFLICT (project_id, user_id)
                DO UPDATE SET role = EXCLUDED.role;
                """
            ),
            {"pid": project_id, "uid": str(user[0]), "role": role},
        )
    return True


def remove_member(engine: Engine, project_id: str, email: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM project_memberships
                WHERE project_id = :pid
                  AND user_id IN (SELECT id FROM users WHERE lower(email) = lower(:email));
                """
            ),
            {"pid": project_id, "email": email.strip()},
        )


def get_project_settings(engine: Engine, project_id: str) -> ProjectSettings:
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                SELECT llm_base_url, llm_model, llm_api_key, llm_timeout_seconds, languagetool_base_url
                , active_preset_key, custom_presets_json, dimensions_override_json
                FROM project_settings
                WHERE project_id = :pid
                """
                ),
                {"pid": project_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            conn.execute(
                text("INSERT INTO project_settings(project_id) VALUES (:pid)"), {"pid": project_id}
            )
            return ProjectSettings()
    return ProjectSettings(
        llm_base_url=str(row["llm_base_url"] or ""),
        llm_model=str(row["llm_model"] or ""),
        llm_api_key=str(row["llm_api_key"] or ""),
        llm_timeout_seconds=str(row["llm_timeout_seconds"] or ""),
        languagetool_base_url=str(row["languagetool_base_url"] or ""),
        active_preset_key=str(row["active_preset_key"] or "roman"),
        custom_presets_json=str(row["custom_presets_json"] or ""),
        dimensions_override_json=str(row["dimensions_override_json"] or ""),
    )


def update_project_settings(engine: Engine, project_id: str, settings: ProjectSettings) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO project_settings(
                    project_id,
                    llm_base_url,
                    llm_model,
                    llm_api_key,
                    llm_timeout_seconds,
                    languagetool_base_url,
                    active_preset_key,
                    custom_presets_json,
                    dimensions_override_json,
                    updated_at
                )
                VALUES (:pid, :llm_base_url, :llm_model, :llm_api_key, :llm_timeout_seconds, :languagetool_base_url, :active_preset_key, :custom_presets_json, :dimensions_override_json, NOW())
                ON CONFLICT (project_id)
                DO UPDATE SET
                    llm_base_url = EXCLUDED.llm_base_url,
                    llm_model = EXCLUDED.llm_model,
                    llm_api_key = EXCLUDED.llm_api_key,
                    llm_timeout_seconds = EXCLUDED.llm_timeout_seconds,
                    languagetool_base_url = EXCLUDED.languagetool_base_url,
                    active_preset_key = EXCLUDED.active_preset_key,
                    custom_presets_json = EXCLUDED.custom_presets_json,
                    dimensions_override_json = EXCLUDED.dimensions_override_json,
                    updated_at = NOW();
                """
            ),
            {
                "pid": project_id,
                "llm_base_url": settings.llm_base_url,
                "llm_model": settings.llm_model,
                "llm_api_key": settings.llm_api_key,
                "llm_timeout_seconds": settings.llm_timeout_seconds,
                "languagetool_base_url": settings.languagetool_base_url,
                "active_preset_key": settings.active_preset_key,
                "custom_presets_json": settings.custom_presets_json,
                "dimensions_override_json": settings.dimensions_override_json,
            },
        )


def delete_project_as_admin(engine: Engine, project_id: str, actor_user_id: str) -> None:
    """Supprime un projet après vérification stricte du rôle admin."""
    delete_project(engine, project_id, actor_user_id)


def add_or_update_member_as_admin(
    engine: Engine,
    project_id: str,
    actor_user_id: str,
    email: str,
    role: str,
) -> bool:
    """Ajoute ou met à jour un membre après vérification stricte du rôle admin."""
    require_admin(engine, project_id, actor_user_id)
    return add_or_update_member(engine, project_id, email, role)


def remove_member_as_admin(
    engine: Engine,
    project_id: str,
    actor_user_id: str,
    email: str,
) -> None:
    """Retire un membre après vérification stricte du rôle admin."""
    require_admin(engine, project_id, actor_user_id)
    remove_member(engine, project_id, email)


def update_project_settings_as_admin(
    engine: Engine,
    project_id: str,
    actor_user_id: str,
    settings: ProjectSettings,
) -> None:
    """Met à jour les réglages après vérification stricte du rôle admin."""
    require_admin(engine, project_id, actor_user_id)
    update_project_settings(engine, project_id, settings)


def _normalize_entry_df(df: pd.DataFrame, project_id: str) -> pd.DataFrame:
    out = df.copy()
    if "structure" not in out.columns and "forme" in out.columns:
        out["structure"] = out["forme"]
    if "format" not in out.columns and "support" in out.columns:
        out["format"] = out["support"]
    if "public" not in out.columns:
        out["public"] = ""
    for col in ENTRY_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["project_id"] = project_id
    return out[ENTRY_COLUMNS].astype(str).replace(["nan", "None", "<NA>"], "")


def load_project_entries(engine: Engine, project_id: str, user_id: str) -> pd.DataFrame:
    require_role(engine, project_id, user_id, ("admin", "collaborator", "viewer"))
    ensure_schema(engine)
    with engine.begin() as conn:
        df = pd.read_sql(
            text("SELECT * FROM entries WHERE project_id = :pid ORDER BY date DESC, id DESC"),
            conn,
            params={"pid": project_id},
        )
    if df.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)
    if "structure" not in df.columns and "forme" in df.columns:
        df["structure"] = df["forme"]
    if "format" not in df.columns and "support" in df.columns:
        df["format"] = df["support"]
    if "public" not in df.columns:
        df["public"] = ""
    return df.astype(str).replace(["nan", "None", "<NA>"], "")


def update_project_entries(engine: Engine, project_id: str, df: pd.DataFrame, user_id: str) -> None:
    require_role(engine, project_id, user_id, ("admin", "collaborator"))
    payload = _normalize_entry_df(df, project_id)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM entries WHERE project_id = :pid"), {"pid": project_id})
        if not payload.empty:
            payload.to_sql("entries", conn, if_exists="append", index=False, method="multi")
