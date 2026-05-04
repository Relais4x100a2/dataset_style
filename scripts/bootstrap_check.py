"""
Validation de la configuration avant déploiement.

Vérifie :
  - Variables d'environnement obligatoires
  - Connexion à la base PostgreSQL
  - Connexion au service SuperTokens
  - Intégrité minimale du schéma (tables attendues présentes)

Usage :
    python scripts/bootstrap_check.py
    python scripts/bootstrap_check.py --strict   # échoue si super admin absent

Exit codes :
    0  Tout OK
    1  Au moins une vérification a échoué
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import NamedTuple

import requests
from sqlalchemy import text

# Permettre l'import depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import initialize_runtime_config
from src.database import create_db_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS: list[str] = [
    "DATABASE_URL",
    "SUPERTOKENS_CONNECTION_URI",
    "SUPERTOKENS_API_KEY",
]

EXPECTED_TABLES: list[str] = [
    "users",
    "projects",
    "project_memberships",
    "project_settings",
    "entries",
    "user_deprovision_ops",
]


class CheckResult(NamedTuple):
    """Résultat d'une vérification individuelle."""

    name: str
    passed: bool
    detail: str


def check_env_vars() -> list[CheckResult]:
    """Vérifie la présence des variables d'environnement obligatoires."""
    results: list[CheckResult] = []
    for var in REQUIRED_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            results.append(CheckResult(f"env:{var}", True, "présente"))
        else:
            results.append(CheckResult(f"env:{var}", False, "MANQUANTE ou vide"))
    return results


def check_database_connection() -> CheckResult:
    """Vérifie la connexion à PostgreSQL et l'accessibilité de la base."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return CheckResult("db:connection", False, "DATABASE_URL non définie")
    try:
        engine = create_db_engine(url)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT version()")).scalar()
        version = str(row or "").split(",")[0]
        return CheckResult("db:connection", True, version)
    except Exception as exc:
        return CheckResult("db:connection", False, f"{type(exc).__name__}: {exc}")


def check_database_schema() -> list[CheckResult]:
    """Vérifie que les tables attendues existent dans la base."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return [CheckResult("db:schema", False, "DATABASE_URL non définie")]

    results: list[CheckResult] = []
    try:
        engine = create_db_engine(url)
        with engine.connect() as conn:
            for table in EXPECTED_TABLES:
                row = conn.execute(
                    text(
                        "SELECT EXISTS ("
                        "  SELECT 1 FROM information_schema.tables"
                        "  WHERE table_schema = 'public' AND table_name = :tbl"
                        ")"
                    ),
                    {"tbl": table},
                ).scalar()
                exists = bool(row)
                results.append(
                    CheckResult(
                        f"db:table:{table}",
                        exists,
                        "présente" if exists else "ABSENTE (ensure_schema() requis)",
                    )
                )
    except Exception as exc:
        results.append(CheckResult("db:schema", False, f"{type(exc).__name__}: {exc}"))
    return results


def check_super_admin_exists() -> CheckResult:
    """Vérifie qu'au moins un super admin est configuré en base."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return CheckResult("db:super_admin", False, "DATABASE_URL non définie")
    try:
        engine = create_db_engine(url)
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM users WHERE is_super_admin = TRUE AND disabled_at IS NULL")
            ).scalar()
        n = int(count or 0)
        if n > 0:
            return CheckResult("db:super_admin", True, f"{n} super admin(s) actif(s)")
        return CheckResult(
            "db:super_admin",
            False,
            "Aucun super admin — exécuter le SQL de promotion (voir docs/caprover_deployment.md §6.2)",
        )
    except Exception as exc:
        return CheckResult("db:super_admin", False, f"{type(exc).__name__}: {exc}")


def check_supertokens_connection() -> CheckResult:
    """Vérifie que SuperTokens répond sur /hello."""
    uri = os.environ.get("SUPERTOKENS_CONNECTION_URI", "").strip().rstrip("/")
    if not uri:
        return CheckResult("supertokens:connection", False, "SUPERTOKENS_CONNECTION_URI non définie")
    try:
        api_key = os.environ.get("SUPERTOKENS_API_KEY", "").strip()
        headers: dict[str, str] = {}
        if api_key:
            headers["api-key"] = api_key
        resp = requests.get(f"{uri}/hello", headers=headers, timeout=5)
        if resp.status_code == 200 and "OK" in resp.text:
            return CheckResult("supertokens:connection", True, f"HTTP {resp.status_code} — {resp.text.strip()}")
        return CheckResult(
            "supertokens:connection",
            False,
            f"HTTP {resp.status_code} — réponse inattendue: {resp.text[:100]}",
        )
    except requests.ConnectionError as exc:
        return CheckResult("supertokens:connection", False, f"Connexion refusée: {exc}")
    except requests.Timeout:
        return CheckResult("supertokens:connection", False, "Timeout (5s) — service inaccessible")
    except Exception as exc:
        return CheckResult("supertokens:connection", False, f"{type(exc).__name__}: {exc}")


def check_supertokens_api_key() -> CheckResult:
    """Vérifie que la clé API SuperTokens est acceptée."""
    uri = os.environ.get("SUPERTOKENS_CONNECTION_URI", "").strip().rstrip("/")
    api_key = os.environ.get("SUPERTOKENS_API_KEY", "").strip()
    if not uri:
        return CheckResult("supertokens:api_key", False, "SUPERTOKENS_CONNECTION_URI non définie")
    if not api_key:
        return CheckResult("supertokens:api_key", False, "SUPERTOKENS_API_KEY vide — auth désactivée ou manquante")
    try:
        resp = requests.get(
            f"{uri}/recipe/dashboard/api/list",
            headers={"api-key": api_key},
            timeout=5,
        )
        if resp.status_code in (200, 404):
            return CheckResult("supertokens:api_key", True, f"Clé acceptée (HTTP {resp.status_code})")
        if resp.status_code == 401:
            return CheckResult("supertokens:api_key", False, "Clé API refusée (HTTP 401) — vérifier SUPERTOKENS_API_KEY vs API_KEYS")
        return CheckResult("supertokens:api_key", True, f"HTTP {resp.status_code} (présumé OK)")
    except Exception as exc:
        return CheckResult("supertokens:api_key", False, f"{type(exc).__name__}: {exc}")


def _print_result(result: CheckResult) -> None:
    """Affiche un résultat de vérification avec icône."""
    icon = "✓" if result.passed else "✗"
    level = logging.INFO if result.passed else logging.ERROR
    logger.log(level, "%s  %-40s %s", icon, result.name, result.detail)


def run_checks(*, strict: bool = False) -> bool:
    """Exécute toutes les vérifications et retourne True si tout est OK.

    Args:
        strict: Si True, l'absence de super admin est une erreur bloquante.

    Returns:
        True si toutes les vérifications obligatoires sont passées.
    """
    initialize_runtime_config()

    all_results: list[CheckResult] = []

    logger.info("=" * 60)
    logger.info("Bootstrap check — Dataset Style")
    logger.info("=" * 60)

    logger.info("\n── Variables d'environnement ──")
    env_results = check_env_vars()
    for r in env_results:
        _print_result(r)
    all_results.extend(env_results)

    logger.info("\n── Base de données ──")
    db_conn = check_database_connection()
    _print_result(db_conn)
    all_results.append(db_conn)

    if db_conn.passed:
        schema_results = check_database_schema()
        for r in schema_results:
            _print_result(r)
        all_results.extend(schema_results)

        admin_result = check_super_admin_exists()
        _print_result(admin_result)
        if strict:
            all_results.append(admin_result)
        elif not admin_result.passed:
            logger.warning("  ↳ Super admin absent — déploiement possible mais bootstrap incomplet")

    logger.info("\n── SuperTokens ──")
    st_conn = check_supertokens_connection()
    _print_result(st_conn)
    all_results.append(st_conn)

    if st_conn.passed:
        st_key = check_supertokens_api_key()
        _print_result(st_key)
        all_results.append(st_key)

    logger.info("\n" + "=" * 60)
    failed = [r for r in all_results if not r.passed]
    if failed:
        logger.error("RÉSULTAT : %d vérification(s) échouée(s)", len(failed))
        for r in failed:
            logger.error("  ✗ %s — %s", r.name, r.detail)
        return False

    logger.info("RÉSULTAT : Toutes les vérifications sont passées ✓")
    return True


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Valide la configuration avant déploiement Dataset Style"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Échoue si aucun super admin n'est configuré",
    )
    args = parser.parse_args()

    success = run_checks(strict=args.strict)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
