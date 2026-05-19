"""Télémétrie légère du scénario critique UX (issue-020 / GitHub #142).

Append-only JSONL lorsque ``DATASET_STYLE_UX_TELEMETRY_DIR`` pointe vers un
répertoire accessible en écriture. Aucune table PostgreSQL : bundle protocole
+ fichiers agrégés pour l'archive de bascule (issue-001).

Les ``milestone_code`` reprennent les IDs flux de ``docs/migration_parity_matrix.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Final, Literal

from src.api_errors import resolve_exception_for_api

logger = logging.getLogger(__name__)

UX_TELEMETRY_DIR_ENV: Final = "DATASET_STYLE_UX_TELEMETRY_DIR"
SCENARIO_CRITICAL_V1: Final = "critical_v1_issue_020"

MILESTONE_SB_CTX: Final = "SB-CTX"
MILESTONE_ENT_NEW_WRITE: Final = "ENT-NEW-WRITE"
MILESTONE_EDI_SAVE: Final = "EDI-SAVE"
MILESTONE_EXP_SCOPE: Final = "EXP-SCOPE"
MILESTONE_EXP_DL: Final = "EXP-DL"

MILESTONE_CODES_CRITICAL_V1: Final[set[str]] = {
    MILESTONE_SB_CTX,
    MILESTONE_ENT_NEW_WRITE,
    MILESTONE_EDI_SAVE,
    MILESTONE_EXP_SCOPE,
    MILESTONE_EXP_DL,
}

Surface = Literal["streamlit", "webapp"]

_STREAMLIT_RUN_ID_KEY: Final = "_issue020_ux_run_id"


def fingerprint_project(project_id: str) -> str:
    """Empreinte courte non réversible pour corrélation interne sans exposer l'ID projet."""
    digest = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    return digest[:16]


def emit_once_per_session_key(emitted: set[str], dedupe_key: str) -> bool:
    """Retourne True si la clé est nouvelle et l'enregistre dans ``emitted``."""
    if dedupe_key in emitted:
        return False
    emitted.add(dedupe_key)
    return True


def ensure_streamlit_run_id(session: MutableMapping[str, Any]) -> str:
    """Identifiant anonyme stable pour la session Streamlit (reruns inclus)."""
    existing = session.get(_STREAMLIT_RUN_ID_KEY)
    if isinstance(existing, str) and existing.startswith("ux_") and len(existing) > 8:
        return existing
    run_id = f"ux_{uuid.uuid4().hex}"
    session[_STREAMLIT_RUN_ID_KEY] = run_id
    return run_id


def streamlit_dedupe_bucket(session: MutableMapping[str, Any]) -> set[str]:
    """Ensemble de clés de déduplication UX (persisté dans ``session_state``)."""
    key = "_issue020_ux_emit_dedupe_keys"
    bucket = session.get(key)
    if not isinstance(bucket, set):
        bucket = set()
        session[key] = bucket
    return bucket  # type: ignore[return-value]


def _telemetry_dir(environ: Mapping[str, str]) -> Path | None:
    raw = (environ.get(UX_TELEMETRY_DIR_ENV) or "").strip()
    if not raw:
        return None
    return Path(raw)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        fh.write("\n")


def record_ux_scenario_event(
    *,
    run_id: str,
    milestone_code: str,
    surface: Surface,
    project_fp: str,
    scenario_id: str = SCENARIO_CRITICAL_V1,
    monotonic_ns: int | None = None,
    extra: dict[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Enregistre un jalon de parcours (temps côté frontière métier).

    Args:
        run_id: Corrélation anonyme (ex. ``ensure_streamlit_run_id``).
        milestone_code: ID flux matrice (ex. ``EXP-SCOPE``).
        surface: ``streamlit`` ou ``webapp``.
        project_fp: Empreinte projet (:func:`fingerprint_project`).
        scenario_id: Identifiant de scénario versionné.
        monotonic_ns: Horodatage ``time.perf_counter_ns()`` ; défaut = maintenant.
        extra: Champs additionnels non sensibles (périmètre export, format, …).
        environ: Carte d'environnement (tests) ; défaut ``os.environ``.
    """
    if milestone_code not in MILESTONE_CODES_CRITICAL_V1:
        raise ValueError(f"Unknown milestone_code {milestone_code!r}")
    env_map = os.environ if environ is None else environ
    base_dir = _telemetry_dir(env_map)
    ts = time.perf_counter_ns() if monotonic_ns is None else int(monotonic_ns)
    row: dict[str, Any] = {
        "kind": "ux_milestone",
        "schema": "ux_scenario_event_v1",
        "scenario_id": scenario_id,
        "run_id": run_id,
        "surface": surface,
        "milestone_code": milestone_code,
        "project_fp": project_fp,
        "monotonic_ns": ts,
    }
    if extra:
        row["extra"] = extra
    logger.info("ux_scenario_event %s", json.dumps(row, ensure_ascii=False))
    if base_dir is None:
        return
    day = time.gmtime()[:3]
    fname = f"ux_scenario_{day[0]:04d}{day[1]:02d}{day[2]:02d}.jsonl"
    try:
        _append_jsonl(base_dir / fname, row)
    except OSError:
        logger.exception("ux_telemetry_write_failed path=%s", base_dir / fname)


def record_ux_error_event(
    *,
    run_id: str,
    surface: Surface,
    milestone_context: str,
    project_fp: str,
    exception: BaseException,
    streamlit_category: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Erreur utilisateur / persistance avec code aligné ``api_errors`` quand mappable."""
    resolved = resolve_exception_for_api(exception, include_technical_detail=False)
    env_map = os.environ if environ is None else environ
    base_dir = _telemetry_dir(env_map)
    row: dict[str, Any] = {
        "kind": "ux_error",
        "schema": "ux_error_event_v1",
        "run_id": run_id,
        "surface": surface,
        "milestone_context": milestone_context,
        "project_fp": project_fp,
        "api_error_code": resolved.code,
        "api_http_status": resolved.http_status,
    }
    if streamlit_category:
        row["streamlit_category"] = streamlit_category
    logger.info("ux_error_event %s", json.dumps(row, ensure_ascii=False))
    if base_dir is None:
        return
    day = time.gmtime()[:3]
    fname = f"ux_error_{day[0]:04d}{day[1]:02d}{day[2]:02d}.jsonl"
    try:
        _append_jsonl(base_dir / fname, row)
    except OSError:
        logger.exception("ux_telemetry_write_failed path=%s", base_dir / fname)


def record_streamlit_validation_error(
    *,
    run_id: str,
    milestone_context: str,
    project_fp: str,
    category: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Cas sans exception API (ex. champs obligatoires manquants côté UI)."""
    env_map = os.environ if environ is None else environ
    base_dir = _telemetry_dir(env_map)
    row: dict[str, Any] = {
        "kind": "ux_error",
        "schema": "ux_error_event_v1",
        "run_id": run_id,
        "surface": "streamlit",
        "milestone_context": milestone_context,
        "project_fp": project_fp,
        "api_error_code": None,
        "streamlit_category": category,
    }
    logger.info("ux_error_event %s", json.dumps(row, ensure_ascii=False))
    if base_dir is None:
        return
    day = time.gmtime()[:3]
    fname = f"ux_error_{day[0]:04d}{day[1]:02d}{day[2]:02d}.jsonl"
    try:
        _append_jsonl(base_dir / fname, row)
    except OSError:
        logger.exception("ux_telemetry_write_failed path=%s", base_dir / fname)
