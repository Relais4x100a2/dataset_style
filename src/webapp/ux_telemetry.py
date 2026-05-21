"""Jalons UX FastAPI alignés sur ``docs/migration_parity_matrix.md`` (issue-020 / GitHub #182).

Corrélation anonyme : le client génère un ``run_id`` stable (``ux_`` + hex) et
l'envoie dans ``X-Dataset-Style-Ux-Run-Id``. L'écriture JSONL n'a lieu que si
``DATASET_STYLE_UX_TELEMETRY_DIR`` est défini (même condition que Streamlit).

``EXP-SCOPE`` n'est **pas** émis depuis les routes export : le récap qualité
pré-export Streamlit n'a pas d'équivalent HTTP (écart documenté issue-015).
Chaque téléchargement CSV ou JSONL émet un jalon ``EXP-DL`` (extra
``delivery`` = ``csv`` | ``jsonl``) ; comparer les temps inter-jalons au
parcours Streamlit en tenant compte de cette granularité.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Final, Literal

from fastapi import Request
from starlette.responses import Response

from src.api_errors import TenantResourceOpaqueDenial
from src.database import ProjectRecord
from src.project_session import MembershipProject, resolve_active_project
from src.ux_scenario_telemetry import (
    MILESTONE_EDI_SAVE,
    MILESTONE_ENT_NEW_WRITE,
    MILESTONE_EXP_DL,
    MILESTONE_SB_CTX,
    SCENARIO_CRITICAL_V1,
    UX_TELEMETRY_DIR_ENV,
    fingerprint_project,
    record_ux_error_event,
    record_ux_scenario_event,
)

UX_RUN_ID_HEADER: Final = "X-Dataset-Style-Ux-Run-Id"
UX_SCENARIO_ID_HEADER: Final = "X-Dataset-Style-Ux-Scenario-Id"
UX_TELEMETRY_ACTIVE_HEADER: Final = "X-Dataset-Style-Ux-Telemetry"

_RUN_ID_RE = re.compile(r"^ux_[0-9a-f]{8,120}$", re.IGNORECASE)
_SCENARIO_ID_RE = re.compile(r"^[a-z0-9_]{1,80}$")

_dedupe_keys: set[str] = set()
_DEDUPE_MAX_KEYS: Final = 4096


def reset_webapp_ux_dedupe_for_tests() -> None:
    """Vide le cache de déduplication (tests uniquement)."""
    _dedupe_keys.clear()


def ux_file_collection_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Vrai si l'écriture JSONL UX est activée (même condition que Streamlit)."""
    env_map = os.environ if environ is None else environ
    return bool((env_map.get(UX_TELEMETRY_DIR_ENV) or "").strip())


def _dedupe_consume(key: str) -> bool:
    if key in _dedupe_keys:
        return False
    if len(_dedupe_keys) >= _DEDUPE_MAX_KEYS:
        _dedupe_keys.clear()
    _dedupe_keys.add(key)
    return True


def webapp_ux_run_id(request: Request) -> str | None:
    """Retourne un ``run_id`` valide depuis l'en-tête client, ou ``None``."""
    raw = request.headers.get(UX_RUN_ID_HEADER.lower())
    if raw is None:
        return None
    candidate = raw.strip()
    if _RUN_ID_RE.fullmatch(candidate) is None:
        return None
    return candidate.lower()


def webapp_ux_scenario_id(request: Request) -> str:
    """Identifiant de scénario versionné (défaut = parcours critique v1)."""
    raw = request.headers.get(UX_SCENARIO_ID_HEADER.lower())
    if not raw:
        return SCENARIO_CRITICAL_V1
    s = raw.strip()
    if not s or _SCENARIO_ID_RE.match(s) is None:
        return SCENARIO_CRITICAL_V1
    return s


def maybe_record_webapp_sb_ctx(
    request: Request,
    *,
    projects: list[ProjectRecord],
    active_hint: str | None,
) -> None:
    """Émet ``SB-CTX`` une fois par couple ``(run_id, projet actif)`` si collecte active."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    summaries = [MembershipProject(p.project_id, p.role) for p in projects]
    pid, _ = resolve_active_project((active_hint or "").strip(), summaries)
    if not pid:
        return
    fp = fingerprint_project(pid)
    if not _dedupe_consume(f"SB-CTX:{rid}:{fp}"):
        return
    record_ux_scenario_event(
        run_id=rid,
        milestone_code=MILESTONE_SB_CTX,
        surface="webapp",
        project_fp=fp,
        scenario_id=webapp_ux_scenario_id(request),
        extra={"project_list_len": len(projects)},
    )


def record_webapp_persist_entry_milestone(
    request: Request,
    *,
    project_id: str,
    milestone_code: str,
) -> None:
    """Après persistance réussie d'une fiche (création ou édition)."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    allowed = (MILESTONE_ENT_NEW_WRITE, MILESTONE_EDI_SAVE)
    if milestone_code not in allowed:
        raise ValueError(f"milestone_code must be one of {allowed!r}, got {milestone_code!r}")
    record_ux_scenario_event(
        run_id=rid,
        milestone_code=milestone_code,
        surface="webapp",
        project_fp=fingerprint_project(project_id),
        scenario_id=webapp_ux_scenario_id(request),
        extra={"saved": True},
    )


def maybe_record_webapp_entry_access_denied(
    request: Request,
    *,
    project_id: str,
    milestone_context: str,
) -> None:
    """404 opaque après tentative d'écriture (ex. entrée hors projet)."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    record_ux_error_event(
        run_id=rid,
        surface="webapp",
        milestone_context=milestone_context,
        project_fp=fingerprint_project(project_id),
        exception=TenantResourceOpaqueDenial(),
    )


def record_webapp_export_payload_too_large(
    request: Request,
    *,
    project_id: str,
    exc: BaseException,
) -> None:
    """Erreur export (plafond lignes) — ``milestone_context`` aligné flux export."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    record_ux_error_event(
        run_id=rid,
        surface="webapp",
        milestone_context=MILESTONE_EXP_DL,
        project_fp=fingerprint_project(project_id),
        exception=exc,
    )


def record_webapp_export_milestones(
    request: Request,
    *,
    project_id: str,
    scope: str,
    export_row_count: int,
    delivery: Literal["csv", "jsonl"],
    csv_byte_len: int,
    jsonl_byte_len: int | None,
    jsonl_format: str | None,
) -> None:
    """Émet uniquement ``EXP-DL`` pour cette réponse (pas ``EXP-SCOPE`` — écart matrice)."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    fp = fingerprint_project(project_id)
    if not _dedupe_consume(f"EXP-DL:{rid}:{fp}:{scope}:{delivery}"):
        return
    dl_extra: dict[str, object] = {
        "export_scope": scope,
        "delivery": delivery,
        "export_row_count": export_row_count,
        "csv_bytes": csv_byte_len,
    }
    if jsonl_byte_len is not None:
        dl_extra["jsonl_bytes"] = jsonl_byte_len
    if jsonl_format is not None:
        dl_extra["jsonl_format"] = jsonl_format
    record_ux_scenario_event(
        run_id=rid,
        milestone_code=MILESTONE_EXP_DL,
        surface="webapp",
        project_fp=fp,
        scenario_id=webapp_ux_scenario_id(request),
        extra=dl_extra,
    )


def attach_ux_telemetry_response_markers(response: Response, request: Request) -> None:
    """En-têtes de corrélation lorsque la collecte fichier est active et ``run_id`` valide."""
    if not ux_file_collection_enabled():
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    response.headers[UX_RUN_ID_HEADER] = rid
    response.headers[UX_TELEMETRY_ACTIVE_HEADER] = "1"
