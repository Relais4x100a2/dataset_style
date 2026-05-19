"""Jalons UX FastAPI alignés sur ``docs/migration_parity_matrix.md`` (issue-020 / #142).

Les en-têtes HTTP optionnels permettent une corrélation anonyme sans cookie dédié
ni persistance en base : le client (navigateur) génère un ``run_id`` stable
(``ux_`` + 32 hex) et le renvoie sur les requêtes du parcours mesuré.

``SB-CTX`` n'est émis que si ``X-Dataset-Style-Ux-Shell-Init: 1`` est présent sur
``GET /api/projects`` afin d'éviter un bruit de mesure sur les rafraîchissements
répétés de la liste projets.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from fastapi import Request

from src.ux_scenario_telemetry import (
    MILESTONE_EDI_SAVE,
    MILESTONE_ENT_NEW_WRITE,
    MILESTONE_EXP_DL,
    MILESTONE_EXP_SCOPE,
    MILESTONE_SB_CTX,
    fingerprint_project,
    record_ux_error_event,
    record_ux_scenario_event,
)

UX_RUN_ID_HEADER: Final = "X-Dataset-Style-Ux-Run-Id"
UX_SHELL_INIT_HEADER: Final = "X-Dataset-Style-Ux-Shell-Init"

_RUN_ID_RE = re.compile(r"^ux_[0-9a-f]{32}$")


def webapp_ux_run_id(request: Request) -> str | None:
    """Retourne un ``run_id`` valide depuis l'en-tête client, ou ``None``."""
    raw = request.headers.get(UX_RUN_ID_HEADER)
    if raw is None:
        return None
    candidate = raw.strip()
    if _RUN_ID_RE.fullmatch(candidate) is None:
        return None
    return candidate


def project_fp_for_projects_list(
    *,
    project_ids_in_order: list[str],
    active_hint: str | None,
) -> str:
    """Empreinte projet pour ``SB-CTX`` (hint actif sinon premier projet listé)."""
    if active_hint:
        hid = active_hint.strip()
        if hid and hid in project_ids_in_order:
            return fingerprint_project(hid)
    if project_ids_in_order:
        return fingerprint_project(project_ids_in_order[0])
    return fingerprint_project("__webapp_no_project__")


def maybe_record_webapp_sb_ctx(
    request: Request,
    *,
    project_ids_in_order: list[str],
    active_hint: str | None,
) -> None:
    """Émet ``SB-CTX`` lors du premier chargement shell documenté (en-tête d'init)."""
    if request.headers.get(UX_SHELL_INIT_HEADER, "").strip() != "1":
        return
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    fp = project_fp_for_projects_list(
        project_ids_in_order=project_ids_in_order,
        active_hint=active_hint,
    )
    record_ux_scenario_event(
        run_id=rid,
        milestone_code=MILESTONE_SB_CTX,
        surface="webapp",
        project_fp=fp,
        extra={"project_list_len": len(project_ids_in_order)},
    )


def record_webapp_persist_entry_milestone(
    request: Request,
    *,
    project_id: str,
    milestone_code: str,
) -> None:
    """Après persistance réussie d'une fiche (création ou édition)."""
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
        extra={"saved": True},
    )


def record_webapp_export_payload_too_large(
    request: Request,
    *,
    project_id: str,
    exc: BaseException,
) -> None:
    """Erreur export (plafond lignes) — ``milestone_context`` aligné flux export."""
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
    """Périmètre résolu (``EXP-SCOPE``) puis octets prêts (``EXP-DL``), par requête HTTP."""
    rid = webapp_ux_run_id(request)
    if rid is None:
        return
    fp = fingerprint_project(project_id)
    record_ux_scenario_event(
        run_id=rid,
        milestone_code=MILESTONE_EXP_SCOPE,
        surface="webapp",
        project_fp=fp,
        extra={"export_scope": scope, "export_row_count": export_row_count},
    )
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
        extra=dl_extra,
    )
