"""Tests for issue-020 UX scenario baseline telemetry (no PostgreSQL)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from src import ux_scenario_telemetry as uxt


def test_milestone_codes_match_parity_matrix_ids() -> None:
    """Stable flow IDs must stay aligned with ``docs/migration_parity_matrix.md``."""
    expected = {"SB-CTX", "ENT-NEW-WRITE", "EDI-SAVE", "EXP-SCOPE", "EXP-DL"}
    assert expected <= uxt.MILESTONE_CODES_CRITICAL_V1


def test_fingerprint_project_is_stable_short_hash() -> None:
    fp = uxt.fingerprint_project("proj-abc")
    assert len(fp) == 16
    assert fp == uxt.fingerprint_project("proj-abc")
    assert fp != uxt.fingerprint_project("proj-other")


def test_record_event_writes_jsonl_when_dir_configured(tmp_path: Path) -> None:
    env = {uxt.UX_TELEMETRY_DIR_ENV: str(tmp_path)}
    uxt.record_ux_scenario_event(
        run_id="ux_test_run",
        milestone_code="SB-CTX",
        surface="streamlit",
        project_fp="a" * 16,
        environ=env,
    )
    files = list(tmp_path.glob("ux_scenario_*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["milestone_code"] == "SB-CTX"
    assert payload["surface"] == "streamlit"
    assert payload["scenario_id"] == uxt.SCENARIO_CRITICAL_V1
    assert payload["run_id"] == "ux_test_run"
    assert payload["project_fp"] == "a" * 16
    assert "monotonic_ns" in payload


def test_record_event_skips_file_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(uxt.UX_TELEMETRY_DIR_ENV, raising=False)
    uxt.record_ux_scenario_event(
        run_id="ux_test_run",
        milestone_code="SB-CTX",
        surface="streamlit",
        project_fp="b" * 16,
        environ=os.environ,
    )
    assert list(tmp_path.glob("*.jsonl")) == []


def test_emit_once_dedupe_per_key() -> None:
    emitted: set[str] = set()

    assert uxt.emit_once_per_session_key(emitted, "k1") is True
    assert uxt.emit_once_per_session_key(emitted, "k1") is False
    assert uxt.emit_once_per_session_key(emitted, "k2") is True


def test_record_error_includes_api_code_when_resolved(tmp_path: Path) -> None:
    env = {uxt.UX_TELEMETRY_DIR_ENV: str(tmp_path)}

    uxt.record_ux_error_event(
        run_id="ux_err",
        surface="streamlit",
        milestone_context="ENT-NEW-WRITE",
        project_fp="c" * 16,
        exception=PermissionError("role"),
        environ=env,
    )
    files = list(tmp_path.glob("ux_error_*.jsonl"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert payload["kind"] == "ux_error"
    assert payload["api_error_code"] == "FORBIDDEN"


def test_invalid_milestone_raises() -> None:
    with pytest.raises(ValueError, match="milestone_code"):
        uxt.record_ux_scenario_event(
            run_id="x",
            milestone_code="NOT-A-CODE",
            surface="streamlit",
            project_fp="d" * 16,
            environ={},
        )
