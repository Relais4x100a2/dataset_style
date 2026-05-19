"""Super-admin panneau technique saga (issue-019) — télémétrie et relance DLQ."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src.database import DeprovisionOp, UserRecord
from src.webapp import deps as webapp_deps
from src.webapp.app import create_slice_app
from src.webapp.super_admin_saga import counts_in_recent_window


def test_counts_in_recent_window_aligns_with_streamlit_card_states() -> None:
    """Métriques saga : mêmes états que les cartes Streamlit (fenêtre des N dernières ops)."""
    ops = [
        DeprovisionOp(
            operation_id="a",
            target_user_id="t1",
            actor_user_id="act",
            state="pending",
            retry_count=0,
            last_error="",
            next_retry_at="",
            quarantined_at="",
        ),
        DeprovisionOp(
            operation_id="b",
            target_user_id="t2",
            actor_user_id="act",
            state="quarantined",
            retry_count=3,
            last_error="x",
            next_retry_at="",
            quarantined_at="2020-01-01",
        ),
        DeprovisionOp(
            operation_id="c",
            target_user_id="t3",
            actor_user_id="act",
            state="db_done",
            retry_count=0,
            last_error="",
            next_retry_at="",
            quarantined_at="",
        ),
    ]
    got = counts_in_recent_window(ops)
    assert got == {"pending": 1, "provider_done": 0, "failed": 0, "quarantined": 1}


def test_super_admin_saga_telemetry_returns_403_for_non_super_admin() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="u1",
        email="curator@example.com",
        display_name="C",
        is_super_admin=False,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.get(
            "/api/super-admin/saga/telemetry",
            headers={"Authorization": "Bearer t"},
        )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN"


def test_super_admin_saga_telemetry_returns_200_contract() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    op = DeprovisionOp(
        operation_id="op1",
        target_user_id="t1",
        actor_user_id="admin1",
        state="quarantined",
        retry_count=2,
        last_error="boom",
        next_retry_at="",
        quarantined_at="2021-06-01",
    )
    fake_telemetry = {
        "recentOpsLimit": 100,
        "dlqPreviewLimit": 50,
        "retryQueuePreviewLimit": 50,
        "stateCountsInRecentWindow": {
            "pending": 0,
            "provider_done": 0,
            "failed": 0,
            "quarantined": 1,
        },
        "totalsByState": {
            "pending": 0,
            "provider_done": 0,
            "db_done": 0,
            "completed": 0,
            "failed": 0,
            "quarantined": 1,
        },
        "recentOps": [],
        "dlqOps": [
            {
                "operationId": op.operation_id,
                "targetUserId": op.target_user_id,
                "actorUserId": op.actor_user_id,
                "state": op.state,
                "retryCount": op.retry_count,
                "lastError": op.last_error,
                "nextRetryAt": None,
                "quarantinedAt": op.quarantined_at,
            }
        ],
        "retryQueueOps": [],
    }
    with patch("src.webapp.app.build_deprovision_telemetry_payload", return_value=fake_telemetry):
        with TestClient(app) as client:
            r = client.get(
                "/api/super-admin/saga/telemetry",
                headers={"Authorization": "Bearer t"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["stateCountsInRecentWindow"]["quarantined"] == 1
    assert body["totalsByState"]["quarantined"] == 1
    assert len(body["dlqOps"]) == 1
    assert body["dlqOps"][0]["operationId"] == "op1"


def test_super_admin_saga_replay_requires_confirm_true() -> None:
    app = create_slice_app(engine=MagicMock())
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with TestClient(app) as client:
        r = client.post(
            "/api/super-admin/saga/replay-quarantined",
            headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
            json={"confirm": False, "operationId": "x"},
        )
    assert r.status_code == 422


def test_super_admin_saga_replay_maps_runtime_error_to_bad_request() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    with patch(
        "src.webapp.app.replay_quarantined_operation",
        side_effect=RuntimeError("Opération non rejouable (introuvable ou non quarantined)."),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/super-admin/saga/replay-quarantined",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"confirm": True, "operationId": "missing"},
            )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "BAD_REQUEST"
    assert "rejouable" in err["message"].lower()


def test_super_admin_saga_replay_returns_refreshed_telemetry() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    telemetry = {"recentOpsLimit": 100, "dlqOps": []}
    with (
        patch("src.webapp.app.replay_quarantined_operation") as replay,
        patch("src.webapp.app.build_deprovision_telemetry_payload", return_value=telemetry),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/super-admin/saga/replay-quarantined",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"confirm": True, "operationId": "op99"},
            )
    assert r.status_code == 200
    replay.assert_called_once_with(engine, "admin1", "op99")
    body = r.json()
    assert body["status"] == "ok"
    assert body["telemetry"] == telemetry


def test_super_admin_saga_replay_accepts_snake_case_operation_id() -> None:
    engine = MagicMock()
    app = create_slice_app(engine=engine)
    user = UserRecord(
        user_id="admin1",
        email="admin@example.com",
        display_name="A",
        is_super_admin=True,
    )
    app.dependency_overrides[webapp_deps.require_app_user] = lambda: user
    telemetry = {"recentOpsLimit": 100}
    with (
        patch("src.webapp.app.replay_quarantined_operation") as replay,
        patch("src.webapp.app.build_deprovision_telemetry_payload", return_value=telemetry),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/api/super-admin/saga/replay-quarantined",
                headers={"Authorization": "Bearer t", "Content-Type": "application/json"},
                json={"confirm": True, "operation_id": "op-snake"},
            )
    assert r.status_code == 200
    replay.assert_called_once_with(engine, "admin1", "op-snake")
