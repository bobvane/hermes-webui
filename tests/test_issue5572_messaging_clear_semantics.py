"""Regression coverage for #5572 imported messaging clear semantics."""

from __future__ import annotations

import copy
from collections import OrderedDict
from io import BytesIO
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_agent_modules


def _msg(role: str, content: str, ts: float, mid: str) -> dict:
    return {"id": mid, "role": role, "content": content, "timestamp": ts}


def _install_isolated_session_env(monkeypatch, tmp_path):
    import api.config as config
    import api.models as models
    import api.profiles as profiles
    import api.routes as routes

    monkeypatch.setattr(config, "STATE_DIR", tmp_path, raising=False)
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(config, "SESSION_DIR", session_dir, raising=False)
    monkeypatch.setattr(config, "SESSION_INDEX_FILE", session_dir / "_index.json", raising=False)
    monkeypatch.setattr(models, "SESSION_DIR", session_dir, raising=False)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json", raising=False)
    monkeypatch.setattr(models, "SESSIONS", OrderedDict(), raising=False)
    monkeypatch.setattr(profiles, "get_active_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(models, "_active_state_db_path", lambda: tmp_path / "state.db", raising=False)
    monkeypatch.setattr(routes, "_active_state_db_path", lambda: tmp_path / "state.db", raising=False)
    monkeypatch.setattr(config, "_evict_session_agent", lambda _sid: None, raising=False)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _post_clear(monkeypatch, sid: str):
    import api.routes as routes

    body = b'{"session_id":"%s"}' % sid.encode("utf-8")
    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)

    captured = {}

    def fake_j(_handler, payload, status=200, extra_headers=None):
        captured["payload"] = payload
        captured["status"] = status
        captured["extra_headers"] = extra_headers

    monkeypatch.setattr(routes, "j", fake_j)

    handler = SimpleNamespace(
        headers={"Content-Length": str(len(body))},
        rfile=BytesIO(body),
    )
    routes.handle_post(handler, SimpleNamespace(path="/api/session/clear"))
    return captured


@pytest.mark.parametrize(
    ("source_tag", "source_label"),
    [("telegram", "Telegram"), ("discord", "Discord")],
)
def test_session_clear_preserves_imported_messaging_transcript_and_blocks_state_db_replay(
    monkeypatch,
    tmp_path,
    source_tag,
    source_label,
):
    import api.routes as routes
    from api.models import Session, merge_session_messages_append_only

    _install_isolated_session_env(monkeypatch, tmp_path)

    sid = f"issue5572_clear_{source_tag}"
    session = Session(
        session_id=sid,
        title=f"Imported {source_label}",
        workspace=str(tmp_path),
        model="test-model",
        messages=[
            _msg("user", f"{source_label} sidecar prompt", 1.0, f"{source_tag}-u1"),
            _msg("assistant", f"{source_label} sidecar reply", 2.0, f"{source_tag}-a1"),
        ],
        context_messages=[
            _msg("user", f"{source_label} sidecar prompt", 1.0, f"{source_tag}-cu1"),
            _msg("assistant", f"{source_label} sidecar reply", 2.0, f"{source_tag}-ca1"),
        ],
        tool_calls=[{"id": f"{source_tag}-tool-1", "function": {"name": "terminal"}}],
        created_at=1000.0,
        updated_at=1001.0,
        active_stream_id="stale-stream",
        pending_user_message="pending prompt",
        pending_attachments=[{"name": "pending.txt"}],
        pending_started_at=1002.0,
        pending_user_source="webui",
        is_cli_session=True,
        source_tag=source_tag,
        raw_source=source_tag,
        session_source="messaging",
        source_label=source_label,
    )
    session.save(touch_updated_at=False)

    external_messages = [
        _msg("user", f"{source_label} external prompt", 10.0, f"{source_tag}-ext-u1"),
        _msg("assistant", f"{source_label} external reply", 11.0, f"{source_tag}-ext-a1"),
    ]
    external_before = copy.deepcopy(external_messages)
    state_db_messages = [
        _msg("user", f"{source_label} state prompt", 20.0, f"{source_tag}-db-u1"),
        _msg("assistant", f"{source_label} state reply", 21.0, f"{source_tag}-db-a1"),
    ]
    state_db_before = copy.deepcopy(state_db_messages)

    captured = _post_clear(monkeypatch, sid)

    assert captured["status"] == 200
    assert captured["payload"]["ok"] is True
    assert captured["payload"]["session"]["title"] == "Untitled"
    assert captured["payload"]["session"]["is_cli_session"] is True
    assert captured["payload"]["session"]["source_tag"] == source_tag
    assert captured["payload"]["session"]["raw_source"] == source_tag
    assert captured["payload"]["session"]["session_source"] == "messaging"
    assert captured["payload"]["session"]["source_label"] == source_label

    loaded = Session.load(sid)
    assert loaded is not None
    assert loaded.is_cli_session is True
    assert loaded.session_source == "messaging"
    assert loaded.source_tag == source_tag
    assert loaded.raw_source == source_tag
    assert loaded.source_label == source_label
    assert loaded.messages == []
    assert loaded.context_messages == []
    assert loaded.tool_calls == []
    assert loaded.truncation_watermark == 0.0
    assert loaded.truncation_boundary == 0.0
    assert loaded.active_stream_id is None
    assert loaded.pending_user_message is None
    assert loaded.pending_attachments == []
    assert loaded.pending_started_at is None
    assert loaded.pending_user_source is None
    assert loaded.title == "Untitled"

    persisted = loaded.path.read_text(encoding="utf-8")
    assert '"messages": []' in persisted
    assert '"context_messages": []' in persisted
    assert '"tool_calls": []' in persisted
    assert '"truncation_watermark": 0.0' in persisted
    assert '"truncation_boundary": 0.0' in persisted
    assert '"pending_user_message": null' in persisted
    assert '"pending_started_at": null' in persisted
    assert '"pending_user_source": null' in persisted

    assert external_messages == external_before
    assert state_db_messages == state_db_before

    display_messages = routes._merged_session_messages_for_display(
        loaded,
        cli_messages=external_messages,
    )
    assert display_messages == external_messages

    merged = merge_session_messages_append_only(
        loaded.messages,
        state_db_messages,
        truncation_watermark=loaded.truncation_watermark,
        truncation_boundary=loaded.truncation_boundary,
    )
    assert merged == []
