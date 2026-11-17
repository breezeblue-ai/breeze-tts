import threading

from fastapi.testclient import TestClient

from breeze_infer import api
from breeze_infer.api import app


def test_speech_rejects_whitespace_only_text(monkeypatch):
    # A whitespace-only string passes the Form(...) presence check, then
    # prepare_inputs raises a bare ValueError which FastAPI turns into a 500.
    # The batch endpoint already rejects blank entries with a 400.
    monkeypatch.setattr(api, "_request_lock", threading.Lock())
    client = TestClient(app, raise_server_exceptions=False)
    for text in ("   ", "\t\n"):
        response = client.post("/v1/audio/speech", data={"text": text})
        assert response.status_code == 400, (text, response.status_code, response.text)
        assert not api._request_lock.locked()
