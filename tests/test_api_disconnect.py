"""Exercise real ASGI response lifetimes while retaining generator references."""

import asyncio
import threading
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from starlette.requests import ClientDisconnect

from breeze_infer import api


@pytest.fixture
def inference(monkeypatch, tmp_path):
    state = SimpleNamespace(closed=False, streams=[], path=tmp_path / "reference.wav")
    state.path.write_bytes(b"reference")

    class Runtime:
        sample_rate = 24000

        def iter_audio_chunks(self, *args, **kwargs):
            def generate():
                try:
                    while True:
                        yield SimpleNamespace(audio=np.zeros(32))
                finally:
                    state.closed = True

            stream = generate()
            state.streams.append(stream)
            return stream

    monkeypatch.setattr(api, "_request_lock", threading.Lock())
    monkeypatch.setattr(api, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(api, "prepare_inputs", lambda *args, **kwargs: {})
    for name in ("tokenizer", "audio_tokenizer", "model"):
        monkeypatch.setattr(api.app.state, name, None, raising=False)
    monkeypatch.setattr(api.app.state, "runtime", Runtime(), raising=False)

    async def save(upload):
        return state.path

    monkeypatch.setattr(api, "_save_upload", save)
    return state


async def response():
    return await api.speech(
        text="test",
        instruction=None,
        cfg_scale=1.0,
        ref_audio=SimpleNamespace(filename="reference.wav"),
        ref_text="reference",
        seed=42,
    )


@pytest.mark.parametrize(
    "mode",
    ["disconnect", "before_body", "send_error", "cancel", "model_error", "complete"],
)
def test_response_cleans_up(inference, mode, monkeypatch):
    async def run():
        reply = await response()
        sent = anyio.Event()
        if mode in ("complete", "model_error"):

            def chunks(*args, **kwargs):
                try:
                    yield SimpleNamespace(audio=np.zeros(32))
                    if mode == "model_error":
                        raise ValueError("model failed")
                finally:
                    inference.closed = True

            monkeypatch.setattr(api.app.state.runtime, "iter_audio_chunks", chunks)

        async def receive():
            await sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if mode == "before_body" and message["type"] == "http.response.start":
                sent.set()
                await anyio.sleep_forever()
            if message["type"] == "http.response.body" and message.get("body"):
                assert api._request_lock.locked()
                if mode == "send_error":
                    raise OSError("connection closed")
                sent.set()
                if mode == "cancel":
                    cancel_scope.cancel()
                    await anyio.sleep_forever()

        spec = "2.3" if mode in ("disconnect", "before_body", "cancel") else "2.4"
        expected = ClientDisconnect if mode == "send_error" else ValueError
        with anyio.CancelScope() as cancel_scope:
            if mode in ("send_error", "model_error"):
                with pytest.raises(expected):
                    await reply(
                        {"type": "http", "asgi": {"spec_version": spec}}, receive, send
                    )
            else:
                await reply(
                    {"type": "http", "asgi": {"spec_version": spec}}, receive, send
                )
        assert not api._request_lock.locked()
        assert not inference.path.exists()
        if mode != "before_body":
            assert inference.closed
        # Keep the old response and model generators alive across another acquisition.
        assert api._request_lock.acquire(blocking=False)
        for stream in inference.streams:
            stream.close()
        assert api._request_lock.locked()
        api._request_lock.release()

    asyncio.run(run())


def test_disconnect_waits_for_inflight_model_step(inference, monkeypatch):
    async def run():
        entered = anyio.Event()
        finish_step = threading.Event()

        def chunks(*args, **kwargs):
            try:
                anyio.from_thread.run_sync(entered.set)
                assert finish_step.wait(timeout=5)
                yield SimpleNamespace(audio=np.zeros(32))
            finally:
                inference.closed = True

        monkeypatch.setattr(api.app.state.runtime, "iter_audio_chunks", chunks)
        reply = await response()

        async def receive():
            await entered.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        async with anyio.create_task_group() as group:
            group.start_soon(
                reply, {"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send
            )
            await entered.wait()
            try:
                await anyio.sleep(0.02)
                assert api._request_lock.locked()
                with pytest.raises(api.HTTPException) as error:
                    await response()
                assert error.value.status_code == 409
                assert not inference.closed
            finally:
                finish_step.set()
        assert inference.closed
        assert not api._request_lock.locked()
        assert not inference.path.exists()

    asyncio.run(run())


def test_cancel_during_upload_releases_lock(inference, monkeypatch):
    async def save(upload):
        raise asyncio.CancelledError()

    monkeypatch.setattr(api, "_save_upload", save)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await response()
        assert not api._request_lock.locked()

    asyncio.run(run())


def test_cancelled_upload_removes_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api.tempfile, "tempdir", str(tmp_path))

    class Upload:
        filename = "reference.wav"

        async def read(self):
            raise asyncio.CancelledError()

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await api._save_upload(Upload())
        assert list(tmp_path.iterdir()) == []

    asyncio.run(run())
