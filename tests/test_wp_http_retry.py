from __future__ import annotations

import io
import json
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from scripts.wp import http_retry


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _http_error(code: int, body: bytes = b"temporary") -> HTTPError:
    return HTTPError("https://api.github.test", code, "failed", Message(), io.BytesIO(body))


def test_request_json_retries_transient_http_and_network_errors(monkeypatch):
    outcomes = [_http_error(503), URLError("reset"), _Response({"ok": True})]
    sleeps: list[float] = []

    def fake_urlopen(_request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(http_retry, "urlopen", fake_urlopen)
    monkeypatch.setattr(http_retry.time, "sleep", sleeps.append)

    assert http_retry.request_json("https://api.github.test") == {"ok": True}
    assert sleeps == [2.0, 4.0]


def test_request_json_does_not_retry_non_transient_http_error(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(http_retry, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(_http_error(404)))
    monkeypatch.setattr(http_retry.time, "sleep", sleeps.append)

    with pytest.raises(HTTPError) as exc_info:
        http_retry.request_json("https://api.github.test")

    assert exc_info.value.code == 404
    assert sleeps == []
