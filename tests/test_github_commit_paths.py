from __future__ import annotations

import base64
import io
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from scripts.wp import github_commit_paths as commit_paths


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def http_error(code: int, body: bytes, **headers: str) -> HTTPError:
    message = Message()
    for key, value in headers.items():
        message[key.replace("_", "-")] = value
    return HTTPError("https://api.github.test", code, "failed", message, io.BytesIO(body))


def test_request_retries_transient_http_and_network_errors(monkeypatch):
    responses = [
        http_error(503, b"temporarily unavailable"),
        URLError("reset"),
        FakeResponse(b'{"ok": true}'),
    ]
    sleeps: list[int] = []

    def fake_urlopen(_req, timeout):
        assert timeout == 60
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(commit_paths, "urlopen", fake_urlopen)
    monkeypatch.setattr(commit_paths.time, "sleep", sleeps.append)

    assert commit_paths.request("GET", "https://api.github.test", "token") == {"ok": True}
    assert sleeps == [2, 4]


def test_request_rejects_non_transient_http_and_limits_error_body(monkeypatch):
    monkeypatch.setattr(
        commit_paths,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error(404, b"x" * 2000)),
    )
    with pytest.raises(commit_paths.GitHubRequestError, match=r"failed: 404") as exc_info:
        commit_paths.request("GET", "https://api.github.test", "token")
    assert exc_info.value.retryable is False
    assert len(str(exc_info.value)) < 1200


def test_graphql_commit_is_atomic(monkeypatch, tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.json"
    first.write_text("alpha", encoding="utf-8")
    second.write_text('{"value": 2}', encoding="utf-8")
    calls: list[tuple[str, dict]] = []

    def fake_graphql(query, variables, token):
        assert token == "token"
        calls.append((query, variables))
        if "query(" in query:
            return {"repository": {"ref": {"target": {"oid": "head-sha"}}}}
        return {"createCommitOnBranch": {"commit": {"oid": "commit-sha"}}}

    monkeypatch.setattr(commit_paths, "graphql_request", fake_graphql)
    result = commit_paths.commit_via_graphql(
        [(first, "data/first.txt"), (second, "data/second.json")],
        "owner/repo",
        "main",
        "Update data",
        "token",
    )

    assert result == "commit-sha"
    commit_input = calls[1][1]["input"]
    assert commit_input["expectedHeadOid"] == "head-sha"
    additions = commit_input["fileChanges"]["additions"]
    assert [item["path"] for item in additions] == ["data/first.txt", "data/second.json"]
    assert base64.b64decode(additions[0]["contents"]) == b"alpha"
