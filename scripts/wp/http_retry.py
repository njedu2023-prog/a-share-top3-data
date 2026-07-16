from __future__ import annotations

import json
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def _retry_after_seconds(exc: HTTPError, attempt: int) -> float:
    retry_after = str(exc.headers.get("Retry-After", "") or "").strip()
    try:
        return min(max(float(retry_after), 0.0), 30.0)
    except ValueError:
        return min(float(2**attempt), 30.0)


def _is_retryable_http_error(exc: HTTPError, body: str) -> bool:
    if exc.code in RETRYABLE_HTTP_STATUS:
        return True
    lowered = body.lower()
    return exc.code == 403 and ("rate limit" in lowered or "secondary limit" in lowered)


def request_json(
    url: str,
    *,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
    user_agent: str = "WP-data-control-plane",
    attempts: int = 4,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except HTTPError as exc:
            body = exc.read(512).decode("utf-8", errors="replace")
            if attempt >= attempts or not _is_retryable_http_error(exc, body):
                raise
            delay = _retry_after_seconds(exc, attempt)
            print(
                f"::warning::Transient HTTP {exc.code}; retry {attempt}/{attempts - 1} "
                f"in {delay:g}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
        except (URLError, TimeoutError, OSError) as exc:
            if attempt >= attempts:
                raise
            delay = min(float(2**attempt), 30.0)
            print(
                f"::warning::Transient network error {type(exc).__name__}; "
                f"retry {attempt}/{attempts - 1} in {delay:g}s.",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError("HTTP request retry loop exited unexpectedly.")
