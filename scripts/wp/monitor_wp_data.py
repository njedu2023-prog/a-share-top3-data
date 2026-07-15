from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")
REPO = os.environ.get("GITHUB_REPOSITORY", "njedu2023-prog/a-share-top3-data")
WORKFLOW = os.environ.get("WP_DATA_WORKFLOW", "run_wp_data_10min.yml")
MANIFEST_PATH = "data/wp/latest/wp_manifest.json"
MAX_AGE_MIN = float(os.environ.get("WP_DATA_MONITOR_MAX_AGE_MIN", "25"))
SESSION_GRACE_MIN = float(os.environ.get("WP_DATA_MONITOR_SESSION_GRACE_MIN", "25"))


def now_cn() -> datetime:
    return datetime.now(CN_TZ).replace(tzinfo=None)


def in_trade_window(now: datetime) -> bool:
    return time(9, 25) <= now.time() <= time(11, 35) or time(12, 55) <= now.time() <= time(15, 10)


def session_start(now: datetime) -> datetime | None:
    if time(9, 25) <= now.time() <= time(11, 35):
        return datetime.combine(now.date(), time(9, 25))
    if time(12, 55) <= now.time() <= time(15, 10):
        return datetime.combine(now.date(), time(12, 55))
    return None


def parse_dt(value: Any) -> datetime | None:
    text_value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text_value[:19], fmt)
        except ValueError:
            continue
    return None


def request_json(
    url: str,
    token: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "WP-data-monitor",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def read_manifest(token: str) -> dict[str, Any]:
    encoded = quote(MANIFEST_PATH, safe="/")
    payload = request_json(f"https://api.github.com/repos/{REPO}/contents/{encoded}?ref=main", token=token)
    content = "".join(str(payload.get("content", "")).split())
    if payload.get("encoding") != "base64" or not content:
        raise RuntimeError("Unsupported upstream manifest payload.")
    return json.loads(base64.b64decode(content).decode("utf-8-sig"))


def is_trade_day(token: str, day: str) -> bool:
    payload = request_json(
        "https://api.tushare.pro",
        method="POST",
        payload={
            "api_name": "trade_cal",
            "token": token,
            "params": {"exchange": "SSE", "start_date": day, "end_date": day},
            "fields": "cal_date,is_open",
        },
    )
    if int(payload.get("code", -1)) != 0:
        raise RuntimeError(f"Tushare trade_cal failed: {payload.get('msg')}")
    fields = payload.get("data", {}).get("fields", [])
    items = payload.get("data", {}).get("items", [])
    if not items or "is_open" not in fields:
        return False
    return int(items[0][fields.index("is_open")]) == 1


def repair_run_active(token: str) -> bool:
    payload = request_json(
        f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW}/runs?event=workflow_dispatch&per_page=10",
        token=token,
    )
    return any(run.get("status") in {"queued", "in_progress", "waiting", "pending"} for run in payload.get("workflow_runs", []))


def dispatch_repair(token: str) -> None:
    request_json(
        f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW}/dispatches",
        token=token,
        method="POST",
        payload={"ref": "main", "inputs": {"mode": "due"}},
    )


def monitor() -> int:
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    current = now_cn()
    today = current.strftime("%Y%m%d")
    print(f"WP data monitor at {current:%Y-%m-%d %H:%M:%S} Asia/Shanghai")

    if not in_trade_window(current):
        print("Outside A-share trading window; no monitoring action.")
        return 0
    if not github_token:
        print("::error::GITHUB_TOKEN is not configured.")
        return 1
    if not tushare_token:
        print("::error::TUSHARE_TOKEN is not configured.")
        return 1

    try:
        if not is_trade_day(tushare_token, today):
            print(f"{today} is not an A-share trading day; no monitoring action.")
            return 0
        manifest = read_manifest(github_token)
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot evaluate WP data freshness: {exc}")
        return 1

    generated_at = parse_dt(manifest.get("generated_at"))
    age_min = (current - generated_at).total_seconds() / 60 if generated_at else None
    fresh = (
        manifest.get("status") == "ok"
        and str(manifest.get("source_trade_date")) == today
        and age_min is not None
        and age_min <= MAX_AGE_MIN
    )
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "source_trade_date": manifest.get("source_trade_date"),
                "generated_at": manifest.get("generated_at"),
                "age_min": age_min,
            },
            ensure_ascii=False,
        )
    )
    if fresh:
        print("WP upstream data is fresh.")
        return 0

    start = session_start(current)
    if start and (current - start).total_seconds() / 60 < SESSION_GRACE_MIN:
        print("::warning::Upstream is not fresh yet, but the trading-session startup grace is active.")
        return 0

    try:
        if repair_run_active(github_token):
            print("::warning::Upstream is stale and a repair run is already active.")
            return 0
        dispatch_repair(github_token)
        print("::warning::Upstream is stale; dispatched a guarded single-run repair.")
        return 0
    except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"::error::Cannot dispatch upstream repair: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(monitor())
