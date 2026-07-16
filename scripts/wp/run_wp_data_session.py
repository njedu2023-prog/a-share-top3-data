from __future__ import annotations

import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time, timedelta
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

import tushare as ts

try:
    from scripts.wp.http_retry import request_json
    from scripts.wp.wp_schedule import due_decision
except ModuleNotFoundError:  # Executed as python scripts/wp/run_wp_data_session.py.
    from http_retry import request_json
    from wp_schedule import due_decision


CN_TZ = ZoneInfo("Asia/Shanghai")
INTERVAL_SECONDS = int(os.environ.get("WP_SESSION_INTERVAL_SECONDS", "600"))
SCHEDULE_GRACE_SECONDS = int(os.environ.get("WP_SCHEDULE_GRACE_SECONDS", "600"))
MIN_RUN_SPACING_SECONDS = int(os.environ.get("WP_MIN_RUN_SPACING_SECONDS", "300"))
MORNING_PREP = time(9, 15)
MORNING_START = time(9, 25)
MORNING_END = time(11, 35)
AFTERNOON_PREP = time(12, 45)
AFTERNOON_START = time(12, 55)
AFTERNOON_END = time(15, 10)


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def session_window(now: datetime) -> tuple[str, datetime, datetime] | None:
    today = now.date()
    morning_prep = datetime.combine(today, MORNING_PREP, CN_TZ)
    morning_start = datetime.combine(today, MORNING_START, CN_TZ)
    morning_end = datetime.combine(today, MORNING_END, CN_TZ)
    afternoon_prep = datetime.combine(today, AFTERNOON_PREP, CN_TZ)
    afternoon_start = datetime.combine(today, AFTERNOON_START, CN_TZ)
    afternoon_end = datetime.combine(today, AFTERNOON_END, CN_TZ)
    grace = timedelta(seconds=SCHEDULE_GRACE_SECONDS)
    if morning_prep <= now <= morning_end + grace:
        return "morning", morning_start, morning_end
    if afternoon_prep <= now <= afternoon_end + grace:
        return "afternoon", afternoon_start, afternoon_end
    return None


def fixed_slots(start: datetime, end: datetime) -> list[datetime]:
    slots: list[datetime] = []
    target = start
    while target <= end:
        slots.append(target)
        target += timedelta(seconds=INTERVAL_SECONDS)
    if not slots or slots[-1] != end:
        slots.append(end)
    return slots


def in_run_window(now: datetime) -> bool:
    today = now.date()
    morning_start = datetime.combine(today, MORNING_START, CN_TZ)
    morning_end = datetime.combine(today, MORNING_END, CN_TZ) + timedelta(seconds=SCHEDULE_GRACE_SECONDS)
    afternoon_start = datetime.combine(today, AFTERNOON_START, CN_TZ)
    afternoon_end = datetime.combine(today, AFTERNOON_END, CN_TZ) + timedelta(seconds=SCHEDULE_GRACE_SECONDS)
    return morning_start <= now <= morning_end or afternoon_start <= now <= afternoon_end


def is_trade_day(token: str, day: str) -> bool:
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date=day, end_date=day)
    return bool(len(cal) and int(cal.iloc[0].get("is_open", 0)) == 1)


def dispatch_wp_update() -> None:
    token = os.environ.get("WP_TRIGGER_TOKEN", "").strip()
    target_repo = os.environ.get("WP_TRIGGER_REPO", "njedu2023-prog/WP").strip()
    if not token:
        print("::warning::WP_TRIGGER_TOKEN is not configured; WP schedule and monitor remain the fallback.")
        return

    payload = {
        "event_type": "wp_data_ready",
        "client_payload": {
            "source_repository": os.environ.get("GITHUB_REPOSITORY", "njedu2023-prog/a-share-top3-data"),
            "completed_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    try:
        request_json(
            f"https://api.github.com/repos/{target_repo}/dispatches",
            token=token,
            method="POST",
            payload=payload,
            user_agent="WP-upstream-dispatcher",
        )
        print(f"Triggered WP through repository_dispatch: {target_repo}")
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        print(f"::warning::Cannot trigger WP directly: {exc}; schedule and monitor remain the fallback.")


def run_once() -> None:
    env = os.environ.copy()
    env.setdefault("ENABLE_AUCTION", "1")
    env.setdefault("ENABLE_MINUTE", "1")
    env.setdefault("REALTIME_MINUTE_ONLY", "1")
    env.setdefault("ENABLE_MARKET_MINUTE_SCAN", "1")
    env.setdefault("TRY_FULL_MARKET_MINUTE", "0")
    env.setdefault("REALTIME_QUOTE_CHUNK_SIZE", "300")
    env.setdefault("MINUTE_FREQ", "1min")
    env.setdefault("MAX_MINUTE_SYMBOLS", "6000")
    env.setdefault("WP_INTRADAY_MIN_PCT", "8")
    subprocess.run([sys.executable, "scripts/fetch_daily_snapshots.py"], check=True, env=env)
    subprocess.run([sys.executable, "scripts/wp/wp_pipeline.py"], check=True, env=env)
    current = now_cn()
    commit_paths = [
        "data/wp/latest",
        f"data/wp/reports_input/{current:%Y}/{current:%Y%m%d}",
    ]
    subprocess.run(
        [sys.executable, "scripts/wp/github_commit_paths.py", "Update WP data", *commit_paths],
        check=True,
        env=env,
    )
    dispatch_wp_update()


def require_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured.")
    return token


def run_once_if_due() -> None:
    current = now_cn()
    decision = due_decision(current, "data/wp/latest/wp_manifest.json")
    if not decision.should_run:
        target = decision.target_slot.strftime("%H:%M") if decision.target_slot else "none"
        print(f"Skip WP data update: {decision.reason}; target={target}")
        return

    token = require_token()
    trade_date = current.strftime("%Y%m%d")
    if not is_trade_day(token, trade_date):
        print(f"Skip WP data update: {trade_date} is not an A-share trading day.")
        return
    target_text = decision.target_slot.strftime("%Y-%m-%d %H:%M:%S") if decision.target_slot else ""
    os.environ["WP_TARGET_SLOT"] = target_text
    print(
        f"WP data catch-up update started: actual={current:%Y-%m-%d %H:%M:%S}, "
        f"target={target_text}, reason={decision.reason}"
    )
    run_once()
    print(f"WP data single update completed: {now_cn():%Y-%m-%d %H:%M:%S}")


def run_session() -> None:
    token = require_token()
    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    if not is_trade_day(token, trade_date):
        print(f"Skip WP data session: {trade_date} is not an A-share trading day.")
        return

    window = session_window(current)
    if window is None:
        print(f"Skip WP data session outside trading session prep/window: {current:%Y-%m-%d %H:%M:%S}")
        return

    session_name, start_dt, end_dt = window
    slots = fixed_slots(start_dt, end_dt)
    successful_runs = 0
    failed_runs = 0
    last_run_completed: datetime | None = None
    print(
        f"WP data {session_name} session targets: "
        + ", ".join(slot.strftime("%H:%M") for slot in slots)
    )

    for slot in slots:
        current = now_cn()
        if current < slot:
            wait_seconds = max(0.0, (slot - current).total_seconds())
            print(f"Wait for fixed WP data slot {slot:%Y-%m-%d %H:%M:%S}, sleep={wait_seconds:.0f}s")
            time_module.sleep(wait_seconds)
            current = now_cn()

        late_seconds = (current - slot).total_seconds()
        if late_seconds > SCHEDULE_GRACE_SECONDS:
            print(f"Skip expired WP data slot {slot:%H:%M}; late={late_seconds:.0f}s")
            continue
        if (
            last_run_completed is not None
            and slot != end_dt
            and (current - last_run_completed).total_seconds() < MIN_RUN_SPACING_SECONDS
        ):
            print(f"Skip compressed WP data slot {slot:%H:%M}; previous run just completed.")
            continue

        print(f"WP data fixed-slot iteration started: target={slot:%H:%M}, actual={current:%H:%M:%S}")
        try:
            run_once()
            successful_runs += 1
            last_run_completed = now_cn()
            print(f"WP data fixed-slot iteration completed: {last_run_completed:%Y-%m-%d %H:%M:%S}")
        except (subprocess.CalledProcessError, HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            failed_runs += 1
            print(f"::error::WP data slot {slot:%H:%M} failed: {exc}")

    if successful_runs == 0:
        raise RuntimeError(f"WP data {session_name} session completed without a successful iteration.")
    print(
        f"WP data {session_name} session completed: {now_cn():%Y-%m-%d %H:%M:%S}; "
        f"success={successful_runs}, failed={failed_runs}"
    )


def main() -> None:
    mode = os.environ.get("WP_DATA_RUN_MODE", "once").strip().lower()
    if mode == "session":
        run_session()
    elif mode == "due":
        run_once_if_due()
    elif mode == "once":
        run_once()
    else:
        raise ValueError(f"Unsupported WP_DATA_RUN_MODE: {mode}")


if __name__ == "__main__":
    main()
