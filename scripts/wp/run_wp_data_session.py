from __future__ import annotations

import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import tushare as ts


CN_TZ = ZoneInfo("Asia/Shanghai")
INTERVAL_SECONDS = int(os.environ.get("WP_SESSION_INTERVAL_SECONDS", "600"))
SESSIONS = (
    (time(9, 0), time(9, 25), time(11, 35)),
    (time(12, 30), time(12, 55), time(15, 10)),
)


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_session(now: datetime) -> tuple[datetime, datetime] | None:
    today = now.date()
    for prep_start, run_start, run_end in SESSIONS:
        prep_dt = datetime.combine(today, prep_start, CN_TZ)
        start_dt = datetime.combine(today, run_start, CN_TZ)
        end_dt = datetime.combine(today, run_end, CN_TZ)
        if prep_dt <= now <= end_dt:
            return start_dt, end_dt
    return None


def is_trade_day(token: str, day: str) -> bool:
    ts.set_token(token)
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date=day, end_date=day)
    return bool(len(cal) and int(cal.iloc[0].get("is_open", 0)) == 1)


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
    subprocess.run(
        [sys.executable, "scripts/wp/github_commit_paths.py", "Update WP data", "data/wp", "data/latest"],
        check=True,
        env=env,
    )


def run_session() -> None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        print("Skip WP data session: TUSHARE_TOKEN is not configured.")
        return

    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    if not is_trade_day(token, trade_date):
        print(f"Skip WP data session: {trade_date} is not an A-share trading day.")
        return

    session = today_session(current)
    if session is None:
        print(f"Skip WP data session outside trading session prep/window: {current:%Y-%m-%d %H:%M:%S}")
        return

    start_dt, end_dt = session
    if current < start_dt:
        wait_seconds = max(0.0, (start_dt - current).total_seconds())
        print(f"Wait until A-share session start: {start_dt:%Y-%m-%d %H:%M:%S}, wait={wait_seconds:.0f}s")
        time_module.sleep(wait_seconds)

    while now_cn() <= end_dt:
        iteration_start = now_cn()
        print(f"WP data iteration started: {iteration_start:%Y-%m-%d %H:%M:%S}")
        run_once()
        next_at = iteration_start + timedelta(seconds=INTERVAL_SECONDS)
        current = now_cn()
        sleep_seconds = min((next_at - current).total_seconds(), (end_dt - current).total_seconds())
        if sleep_seconds <= 0:
            continue
        print(f"Next WP data iteration at {next_at:%Y-%m-%d %H:%M:%S}, sleep={sleep_seconds:.0f}s")
        time_module.sleep(sleep_seconds)

    print(f"WP data session completed: {now_cn():%Y-%m-%d %H:%M:%S}")


def main() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        run_once()
        return
    run_session()


if __name__ == "__main__":
    main()
