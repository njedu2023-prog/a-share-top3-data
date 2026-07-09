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
PREP_START = time(9, 0)
RUN_START = time(9, 25)
LUNCH_START = time(11, 35)
LUNCH_END = time(12, 55)
RUN_END = time(15, 10)


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_window(now: datetime) -> tuple[datetime, datetime, datetime, datetime] | None:
    today = now.date()
    prep_dt = datetime.combine(today, PREP_START, CN_TZ)
    start_dt = datetime.combine(today, RUN_START, CN_TZ)
    lunch_start_dt = datetime.combine(today, LUNCH_START, CN_TZ)
    lunch_end_dt = datetime.combine(today, LUNCH_END, CN_TZ)
    end_dt = datetime.combine(today, RUN_END, CN_TZ)
    if prep_dt <= now <= end_dt:
        return start_dt, lunch_start_dt, lunch_end_dt, end_dt
    return None


def in_run_window(now: datetime) -> bool:
    current = now.time()
    return (RUN_START <= current <= LUNCH_START) or (LUNCH_END <= current <= RUN_END)


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


def run_once_if_due() -> None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        print("Skip WP data update: TUSHARE_TOKEN is not configured.")
        return

    current = now_cn()
    trade_date = current.strftime("%Y%m%d")
    if not is_trade_day(token, trade_date):
        print(f"Skip WP data update: {trade_date} is not an A-share trading day.")
        return
    if not in_run_window(current):
        print(f"Skip WP data update outside A-share trading window: {current:%Y-%m-%d %H:%M:%S}")
        return

    print(f"WP data single update started: {current:%Y-%m-%d %H:%M:%S}")
    run_once()
    print(f"WP data single update completed: {now_cn():%Y-%m-%d %H:%M:%S}")


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

    window = today_window(current)
    if window is None:
        print(f"Skip WP data session outside trading session prep/window: {current:%Y-%m-%d %H:%M:%S}")
        return

    start_dt, lunch_start_dt, lunch_end_dt, end_dt = window
    if current < start_dt:
        wait_seconds = max(0.0, (start_dt - current).total_seconds())
        print(f"Wait until A-share session start: {start_dt:%Y-%m-%d %H:%M:%S}, wait={wait_seconds:.0f}s")
        time_module.sleep(wait_seconds)

    while now_cn() <= end_dt:
        iteration_start = now_cn()
        if lunch_start_dt <= iteration_start < lunch_end_dt:
            sleep_seconds = max(0.0, (lunch_end_dt - iteration_start).total_seconds())
            print(f"Pause during A-share lunch break until {lunch_end_dt:%Y-%m-%d %H:%M:%S}, sleep={sleep_seconds:.0f}s")
            time_module.sleep(sleep_seconds)
            continue
        print(f"WP data iteration started: {iteration_start:%Y-%m-%d %H:%M:%S}")
        run_once()
        next_at = iteration_start + timedelta(seconds=INTERVAL_SECONDS)
        current = now_cn()
        next_boundary = lunch_start_dt if current < lunch_start_dt < next_at else end_dt
        sleep_seconds = min((next_at - current).total_seconds(), (next_boundary - current).total_seconds())
        if sleep_seconds <= 0:
            continue
        print(f"Next WP data iteration at {next_at:%Y-%m-%d %H:%M:%S}, sleep={sleep_seconds:.0f}s")
        time_module.sleep(sleep_seconds)

    print(f"WP data session completed: {now_cn():%Y-%m-%d %H:%M:%S}")


def main() -> None:
    mode = os.environ.get("WP_DATA_RUN_MODE", "once").strip().lower()
    if mode == "session":
        run_session()
        return
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch" and os.environ.get("WP_FORCE_SESSION") == "1":
        run_once()
        return
    run_once_if_due()


if __name__ == "__main__":
    main()
