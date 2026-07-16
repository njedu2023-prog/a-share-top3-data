from __future__ import annotations

import json
import os
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts


CN_TZ = ZoneInfo("Asia/Shanghai")
DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
    "pct_chg",
]
LIMIT_FIELDS = ["ts_code", "trade_date", "up_limit", "down_limit"]


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def write_output(name: str, value: object) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def validate_frame(
    frame: pd.DataFrame | None,
    *,
    label: str,
    trade_date: str,
    required_columns: list[str],
    min_rows: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise RuntimeError(f"{label} returned no rows for {trade_date}")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns: {', '.join(missing)}")
    out = frame.copy()
    out["trade_date"] = out["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = out[out["trade_date"].eq(trade_date)].copy()
    if len(out) < min_rows:
        raise RuntimeError(
            f"{label} returned only {len(out)} rows for {trade_date}; minimum is {min_rows}"
        )
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    return out.drop_duplicates("ts_code", keep="last").sort_values("ts_code").reset_index(drop=True)


def fetch_with_retry(
    fetcher,
    *,
    label: str,
    trade_date: str,
    fields: list[str],
    min_rows: int,
    max_retry: int,
    retry_delay_seconds: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            frame = fetcher(trade_date=trade_date, fields=",".join(fields))
            validated = validate_frame(
                frame,
                label=label,
                trade_date=trade_date,
                required_columns=fields,
                min_rows=min_rows,
            )
            print(f"{label} close truth ready: trade_date={trade_date}, rows={len(validated)}")
            return validated
        except Exception as exc:
            last_error = exc
            print(f"::warning::{label} attempt {attempt}/{max_retry} failed: {exc}")
            if attempt < max_retry:
                time.sleep(retry_delay_seconds)
    raise RuntimeError(f"cannot publish {label} close truth: {last_error}") from last_error


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def is_open_trade_day(pro, trade_date: str) -> bool:
    calendar = pro.trade_cal(exchange="SSE", start_date=trade_date, end_date=trade_date)
    return bool(
        calendar is not None
        and not calendar.empty
        and int(calendar.iloc[0].get("is_open", 0)) == 1
    )


def main() -> None:
    current = now_cn()
    requested = os.environ.get("TRADE_DATE", "").strip()
    trade_date = requested or current.strftime("%Y%m%d")
    write_output("trade_date", trade_date)
    write_output("year", trade_date[:4])

    if not requested and current.time() < clock_time(15, 5):
        write_output("should_publish", "false")
        print(f"Skip close truth before 15:05 Beijing time: {current:%Y-%m-%d %H:%M:%S}")
        return

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    ts.set_token(token)
    pro = ts.pro_api()
    if not is_open_trade_day(pro, trade_date):
        write_output("should_publish", "false")
        print(f"Skip close truth: {trade_date} is not an A-share trading day")
        return

    max_retry = max(1, int(os.environ.get("WP_CLOSE_TRUTH_MAX_RETRY", "5")))
    retry_delay = max(1, int(os.environ.get("WP_CLOSE_TRUTH_RETRY_DELAY_SECONDS", "15")))
    min_rows = max(1, int(os.environ.get("WP_CLOSE_TRUTH_MIN_ROWS", "1000")))
    daily = fetch_with_retry(
        pro.daily,
        label="daily",
        trade_date=trade_date,
        fields=DAILY_FIELDS,
        min_rows=min_rows,
        max_retry=max_retry,
        retry_delay_seconds=retry_delay,
    )
    limits = fetch_with_retry(
        pro.stk_limit,
        label="stk_limit",
        trade_date=trade_date,
        fields=LIMIT_FIELDS,
        min_rows=min_rows,
        max_retry=max_retry,
        retry_delay_seconds=retry_delay,
    )

    output_dir = Path("data/raw") / trade_date[:4] / trade_date
    write_csv_atomic(daily[DAILY_FIELDS], output_dir / "daily.csv")
    write_csv_atomic(limits[LIMIT_FIELDS], output_dir / "stk_limit.csv")
    manifest = {
        "status": "ok",
        "trade_date": trade_date,
        "generated_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "daily_rows": int(len(daily)),
        "stk_limit_rows": int(len(limits)),
        "source": "tushare_close_truth",
    }
    (output_dir / "close_truth_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_output("should_publish", "true")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
