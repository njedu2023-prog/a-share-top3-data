from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WP_ROOT = ROOT / "data" / "wp"
LATEST = WP_ROOT / "latest"
DATA_LATEST = ROOT / "data" / "latest"
RAW_ROOT = ROOT / "data" / "raw"
RAW_BASE_URL = "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main"

SCHEMA = [
    "trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low",
    "close", "pre_close", "pct_chg", "amount", "volume", "turnover_rate",
    "volume_ratio", "sector_name", "sector_rank", "sector_limitup_count",
    "sector_gt6_count", "sector_amount_ratio", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
]


def read_csv_source(relative_path: str) -> pd.DataFrame:
    local_path = ROOT / relative_path
    if local_path.exists() and local_path.stat().st_size > 0:
        return pd.read_csv(local_path, encoding="utf-8-sig")
    url = f"{RAW_BASE_URL}/{relative_path}"
    try:
        with urlopen(url, timeout=30) as resp:
            if resp.status >= 400:
                return pd.DataFrame()
        return pd.read_csv(url, encoding="utf-8-sig")
    except (URLError, OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def now_cn() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) + pd.Timedelta(hours=8)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "代码": "ts_code",
        "名称": "name",
        "收盘价": "close",
        "最新价": "price",
        "昨收": "pre_close",
        "涨跌幅": "pct_chg",
        "成交额": "amount",
        "所属板块": "sector_name",
        "板块": "sector_name",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = np.nan
    if out["price"].isna().all():
        out["price"] = out["close"]
    if out["pct_chg"].isna().all() and "close" in out and "pre_close" in out:
        out["pct_chg"] = np.where(pd.to_numeric(out["pre_close"], errors="coerce") > 0, (out["close"] / out["pre_close"] - 1) * 100, 0)
    out["update_time"] = out["update_time"].fillna(now_cn().strftime("%Y-%m-%d %H:%M:%S"))
    out["trade_date"] = out["trade_date"].fillna(now_cn().strftime("%Y%m%d"))
    return out[SCHEMA]


def to_num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def latest_trade_date(*frames: pd.DataFrame) -> str:
    env_date = os.environ.get("TRADE_DATE", "").strip()
    if len(env_date) == 8 and env_date.isdigit():
        return env_date
    dates: list[str] = []
    for frame in frames:
        if not frame.empty and "trade_date" in frame.columns:
            dates.extend(frame["trade_date"].dropna().astype(str).str.replace("-", "", regex=False).tolist())
    dates = [item for item in dates if len(item) == 8 and item.isdigit()]
    if dates:
        return sorted(dates)[-1]
    return now_cn().strftime("%Y%m%d")


def previous_limitup_codes(trade_date: str) -> set[str]:
    dates = []
    year_dir = RAW_ROOT / trade_date[:4]
    if year_dir.exists():
        dates = sorted(path.name for path in year_dir.iterdir() if path.is_dir() and path.name < trade_date)
    if not dates:
        return set()
    prev_date = dates[-1]
    prev = read_csv_source(f"data/raw/{prev_date[:4]}/{prev_date}/limit_list_d.csv")
    if prev.empty or "ts_code" not in prev.columns:
        return set()
    return set(prev["ts_code"].dropna().astype(str).str.strip())


def build_from_latest_data() -> pd.DataFrame:
    daily = read_csv_source("data/latest/daily.csv")
    if daily.empty:
        return pd.DataFrame(columns=SCHEMA)
    daily_basic = read_csv_source("data/latest/daily_basic.csv")
    stock_basic = read_csv_source("data/latest/stock_basic.csv")
    stk_limit = read_csv_source("data/latest/stk_limit.csv")
    limit_list = read_csv_source("data/latest/limit_list_d.csv")
    hot_boards = read_csv_source("data/latest/hot_boards.csv")
    top_list = read_csv_source("data/latest/top_list.csv")
    intraday = read_csv_source("data/latest/intraday_features.csv")

    out = daily.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    trade_date = latest_trade_date(out, daily_basic, stk_limit, limit_list)
    out["trade_date"] = out.get("trade_date", trade_date).fillna(trade_date).astype(str)

    if not daily_basic.empty:
        keep = [c for c in ["ts_code", "turnover_rate", "volume_ratio", "total_mv", "float_mv"] if c in daily_basic.columns]
        out = out.merge(daily_basic[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    if not stock_basic.empty:
        keep = [c for c in ["ts_code", "name", "industry", "market", "list_date"] if c in stock_basic.columns]
        out = out.merge(stock_basic[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    if not stk_limit.empty:
        keep = [c for c in ["ts_code", "up_limit", "down_limit"] if c in stk_limit.columns]
        out = out.merge(stk_limit[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    if not intraday.empty:
        keep = [c for c in ["ts_code", "limit_touch_count", "open_board_count", "limitup_quality_score", "intraday_risk_score"] if c in intraday.columns]
        out = out.merge(intraday[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    close = to_num(out, "close")
    pct_chg = to_num(out, "pct_chg")
    out["pre_close"] = np.where((1 + pct_chg / 100) > 0, close / (1 + pct_chg / 100), np.nan)
    out["price"] = close
    out["volume"] = to_num(out, "vol")
    out["amount"] = to_num(out, "amount") * 1000
    out["sector_name"] = out.get("industry", pd.Series("未分类", index=out.index)).fillna("未分类").astype(str)

    current_limit_codes = set()
    if not limit_list.empty and "ts_code" in limit_list.columns:
        current_limit_codes = set(limit_list["ts_code"].dropna().astype(str).str.strip())
    prev_limit_codes = previous_limitup_codes(trade_date)
    up_limit = to_num(out, "up_limit")
    out["today_limit_up_price"] = up_limit
    out["prev_limit_up_price"] = np.nan
    out["today_limitup"] = np.where(out["ts_code"].isin(current_limit_codes) | ((up_limit > 0) & (close >= up_limit * 0.999)), 1, 0)
    out["pre_day_limitup"] = np.where(out["ts_code"].isin(prev_limit_codes), 1, 0)

    sector_gt6 = out.assign(_gt6=pct_chg >= 6).groupby("sector_name")["_gt6"].sum()
    sector_amount = out.groupby("sector_name")["amount"].sum()
    amount_median = float(sector_amount.median()) if len(sector_amount) else 0.0
    sector_metrics = pd.DataFrame({
        "sector_name": sector_gt6.index,
        "sector_gt6_count": sector_gt6.values,
        "sector_amount_ratio": [(sector_amount.get(name, 0.0) / amount_median) if amount_median > 0 else 1.0 for name in sector_gt6.index],
    })
    if not hot_boards.empty and "industry" in hot_boards.columns:
        boards = hot_boards.rename(columns={"industry": "sector_name", "rank": "sector_rank", "limit_up_count": "sector_limitup_count"})
        keep = [c for c in ["sector_name", "sector_rank", "sector_limitup_count"] if c in boards.columns]
        sector_metrics = sector_metrics.merge(boards[keep].drop_duplicates("sector_name"), on="sector_name", how="left")
    out = out.merge(sector_metrics, on="sector_name", how="left")

    if not top_list.empty and "ts_code" in top_list.columns:
        top = top_list.copy()
        top["dragon_tiger_flag"] = 1
        keep = [c for c in ["ts_code", "dragon_tiger_flag", "net_amount", "net_rate", "reason"] if c in top.columns]
        out = out.merge(top[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    out["sector_rank"] = to_num(out, "sector_rank", 99)
    out["sector_limitup_count"] = to_num(out, "sector_limitup_count", 0)
    out["sector_gt6_count"] = to_num(out, "sector_gt6_count", 0)
    out["sector_amount_ratio"] = to_num(out, "sector_amount_ratio", 1)
    out["ret_5d"] = pct_chg
    out["ret_20d"] = pct_chg
    return normalize(out)


def load_source() -> pd.DataFrame:
    source = os.environ.get("WP_SOURCE_CSV", "")
    if source and Path(source).exists():
        return pd.read_csv(source)
    return build_from_latest_data()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_snapshot() -> pd.DataFrame:
    current = now_cn()
    df = normalize(load_source())
    day_dir = WP_ROOT / "raw" / current.strftime("%Y") / current.strftime("%Y%m%d")
    write_csv(df, day_dir / f"wp_snapshot_{current.strftime('%H%M')}.csv")
    write_csv(df, LATEST / "wp_latest_snapshot.csv")
    return df


def build_features(snapshot: pd.DataFrame | None = None) -> pd.DataFrame:
    current = now_cn()
    df = normalize(snapshot if snapshot is not None else pd.read_csv(LATEST / "wp_latest_snapshot.csv"))
    write_csv(df, WP_ROOT / "features" / current.strftime("%Y") / current.strftime("%Y%m%d") / "wp_features.csv")
    write_csv(df, LATEST / "wp_latest_features.csv")
    return df


def build_candidates(features: pd.DataFrame | None = None) -> pd.DataFrame:
    current = now_cn()
    df = normalize(features if features is not None else pd.read_csv(LATEST / "wp_latest_features.csv"))
    if not df.empty:
        mask = (pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0) >= 6) & (pd.to_numeric(df["pre_day_limitup"], errors="coerce").fillna(0) != 1) & (pd.to_numeric(df["today_limitup"], errors="coerce").fillna(0) != 1)
        df = df.loc[mask].copy()
    write_csv(df, WP_ROOT / "candidates" / current.strftime("%Y") / current.strftime("%Y%m%d") / "wp_candidates.csv")
    write_csv(df, LATEST / "wp_latest_candidates.csv")
    return df


def build_labels() -> pd.DataFrame:
    current = now_cn()
    labels = pd.DataFrame(columns=["trade_date", "ts_code", "name", "next_trade_date", "label_t1_limitup"])
    write_csv(labels, WP_ROOT / "labels" / current.strftime("%Y") / current.strftime("%Y%m%d") / "wp_labels.csv")
    return labels


def build_rank_input(candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    current = now_cn()
    df = normalize(candidates if candidates is not None else pd.read_csv(LATEST / "wp_latest_candidates.csv"))
    write_csv(df, WP_ROOT / "reports_input" / current.strftime("%Y") / current.strftime("%Y%m%d") / "wp_rank_input.csv")
    write_csv(df, LATEST / "wp_latest_rank_input.csv")
    return df


def healthcheck(rank_input: pd.DataFrame | None = None) -> dict:
    df = rank_input if rank_input is not None else pd.read_csv(LATEST / "wp_latest_rank_input.csv")
    payload = {
        "generated_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ok" if not df.empty else "empty_schema_ready",
        "row_count": int(len(df)),
        "candidate_count": int(len(df)),
        "columns": list(df.columns),
        "latest_files": [
            "wp_latest_snapshot.csv",
            "wp_latest_features.csv",
            "wp_latest_candidates.csv",
            "wp_latest_rank_input.csv",
            "wp_manifest.json",
        ],
    }
    ensure(LATEST)
    (LATEST / "wp_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (LATEST / "wp_data_healthcheck.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_all() -> dict:
    snapshot = build_snapshot()
    features = build_features(snapshot)
    candidates = build_candidates(features)
    build_labels()
    rank_input = build_rank_input(candidates)
    return healthcheck(rank_input)


if __name__ == "__main__":
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
