from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WP_ROOT = ROOT / "data" / "wp"
LATEST = WP_ROOT / "latest"
DATA_LATEST = ROOT / "data" / "latest"
RAW_ROOT = ROOT / "data" / "raw"
RAW_BASE_URL = "https://raw.githubusercontent.com/njedu2023-prog/a-share-top3-data/main"
LAST_SOURCE_TRADE_DATE = ""

SCHEMA = [
    "trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low",
    "close", "pre_close", "pct_chg", "amount", "volume", "turnover_rate",
    "volume_ratio", "sector_name", "sector_rank", "sector_limitup_count",
    "sector_gt6_count", "sector_amount_ratio", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
    "amount_ratio_5d", "amount_ratio_20d", "turnover_rate_5d_avg",
    "close_position", "intraday_pullback_pct", "open_to_close_pct",
    "gap_open_pct", "amplitude", "high_20d_break", "platform_break_20d",
    "stage_high_20d", "dragon_tiger_flag", "dragon_tiger_net_rate",
    "dragon_tiger_reason", "limit_touch_count", "open_board_count",
    "limitup_quality_score", "intraday_risk_score",
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


def read_json_url(url: str):
    try:
        req = Request(url, headers={"Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError):
        return None


def now_cn() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) + pd.Timedelta(hours=8)


def target_trade_date() -> str:
    env_date = os.environ.get("TRADE_DATE", "").strip()
    if len(env_date) == 8 and env_date.isdigit():
        return env_date
    return now_cn().strftime("%Y%m%d")


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


def available_raw_dates(trade_date: str, lookback_days: int = 45) -> list[str]:
    start = (datetime.strptime(trade_date, "%Y%m%d") - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    dates: list[str] = []
    year_dir = RAW_ROOT / trade_date[:4]
    if year_dir.exists():
        dates.extend(path.name for path in year_dir.iterdir() if path.is_dir())
    if not dates:
        payload = read_json_url(f"https://api.github.com/repos/njedu2023-prog/a-share-top3-data/contents/data/raw/{trade_date[:4]}?ref=main")
        if isinstance(payload, list):
            dates.extend(item.get("name", "") for item in payload if item.get("type") == "dir")
    return sorted(date for date in dates if start <= date <= trade_date)


def build_history_features(out: pd.DataFrame, trade_date: str, current_daily: pd.DataFrame, daily_basic: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for date in available_raw_dates(trade_date)[-24:]:
        daily_path = "data/latest/daily.csv" if date == trade_date else f"data/raw/{date[:4]}/{date}/daily.csv"
        frame = current_daily if date == trade_date else read_csv_source(daily_path)
        if frame.empty:
            continue
        keep = [c for c in ["ts_code", "trade_date", "close", "high", "low", "amount", "pct_chg"] if c in frame.columns]
        frame = frame[keep].copy()
        frame["trade_date"] = frame.get("trade_date", date)
        frames.append(frame)
    if not frames:
        return out

    hist = pd.concat(frames, ignore_index=True, sort=False)
    hist["ts_code"] = hist["ts_code"].astype(str).str.strip()
    hist["trade_date"] = hist["trade_date"].astype(str).str.replace("-", "", regex=False)
    for col in ["close", "high", "low", "amount", "pct_chg"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.dropna(subset=["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])

    current = hist[hist["trade_date"] == trade_date].copy()
    if current.empty:
        return out
    prev = hist[hist["trade_date"] < trade_date].copy()
    grouped = prev.groupby("ts_code")
    amount_5 = grouped["amount"].tail(5).groupby(prev.loc[grouped["amount"].tail(5).index, "ts_code"]).mean()
    amount_20 = grouped["amount"].tail(20).groupby(prev.loc[grouped["amount"].tail(20).index, "ts_code"]).mean()
    close_5 = grouped["close"].tail(5).groupby(prev.loc[grouped["close"].tail(5).index, "ts_code"]).first()
    close_20 = grouped["close"].tail(20).groupby(prev.loc[grouped["close"].tail(20).index, "ts_code"]).first()
    high_20 = grouped["high"].tail(20).groupby(prev.loc[grouped["high"].tail(20).index, "ts_code"]).max()
    close_high_20 = grouped["close"].tail(20).groupby(prev.loc[grouped["close"].tail(20).index, "ts_code"]).max()
    turnover_5 = pd.Series(dtype="float64")
    if not daily_basic.empty and {"ts_code", "turnover_rate"}.issubset(daily_basic.columns):
        turnover_5 = pd.to_numeric(daily_basic.set_index("ts_code")["turnover_rate"], errors="coerce")

    current = current.set_index("ts_code")
    current_amount = pd.to_numeric(current["amount"], errors="coerce")
    current_close = pd.to_numeric(current["close"], errors="coerce")
    current_high = pd.to_numeric(current["high"], errors="coerce")
    metrics = pd.DataFrame(index=current.index)
    metrics["amount_ratio_5d"] = current_amount / amount_5.reindex(current.index).replace(0, np.nan)
    metrics["amount_ratio_20d"] = current_amount / amount_20.reindex(current.index).replace(0, np.nan)
    metrics["ret_5d"] = (current_close / close_5.reindex(current.index).replace(0, np.nan) - 1) * 100
    metrics["ret_20d"] = (current_close / close_20.reindex(current.index).replace(0, np.nan) - 1) * 100
    metrics["stage_high_20d"] = high_20.reindex(current.index)
    metrics["high_20d_break"] = (current_high >= metrics["stage_high_20d"].fillna(current_high) * 0.999).astype(int)
    metrics["platform_break_20d"] = (current_close >= close_high_20.reindex(current.index).fillna(current_close) * 1.005).astype(int)
    metrics["turnover_rate_5d_avg"] = turnover_5.reindex(current.index)
    metrics = metrics.reset_index().rename(columns={"index": "ts_code"})
    return out.merge(metrics, on="ts_code", how="left", suffixes=("", "_hist"))


def build_from_latest_data() -> pd.DataFrame:
    global LAST_SOURCE_TRADE_DATE
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
    LAST_SOURCE_TRADE_DATE = trade_date
    if trade_date != target_trade_date() and os.environ.get("WP_ALLOW_STALE_DATA", "").strip() != "1":
        return pd.DataFrame(columns=SCHEMA)
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
    out["close_position"] = np.where(to_num(out, "high") > to_num(out, "low"), (close - to_num(out, "low")) / (to_num(out, "high") - to_num(out, "low")) * 100, 50)
    out["intraday_pullback_pct"] = np.where(close > 0, (to_num(out, "high") / close - 1) * 100, 0)
    out["open_to_close_pct"] = np.where(to_num(out, "open") > 0, (close / to_num(out, "open") - 1) * 100, 0)
    out["gap_open_pct"] = np.where(to_num(out, "pre_close") > 0, (to_num(out, "open") / to_num(out, "pre_close") - 1) * 100, 0)
    out["amplitude"] = np.where(to_num(out, "pre_close") > 0, (to_num(out, "high") - to_num(out, "low")) / to_num(out, "pre_close") * 100, 0)

    sector_gt6 = out.assign(_gt6=pct_chg > 6).groupby("sector_name")["_gt6"].sum()
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
        top = top.rename(columns={"net_rate": "dragon_tiger_net_rate", "reason": "dragon_tiger_reason"})
        keep = [c for c in ["ts_code", "dragon_tiger_flag", "dragon_tiger_net_rate", "dragon_tiger_reason"] if c in top.columns]
        out = out.merge(top[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    out = build_history_features(out, trade_date, daily, daily_basic)

    out["sector_rank"] = to_num(out, "sector_rank", 99)
    out["sector_limitup_count"] = to_num(out, "sector_limitup_count", 0)
    out["sector_gt6_count"] = to_num(out, "sector_gt6_count", 0)
    out["sector_amount_ratio"] = to_num(out, "sector_amount_ratio", 1)
    out["ret_5d"] = to_num(out, "ret_5d", 0).replace(0, np.nan).fillna(pct_chg)
    out["ret_20d"] = to_num(out, "ret_20d", 0).replace(0, np.nan).fillna(pct_chg)
    out["amount_ratio_5d"] = to_num(out, "amount_ratio_5d", 1).replace([np.inf, -np.inf], np.nan).fillna(to_num(out, "volume_ratio", 1))
    out["amount_ratio_20d"] = to_num(out, "amount_ratio_20d", 1).replace([np.inf, -np.inf], np.nan).fillna(out["amount_ratio_5d"])
    out["turnover_rate_5d_avg"] = to_num(out, "turnover_rate_5d_avg", 0)
    out["high_20d_break"] = to_num(out, "high_20d_break", 0)
    out["platform_break_20d"] = to_num(out, "platform_break_20d", 0)
    out["stage_high_20d"] = to_num(out, "stage_high_20d", 0)
    out["dragon_tiger_flag"] = to_num(out, "dragon_tiger_flag", 0)
    out["dragon_tiger_net_rate"] = to_num(out, "dragon_tiger_net_rate", 0)
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
        mask = (pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0) > 6) & (pd.to_numeric(df["pre_day_limitup"], errors="coerce").fillna(0) != 1) & (pd.to_numeric(df["today_limitup"], errors="coerce").fillna(0) != 1)
        df = df.loc[mask].copy()
    write_csv(df, WP_ROOT / "candidates" / current.strftime("%Y") / current.strftime("%Y%m%d") / "wp_candidates.csv")
    write_csv(df, LATEST / "wp_latest_candidates.csv")
    return df


def build_labels() -> pd.DataFrame:
    current = now_cn()
    columns = [
        "trade_date", "ts_code", "name", "p_limitup_t1", "wp_score",
        "next_trade_date", "next_day_high", "next_day_close",
        "next_day_limitup_price", "label_t1_limitup",
        "next_day_max_pct", "next_day_close_pct",
    ]
    target = target_trade_date()
    dates = available_raw_dates(target)
    labels = pd.DataFrame(columns=columns)
    if len(dates) >= 2:
        label_date = dates[-2]
        next_date = dates[-1]
        candidates = read_csv_source(f"data/wp/candidates/{label_date[:4]}/{label_date}/wp_candidates.csv")
        if not candidates.empty:
            daily_next = read_csv_source(f"data/raw/{next_date[:4]}/{next_date}/daily.csv")
            limit_next = read_csv_source(f"data/raw/{next_date[:4]}/{next_date}/stk_limit.csv")
            if not daily_next.empty and not limit_next.empty:
                next_frame = daily_next[["ts_code", "high", "close"]].merge(limit_next[["ts_code", "up_limit"]], on="ts_code", how="left")
                next_frame["ts_code"] = next_frame["ts_code"].astype(str).str.strip()
                labels = candidates.merge(next_frame, on="ts_code", how="left")
                price = pd.to_numeric(labels.get("price", labels.get("close", 0)), errors="coerce")
                labels["next_trade_date"] = next_date
                labels["next_day_high"] = pd.to_numeric(labels["high"], errors="coerce")
                labels["next_day_close"] = pd.to_numeric(labels["close_y"] if "close_y" in labels.columns else labels["close"], errors="coerce")
                labels["next_day_limitup_price"] = pd.to_numeric(labels["up_limit"], errors="coerce")
                labels["label_t1_limitup"] = ((labels["next_day_limitup_price"] > 0) & (labels["next_day_high"] >= labels["next_day_limitup_price"] * 0.999)).astype(int)
                labels["next_day_max_pct"] = np.where(price > 0, (labels["next_day_high"] / price - 1) * 100, np.nan)
                labels["next_day_close_pct"] = np.where(price > 0, (labels["next_day_close"] / price - 1) * 100, np.nan)
                labels["p_limitup_t1"] = labels.get("p_limitup_t1", np.nan)
                labels["wp_score"] = labels.get("wp_score", np.nan)
                labels["trade_date"] = label_date
                labels = labels[[col for col in columns if col in labels.columns]]
                for col in columns:
                    if col not in labels.columns:
                        labels[col] = np.nan
                labels = labels[columns]
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
    data_trade_date = ""
    if not df.empty and "trade_date" in df.columns:
        dates = df["trade_date"].dropna().astype(str).str.replace("-", "", regex=False)
        dates = dates[dates.str.len() == 8]
        if not dates.empty:
            data_trade_date = str(sorted(dates.unique())[-1])
    source_trade_date = data_trade_date or LAST_SOURCE_TRADE_DATE
    expected_trade_date = target_trade_date()
    status = "ok" if not df.empty else "empty_schema_ready"
    if source_trade_date and source_trade_date != expected_trade_date:
        status = "stale_data"
    payload = {
        "generated_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "source_trade_date": source_trade_date,
        "expected_trade_date": expected_trade_date,
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
