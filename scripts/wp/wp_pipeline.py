from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from io import StringIO
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
    "sector_gt6_count", "sector_amount_ratio", "sector_net_inflow",
    "sector_turnover", "sector_hot_score", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
    "ret_3d", "ret_10d", "amount_ratio_5d", "amount_ratio_20d", "turnover_rate_5d_avg",
    "close_position", "intraday_pullback_pct", "open_to_close_pct",
    "gap_open_pct", "amplitude", "high_20d_break", "platform_break_20d",
    "stage_high_20d", "ma5_position", "ma10_position", "ma20_position",
    "intraday_vwap_position", "late_pullback_pct", "late_price_change_pct",
    "late_volume_ratio", "tail_lift_flag", "dragon_tiger_flag", "dragon_tiger_net_rate",
    "dragon_tiger_reason", "limit_touch_count", "open_board_count",
    "limitup_quality_score", "intraday_risk_score", "announcement_flag",
    "hot_topic_flag", "auction_price", "auction_vol", "auction_amount",
    "auction_pct_chg", "auction_amount_ratio", "auction_strength_score",
    "realtime_source", "stock_age_days", "suspended_flag", "delist_flag",
    "data_quality_flag",
]


def read_csv_source(relative_path: str) -> pd.DataFrame:
    local_path = ROOT / relative_path
    if local_path.exists() and local_path.stat().st_size > 0:
        return pd.read_csv(local_path, encoding="utf-8-sig")
    url = f"{RAW_BASE_URL}/{relative_path}"
    try:
        request = Request(url, headers={"User-Agent": "WP-direct-processor"})
        with urlopen(request, timeout=30) as resp:
            if resp.status >= 400:
                return pd.DataFrame()
            content = resp.read().decode("utf-8-sig")
        return pd.read_csv(StringIO(content))
    except (
        URLError,
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()


def read_json_url(url: str):
    try:
        req = Request(url, headers={"Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError):
        return None


def now_cn() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=8)


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


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    text = text.where(text.str.fullmatch(r"\d{8}"), "")
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def merge_stock_basic_fields(out: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if stock_basic.empty or "ts_code" not in stock_basic.columns:
        return out

    fields = ["name", "industry", "market", "list_date"]
    keep = ["ts_code", *(field for field in fields if field in stock_basic.columns)]
    basic = stock_basic[keep].drop_duplicates("ts_code").copy()
    basic["ts_code"] = basic["ts_code"].astype(str).str.strip()
    # Realtime quotes also contain funds and ETFs; stock_basic defines the equity universe.
    out = out[out["ts_code"].isin(set(basic["ts_code"]))].copy()
    renamed = {field: f"_stock_basic_{field}" for field in fields if field in basic.columns}
    basic = basic.rename(columns=renamed)
    merged = out.merge(basic, on="ts_code", how="left")

    for field, source_field in renamed.items():
        source = merged[source_field]
        if field in merged.columns:
            current = merged[field]
            has_value = current.fillna("").astype(str).str.strip().ne("")
            merged[field] = current.where(has_value, source)
        else:
            merged[field] = source
    return merged.drop(columns=list(renamed.values()))


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


def latest_historical_daily_basic(trade_date: str) -> pd.DataFrame:
    for date in reversed(available_raw_dates(trade_date)):
        if date >= trade_date:
            continue
        frame = read_csv_source(f"data/raw/{date[:4]}/{date}/daily_basic.csv")
        if not frame.empty:
            frame["ts_code"] = frame["ts_code"].astype(str).str.strip()
            return frame
    return pd.DataFrame()


def previous_limitup_codes(trade_date: str) -> set[str]:
    dates = [date for date in available_raw_dates(trade_date) if date < trade_date]
    if not dates:
        return set()
    prev_date = dates[-1]
    prev = read_csv_source(f"data/raw/{prev_date[:4]}/{prev_date}/limit_list_d.csv")
    if prev.empty or "ts_code" not in prev.columns:
        return set()
    return set(prev["ts_code"].dropna().astype(str).str.strip())


def available_raw_dates(trade_date: str, lookback_days: int = 45) -> list[str]:
    start_dt = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=lookback_days)
    end_dt = datetime.strptime(trade_date, "%Y%m%d")
    start = start_dt.strftime("%Y%m%d")
    dates: list[str] = []
    for year in range(start_dt.year, end_dt.year + 1):
        year_dir = RAW_ROOT / str(year)
        if year_dir.exists():
            dates.extend(path.name for path in year_dir.iterdir() if path.is_dir())
        # A direct WP checkout contains only scripts; the fetch stage then
        # creates today's local directory. Always merge the remote archive so
        # that one local date cannot hide all prior trading days.
        payload = read_json_url(
            "https://api.github.com/repos/njedu2023-prog/"
            f"a-share-top3-data/contents/data/raw/{year}?ref=main"
        )
        if isinstance(payload, list):
            dates.extend(item.get("name", "") for item in payload if item.get("type") == "dir")
    if not dates:
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
    return sorted(set(date for date in dates if start <= date <= trade_date))


def build_history_features(out: pd.DataFrame, trade_date: str, current_daily: pd.DataFrame, daily_basic: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for date in available_raw_dates(trade_date)[-24:]:
        daily_path = "data/latest/daily.csv" if date == trade_date else f"data/raw/{date[:4]}/{date}/daily.csv"
        frame = current_daily if date == trade_date else read_csv_source(daily_path)
        if frame.empty:
            continue
        keep = [c for c in ["ts_code", "trade_date", "close", "high", "low", "amount", "vol", "volume", "pct_chg"] if c in frame.columns]
        frame = frame[keep].copy()
        frame["trade_date"] = frame.get("trade_date", date)
        if "vol" not in frame.columns and "volume" in frame.columns:
            frame["vol"] = frame["volume"]
        if date != trade_date and "amount" in frame.columns:
            frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") * 1000
        frames.append(frame)
    if not frames:
        return out

    hist = pd.concat(frames, ignore_index=True, sort=False)
    hist["ts_code"] = hist["ts_code"].astype(str).str.strip()
    hist["trade_date"] = hist["trade_date"].astype(str).str.replace("-", "", regex=False)
    for col in ["close", "high", "low", "amount", "vol", "pct_chg"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.dropna(subset=["ts_code", "trade_date"]).sort_values(["ts_code", "trade_date"])

    current = hist[hist["trade_date"] == trade_date].copy()
    if current.empty:
        return out
    prev = hist[hist["trade_date"] < trade_date].copy()
    rows = []
    prev_basic = pd.DataFrame()
    if not daily_basic.empty and {"ts_code", "turnover_rate"}.issubset(daily_basic.columns):
        prev_basic = daily_basic.copy()
        prev_basic["ts_code"] = prev_basic["ts_code"].astype(str).str.strip()
        prev_basic = prev_basic.drop_duplicates("ts_code").set_index("ts_code")
    current = current.set_index("ts_code")
    for ts_code, group in prev.groupby("ts_code"):
        if ts_code not in current.index:
            continue
        group = group.sort_values("trade_date")
        cur = current.loc[ts_code]
        close = float(cur.get("close", np.nan))
        high = float(cur.get("high", np.nan))
        amount = float(cur.get("amount", np.nan))
        cur_vol = float(cur.get("vol", np.nan))
        tail3 = group.tail(3)
        tail5 = group.tail(5)
        tail10 = group.tail(10)
        tail20 = group.tail(20)
        close_3 = tail3["close"].iloc[0] if len(tail3) else np.nan
        close_5 = tail5["close"].iloc[0] if len(tail5) else np.nan
        close_10 = tail10["close"].iloc[0] if len(tail10) else np.nan
        close_20 = tail20["close"].iloc[0] if len(tail20) else np.nan
        ma5 = tail5["close"].mean()
        ma10 = tail10["close"].mean()
        ma20 = tail20["close"].mean()
        high_20 = tail20["high"].max()
        close_high_20 = tail20["close"].max()
        avg_vol_5 = tail5["vol"].mean() if "vol" in tail5.columns and len(tail5) else np.nan
        prev_vol = float(group["vol"].iloc[-1]) if "vol" in group.columns and len(group) else np.nan
        prev_turnover = np.nan
        if not prev_basic.empty and ts_code in prev_basic.index:
            prev_turnover = pd.to_numeric(pd.Series([prev_basic.loc[ts_code].get("turnover_rate", np.nan)]), errors="coerce").iloc[0]
        turnover_rate = (
            prev_turnover * cur_vol / prev_vol
            if pd.notna(prev_turnover) and prev_turnover > 0 and pd.notna(cur_vol) and cur_vol > 0 and pd.notna(prev_vol) and prev_vol > 0
            else np.nan
        )
        rows.append({
            "ts_code": ts_code,
            "amount_ratio_5d": amount / tail5["amount"].mean() if len(tail5) and tail5["amount"].mean() > 0 else np.nan,
            "amount_ratio_20d": amount / tail20["amount"].mean() if len(tail20) and tail20["amount"].mean() > 0 else np.nan,
            "volume_ratio": cur_vol / avg_vol_5 if pd.notna(avg_vol_5) and avg_vol_5 > 0 and pd.notna(cur_vol) else np.nan,
            "turnover_rate": turnover_rate,
            "turnover_rate_5d_avg": prev_turnover if pd.notna(prev_turnover) else np.nan,
            "ret_3d": (close / close_3 - 1) * 100 if close_3 and close_3 > 0 else np.nan,
            "ret_5d": (close / close_5 - 1) * 100 if close_5 and close_5 > 0 else np.nan,
            "ret_10d": (close / close_10 - 1) * 100 if close_10 and close_10 > 0 else np.nan,
            "ret_20d": (close / close_20 - 1) * 100 if close_20 and close_20 > 0 else np.nan,
            "ma5_position": (close / ma5 - 1) * 100 if ma5 and ma5 > 0 else np.nan,
            "ma10_position": (close / ma10 - 1) * 100 if ma10 and ma10 > 0 else np.nan,
            "ma20_position": (close / ma20 - 1) * 100 if ma20 and ma20 > 0 else np.nan,
            "stage_high_20d": high_20,
            "high_20d_break": int(high >= high_20 * 0.999) if high_20 and high_20 > 0 else 0,
            "platform_break_20d": int(close >= close_high_20 * 1.005) if close_high_20 and close_high_20 > 0 else 0,
        })
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return out
    return out.merge(metrics, on="ts_code", how="left", suffixes=("", "_hist"))


def build_from_latest_data() -> pd.DataFrame:
    global LAST_SOURCE_TRADE_DATE
    daily = read_csv_source("data/latest/daily.csv")
    realtime_snapshot = read_csv_source("data/latest/realtime_snapshot.csv")
    if daily.empty and realtime_snapshot.empty:
        return pd.DataFrame(columns=SCHEMA)
    daily_basic = read_csv_source("data/latest/daily_basic.csv")
    if daily_basic.empty:
        daily_basic = latest_historical_daily_basic(trade_date=target_trade_date())
    stock_basic = read_csv_source("data/latest/stock_basic.csv")
    stk_limit = read_csv_source("data/latest/stk_limit.csv")
    limit_list = read_csv_source("data/latest/limit_list_d.csv")
    hot_boards = read_csv_source("data/latest/hot_boards.csv")
    top_list = read_csv_source("data/latest/top_list.csv")
    intraday = read_csv_source("data/latest/intraday_features.csv")
    auction_features = read_csv_source("data/latest/auction_features.csv")

    trade_date = latest_trade_date(realtime_snapshot, daily, daily_basic, stk_limit, limit_list)
    use_realtime = (
        not realtime_snapshot.empty
        and "trade_date" in realtime_snapshot.columns
        and str(trade_date) == target_trade_date()
    )
    out = realtime_snapshot.copy() if use_realtime else daily.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    LAST_SOURCE_TRADE_DATE = trade_date
    if trade_date != target_trade_date() and os.environ.get("WP_ALLOW_STALE_DATA", "").strip() != "1":
        return pd.DataFrame(columns=SCHEMA)
    out["trade_date"] = out.get("trade_date", trade_date).fillna(trade_date).astype(str)

    if not daily_basic.empty:
        keep = [c for c in ["ts_code", "turnover_rate", "volume_ratio", "total_mv", "float_mv"] if c in daily_basic.columns]
        out = out.merge(daily_basic[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    out = merge_stock_basic_fields(out, stock_basic)
    if not stk_limit.empty:
        keep = [c for c in ["ts_code", "up_limit", "down_limit"] if c in stk_limit.columns]
        out = out.merge(stk_limit[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    if not intraday.empty:
        keep = [
            c for c in [
                "ts_code", "limit_touch_count", "open_board_count", "limitup_quality_score",
                "intraday_risk_score", "late_volume_ratio", "late_price_weakness",
                "late_price_change_pct", "max_drawdown_after_limit", "intraday_vwap_position",
            ]
            if c in intraday.columns
        ]
        out = out.merge(intraday[keep].drop_duplicates("ts_code"), on="ts_code", how="left")
    if not auction_features.empty and "ts_code" in auction_features.columns:
        keep = [
            c for c in [
                "ts_code", "auction_price", "auction_vol", "auction_amount",
                "auction_pct_chg", "auction_amount_ratio", "auction_strength_score",
            ]
            if c in auction_features.columns
        ]
        out = out.merge(auction_features[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    close = to_num(out, "close")
    pct_chg = to_num(out, "pct_chg")
    pre_close_existing = to_num(out, "pre_close", np.nan)
    out["pre_close"] = pre_close_existing.where(
        pre_close_existing.notna() & (pre_close_existing > 0),
        np.where((1 + pct_chg / 100) > 0, close / (1 + pct_chg / 100), np.nan),
    )
    out["price"] = close
    out["volume"] = to_num(out, "vol")
    amount_raw = to_num(out, "amount")
    is_realtime_amount = out.get("realtime_source", pd.Series("", index=out.index)).fillna("").astype(str).ne("")
    out["amount"] = np.where(is_realtime_amount, amount_raw, amount_raw * 1000)
    out["sector_name"] = out.get("industry", pd.Series("未分类", index=out.index)).fillna("未分类").astype(str)
    if "list_date" in out.columns:
        list_date = parse_yyyymmdd(out["list_date"])
        trade_dt = pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
        out["stock_age_days"] = (trade_dt - list_date).dt.days
    else:
        out["stock_age_days"] = np.nan
    out["delist_flag"] = out.get("name", pd.Series("", index=out.index)).fillna("").astype(str).str.contains("退|退市", regex=True).astype(int)
    out["suspended_flag"] = np.where((close <= 0) | (to_num(out, "amount") <= 0) | (to_num(out, "vol") <= 0), 1, 0)

    current_limit_codes = set()
    if not limit_list.empty and "ts_code" in limit_list.columns:
        limit_dates = []
        if "trade_date" in limit_list.columns:
            limit_dates = (
                limit_list["trade_date"].dropna().astype(str).str.replace("-", "", regex=False).tolist()
            )
        if not limit_dates or str(sorted(limit_dates)[-1]) == str(trade_date):
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
    out["late_pullback_pct"] = to_num(out, "max_drawdown_after_limit", np.nan).fillna(out["intraday_pullback_pct"])
    out["late_price_change_pct"] = to_num(out, "late_price_change_pct", np.nan)
    out["late_price_change_pct"] = out["late_price_change_pct"].where(out["late_price_change_pct"].notna(), -to_num(out, "late_price_weakness", 0))
    out["late_volume_ratio"] = to_num(out, "late_volume_ratio", 1)
    typical_price = (to_num(out, "high") + to_num(out, "low") + close) / 3
    out["intraday_vwap_position"] = to_num(out, "intraday_vwap_position", np.nan)
    out["intraday_vwap_position"] = out["intraday_vwap_position"].where(out["intraday_vwap_position"].notna(), np.where(typical_price > 0, (close / typical_price - 1) * 100, 0))
    out["tail_lift_flag"] = np.where((out["late_volume_ratio"] >= 1.8) & (out["close_position"] >= 82) & (out["open_to_close_pct"] >= 3), 1, 0)
    out["announcement_flag"] = 0

    sector_gt6 = out.assign(_gt6=pct_chg > 8).groupby("sector_name")["_gt6"].sum()
    sector_amount = out.groupby("sector_name")["amount"].sum()
    sector_turnover = out.groupby("sector_name")["turnover_rate"].mean() if "turnover_rate" in out.columns else pd.Series(dtype="float64")
    amount_median = float(sector_amount.median()) if len(sector_amount) else 0.0
    sector_metrics = pd.DataFrame({
        "sector_name": sector_gt6.index,
        "sector_gt6_count": sector_gt6.values,
        "sector_amount_ratio": [(sector_amount.get(name, 0.0) / amount_median) if amount_median > 0 else 1.0 for name in sector_gt6.index],
        "sector_turnover": [sector_turnover.get(name, np.nan) for name in sector_gt6.index],
    })
    if not hot_boards.empty and "industry" in hot_boards.columns:
        boards = hot_boards.rename(columns={"industry": "sector_name", "rank": "sector_rank", "limit_up_count": "sector_limitup_count"})
        keep = [c for c in ["sector_name", "sector_rank", "sector_limitup_count"] if c in boards.columns]
        sector_metrics = sector_metrics.merge(boards[keep].drop_duplicates("sector_name"), on="sector_name", how="left")
    out = out.merge(sector_metrics, on="sector_name", how="left")
    out["hot_topic_flag"] = np.where(to_num(out, "sector_rank", 99) <= 10, 1, 0)
    out["sector_net_inflow"] = 0
    out["sector_hot_score"] = np.maximum(0, 100 - to_num(out, "sector_rank", 99) * 4) + to_num(out, "sector_gt6_count", 0) * 5 + to_num(out, "sector_limitup_count", 0) * 8

    if not top_list.empty and "ts_code" in top_list.columns:
        top = top_list.copy()
        top["dragon_tiger_flag"] = 1
        top = top.rename(columns={"net_rate": "dragon_tiger_net_rate", "reason": "dragon_tiger_reason"})
        keep = [c for c in ["ts_code", "dragon_tiger_flag", "dragon_tiger_net_rate", "dragon_tiger_reason"] if c in top.columns]
        out = out.merge(top[keep].drop_duplicates("ts_code"), on="ts_code", how="left")

    current_for_history = out.copy()
    current_for_history["vol"] = to_num(current_for_history, "volume", np.nan)
    out = build_history_features(out, trade_date, current_for_history, daily_basic)
    for col in ["volume_ratio", "turnover_rate", "turnover_rate_5d_avg"]:
        hist_col = f"{col}_hist"
        if hist_col in out.columns:
            base = pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
            out[col] = base.where(base.notna(), pd.to_numeric(out[hist_col], errors="coerce"))

    out["sector_rank"] = to_num(out, "sector_rank", 99)
    out["sector_limitup_count"] = to_num(out, "sector_limitup_count", 0)
    out["sector_gt6_count"] = to_num(out, "sector_gt6_count", 0)
    out["sector_amount_ratio"] = to_num(out, "sector_amount_ratio", 1)
    out["ret_20d"] = to_num(out, "ret_20d", 0).replace(0, np.nan).fillna(pct_chg)
    out["ret_5d"] = to_num(out, "ret_5d", 0).replace(0, np.nan).fillna(pct_chg)
    out["ret_3d"] = to_num(out, "ret_3d", 0).replace(0, np.nan).fillna(out["ret_5d"])
    out["ret_10d"] = to_num(out, "ret_10d", 0).replace(0, np.nan).fillna(out["ret_20d"])
    out["amount_ratio_5d"] = to_num(out, "amount_ratio_5d", 1).replace([np.inf, -np.inf], np.nan).fillna(to_num(out, "volume_ratio", 1))
    out["amount_ratio_20d"] = to_num(out, "amount_ratio_20d", 1).replace([np.inf, -np.inf], np.nan).fillna(out["amount_ratio_5d"])
    out["turnover_rate_5d_avg"] = to_num(out, "turnover_rate_5d_avg", 0)
    out["high_20d_break"] = to_num(out, "high_20d_break", 0)
    out["platform_break_20d"] = to_num(out, "platform_break_20d", 0)
    out["stage_high_20d"] = to_num(out, "stage_high_20d", 0)
    out["ma5_position"] = to_num(out, "ma5_position", 0)
    out["ma10_position"] = to_num(out, "ma10_position", 0)
    out["ma20_position"] = to_num(out, "ma20_position", 0)
    out["dragon_tiger_flag"] = to_num(out, "dragon_tiger_flag", 0)
    out["dragon_tiger_net_rate"] = to_num(out, "dragon_tiger_net_rate", 0)
    out["sector_net_inflow"] = to_num(out, "sector_net_inflow", 0)
    out["sector_turnover"] = to_num(out, "sector_turnover", 0)
    out["sector_hot_score"] = to_num(out, "sector_hot_score", 0).clip(0, 100)
    out["announcement_flag"] = to_num(out, "announcement_flag", 0)
    out["hot_topic_flag"] = to_num(out, "hot_topic_flag", 0)
    out["auction_price"] = to_num(out, "auction_price", 0)
    out["auction_vol"] = to_num(out, "auction_vol", 0)
    out["auction_amount"] = to_num(out, "auction_amount", 0)
    out["auction_pct_chg"] = to_num(out, "auction_pct_chg", 0)
    out["auction_amount_ratio"] = to_num(out, "auction_amount_ratio", 0)
    out["auction_strength_score"] = to_num(out, "auction_strength_score", 0).clip(0, 100)
    required_core = ["ts_code", "close", "pre_close", "pct_chg", "amount", "today_limit_up_price"]
    out["data_quality_flag"] = np.where(out[required_core].isna().any(axis=1), 1, 0)
    out.loc[(to_num(out, "close") <= 0) | (to_num(out, "pre_close") <= 0) | (to_num(out, "amount") <= 0) | (to_num(out, "today_limit_up_price") <= 0), "data_quality_flag"] = 1
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
        mask = (pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0) > 8) & (pd.to_numeric(df["pre_day_limitup"], errors="coerce").fillna(0) != 1) & (pd.to_numeric(df["today_limitup"], errors="coerce").fillna(0) != 1)
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


def historical_feature_quality(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "history_feature_rows": 0,
            "history_feature_coverage_pct": 100.0,
            "history_features_ready": True,
        }
    pct = pd.to_numeric(frame.get("pct_chg"), errors="coerce")
    ret_5d = pd.to_numeric(frame.get("ret_5d"), errors="coerce")
    ratio_5d = pd.to_numeric(frame.get("amount_ratio_5d"), errors="coerce")
    ma5 = pd.to_numeric(frame.get("ma5_position"), errors="coerce")
    ma20 = pd.to_numeric(frame.get("ma20_position"), errors="coerce")
    meaningful = (
        ratio_5d.notna()
        & ratio_5d.gt(0)
        & (
            ratio_5d.sub(1).abs().gt(1e-6)
            | ret_5d.sub(pct).abs().gt(1e-6)
            | ma5.abs().gt(1e-6)
            | ma20.abs().gt(1e-6)
        )
    )
    coverage = float(meaningful.mean() * 100)
    return {
        "history_feature_rows": int(meaningful.sum()),
        "history_feature_coverage_pct": round(coverage, 2),
        "history_features_ready": bool(len(frame) < 3 or coverage >= 70),
    }


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
    history_quality = historical_feature_quality(df)
    if status == "ok" and not history_quality["history_features_ready"]:
        status = "degraded_history"
    payload = {
        "generated_at": now_cn().strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_slot": os.environ.get("WP_TARGET_SLOT", "").strip(),
        "trigger_event": os.environ.get("GITHUB_EVENT_NAME", "").strip(),
        "status": status,
        "source_trade_date": source_trade_date,
        "expected_trade_date": expected_trade_date,
        "row_count": int(len(df)),
        "candidate_count": int(len(df)),
        **history_quality,
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
