import os
import time
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List, Tuple

import pandas as pd
import tushare as ts
from dateutil import tz

BJ_TZ = tz.gettz("Asia/Shanghai")


# =========================
# 基础工具
# =========================
def bj_now() -> datetime:
    return datetime.now(BJ_TZ)


def bj_today_yyyymmdd() -> str:
    return bj_now().strftime("%Y%m%d")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_columns(cols: Optional[List[str]]) -> Optional[List[str]]:
    if not cols:
        return None
    out = []
    seen = set()
    for c in cols:
        c = str(c).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out or None


def save_df(df: pd.DataFrame, out_csv: Path, *, columns: Optional[List[str]] = None) -> None:
    """
    关键保证：
    - df 有数据：正常写出
    - df 无数据：也要写出“带表头”的标准 CSV（至少包含 columns 指定的列）
    """
    ensure_dir(out_csv.parent)

    cols = _normalize_columns(columns)

    if df is None:
        df = pd.DataFrame()

    # 若 df 为空且没有列，但我们知道期望列：写一个只有表头的空表
    if df.empty and (df.columns is None or len(df.columns) == 0) and cols:
        df = pd.DataFrame(columns=cols)

    # 若 df 非空但缺少部分期望列：补齐（不影响已有数据）
    if cols:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        # 让期望列排在前面
        front = [c for c in cols if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_csv(path: Path) -> pd.DataFrame:
    """
    兼容：
    - 正常 CSV
    - 只有 BOM / 空文件 / 读取失败：返回空 DataFrame
    """
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception:
        pass
    return pd.DataFrame()


def fn_display_name(fn: Callable) -> str:
    """
    兼容：普通函数 / tushare 的接口方法 / functools.partial
    """
    name = getattr(fn, "__name__", None)
    if name:
        return name
    # partial
    func = getattr(fn, "func", None)
    if func is not None:
        return getattr(func, "__name__", func.__class__.__name__)
    return fn.__class__.__name__


def get_pro():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("缺少环境变量 TUSHARE_TOKEN（请在 GitHub Secrets 里配置）")
    ts.set_token(token)
    return ts.pro_api()


def _norm_ts_code(s: Any) -> str:
    x = "" if s is None else str(s).strip()
    return x


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _clip_score(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(100.0, v))


def _trade_date_dash(trade_date: str) -> str:
    return f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"


def _intraday_feature_columns() -> List[str]:
    return [
        "ts_code",
        "trade_date",
        "minute_freq",
        "minute_rows",
        "has_minute_data",
        "first_limit_time",
        "last_limit_time",
        "limit_touch_count",
        "open_board_count",
        "max_drawdown_after_limit",
        "reseal_count",
        "reseal_minutes_avg",
        "reseal_speed_score",
        "reseal_acceptance_score",
        "intraday_vwap",
        "intraday_vwap_position",
        "late_volume_ratio",
        "late_price_change_pct",
        "late_price_weakness",
        "late_limit_hold_minutes",
        "late_withdraw_score",
        "limitup_path_score",
        "limitup_quality_score",
        "intraday_risk_score",
        "intraday_tag",
    ]


def _auction_columns() -> List[str]:
    return ["ts_code", "trade_date", "vol", "price", "amount"]


def _auction_feature_columns() -> List[str]:
    return [
        "ts_code",
        "trade_date",
        "auction_price",
        "auction_vol",
        "auction_amount",
        "auction_pct_chg",
        "auction_amount_ratio",
        "auction_strength_score",
    ]


def _limit_stage_columns() -> List[str]:
    return [
        "trade_date",
        "ts_code",
        "name",
        "limit_times",
        "advance_stage",
        "晋阶",
        "stage_quality_weight",
        "stage_risk_weight",
        "stage_prior",
        "stage_source",
        "up_stat",
    ]


def _minute_columns() -> List[str]:
    return ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]


def _realtime_snapshot_columns() -> List[str]:
    return [
        "ts_code",
        "trade_date",
        "update_time",
        "price",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
        "realtime_source",
    ]


def _realtime_quote_columns() -> List[str]:
    return [
        "ts_code",
        "trade_date",
        "update_time",
        "name",
        "price",
        "open",
        "high",
        "low",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
    ]


def _clear_csv_dir(path: Path) -> None:
    ensure_dir(path)
    for p in path.glob("*.csv"):
        try:
            p.unlink()
        except Exception as e:
            print(f"[WARN] failed to remove stale csv {p}: {repr(e)}")


# =========================
# 交易日处理
# =========================
def resolve_trade_date(pro, requested_trade_date: str) -> str:
    """
    - 如果指定了 TRADE_DATE：优先用它；但若它不是交易日，则回退最近交易日
    - 如果未指定：默认用北京时间今天；若今天非交易日则回退最近交易日
    """
    target = requested_trade_date.strip() if requested_trade_date else bj_today_yyyymmdd()

    end_date = target
    start_date = (datetime.strptime(target, "%Y%m%d") - pd.Timedelta(days=30)).strftime("%Y%m%d")

    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
    if cal is None or cal.empty:
        return target

    cal = cal.sort_values("cal_date")
    row = cal[cal["cal_date"] == target]
    if not row.empty and int(row.iloc[0]["is_open"]) == 1:
        return target

    opened = cal[cal["is_open"] == 1]
    opened = opened[opened["cal_date"] <= target]
    if opened.empty:
        return target
    return str(opened.iloc[-1]["cal_date"])


# =========================
# 重试封装
# =========================
@dataclass
class RetryConfig:
    max_retry: int = 10
    base_sleep_sec: float = 2.0
    max_sleep_sec: float = 20.0
    jitter_sec: float = 0.6


def call_with_retry(
    fn: Callable,
    *,
    retry: RetryConfig,
    allow_empty: bool,
    empty_ok_after_retry: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    allow_empty:
      - True  : 接口返回空 df 也算成功（直接返回空 df）
      - False : 空 df 视为“失败”，走重试；重试后仍空，看 empty_ok_after_retry 决定是否抛错
    empty_ok_after_retry:
      - True  : 重试耗尽仍为空 -> 返回空 df（不抛错，避免 workflow 失败）
      - False : 重试耗尽仍为空 -> 抛错（强制失败）
    """
    last_err: Optional[Exception] = None
    name = fn_display_name(fn)

    for i in range(1, retry.max_retry + 1):
        try:
            df = fn(**kwargs)

            if df is None:
                df = pd.DataFrame()

            if df.empty:
                if allow_empty:
                    print(f"[OK-EMPTY] {name} kwargs={kwargs} -> empty dataframe (allowed)")
                    return df
                raise RuntimeError(f"empty dataframe (not allowed): {name} kwargs={kwargs}")

            print(f"[OK] {name} kwargs={kwargs} -> rows={len(df)}")
            return df

        except Exception as e:
            last_err = e
            sleep = min(retry.max_sleep_sec, retry.base_sleep_sec * (2 ** (i - 1)))
            sleep = sleep + random.random() * retry.jitter_sec
            print(f"[RETRY {i}/{retry.max_retry}] {name} kwargs={kwargs} err={repr(e)} sleep={sleep:.1f}s")
            time.sleep(sleep)

    msg = f"Failed after {retry.max_retry} retries: {name} kwargs={kwargs} last_err={repr(last_err)}"
    if allow_empty or empty_ok_after_retry:
        print(f"[GIVEUP-BUT-CONTINUE] {msg}")
        return pd.DataFrame()
    raise RuntimeError(msg)


# =========================
# 抓取任务定义
# =========================
@dataclass
class FetchJob:
    key: str
    fn: Callable
    kwargs: Dict[str, Any]
    columns: List[str]
    allow_empty: bool = True
    required: bool = False
    note: str = ""


def _fields_to_columns(fields: Optional[str]) -> List[str]:
    if not fields:
        return []
    return [c.strip() for c in fields.split(",") if c.strip()]


def build_jobs(pro, trade_date: str) -> List[FetchJob]:
    """
    日频数据仓库（打板Top10系统）所需的最小核心数据：
    1) limit_list_d      涨停池（系统级强制：只保留“收盘真实涨停”）
    2) limit_break_d     炸板/开板（日）
    3) daily             日线OHLCV+amount
    4) stk_limit         涨跌停价
    5) daily_basic       换手/市值
    可选：
    6) stock_basic
    7) namechange
    8) top_list
    9) moneyflow_hsgt
    """
    jobs: List[FetchJob] = []

    schema_min_code_date = ["ts_code", "trade_date"]

    # 1) limit_list_d（优先请求 Tushare 原生连板字段；字段不兼容时自动回退基础字段）
    limit_list_base_fields = (
        "trade_date,ts_code,name,limit_type,close,up_limit,down_limit,"
        "open_times,fd_amount,first_time,last_time"
    )
    limit_list_fields = (
        limit_list_base_fields
        + ",limit_times,up_stat,industry,turnover_ratio,amount,float_mv,total_mv"
    )

    def _limit_list_d_with_field_fallback(**kwargs):
        try:
            return pro.limit_list_d(**kwargs)
        except Exception as e:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["fields"] = limit_list_base_fields
            print(f"[LIMIT_LIST_D-FIELD-FALLBACK] extended fields failed: {repr(e)}")
            return pro.limit_list_d(**fallback_kwargs)

    _limit_list_d_with_field_fallback.__name__ = "limit_list_d_with_field_fallback"
    jobs.append(
        FetchJob(
            key="limit_list_d",
            fn=_limit_list_d_with_field_fallback,
            kwargs={"trade_date": trade_date, "fields": limit_list_fields},
            columns=_fields_to_columns(limit_list_fields) or schema_min_code_date,
            allow_empty=True,
            required=False,
            note="涨停池（日）（优先拉取 Tushare limit_times/up_stat；后处理中仍强制保留收盘真实涨停池口径）",
        )
    )

    # 2) limit_break_d（可选）
    if hasattr(pro, "limit_break_d"):
        limit_break_fields = "trade_date,ts_code,name,open_times,first_time,last_time,fd_amount"
        jobs.append(
            FetchJob(
                key="limit_break_d",
                fn=pro.limit_break_d,
                kwargs={"trade_date": trade_date, "fields": limit_break_fields},
                columns=_fields_to_columns(limit_break_fields) or (schema_min_code_date + ["open_times"]),
                allow_empty=True,
                required=False,
                note="炸板/开板（日）（显式拉取 open_times）",
            )
        )

    # 3) 日线行情
    daily_fields = "ts_code,trade_date,open,high,low,close,pre_close,vol,amount,pct_chg"
    jobs.append(
        FetchJob(
            key="daily",
            fn=pro.daily,
            kwargs={"trade_date": trade_date, "fields": daily_fields},
            columns=_fields_to_columns(daily_fields),
            allow_empty=True,
            required=False,
            note="日线行情（OHLCV+amount）",
        )
    )

    # 4) 涨跌停价
    stk_limit_fields = "ts_code,trade_date,up_limit,down_limit"
    jobs.append(
        FetchJob(
            key="stk_limit",
            fn=pro.stk_limit,
            kwargs={"trade_date": trade_date, "fields": stk_limit_fields},
            columns=_fields_to_columns(stk_limit_fields),
            allow_empty=True,
            required=False,
            note="涨跌停价（限制价格）",
        )
    )

    # 5) 每日指标
    daily_basic_fields = "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,total_mv,float_mv"
    jobs.append(
        FetchJob(
            key="daily_basic",
            fn=pro.daily_basic,
            kwargs={"trade_date": trade_date, "fields": daily_basic_fields},
            columns=_fields_to_columns(daily_basic_fields),
            allow_empty=True,
            required=False,
            note="每日指标（换手/市值/量比）",
        )
    )

    # 6) 股票基础信息
    stock_basic_fields = "ts_code,symbol,name,area,industry,market,list_date"
    jobs.append(
        FetchJob(
            key="stock_basic",
            fn=pro.stock_basic,
            kwargs={"exchange": "", "list_status": "L", "fields": stock_basic_fields},
            columns=_fields_to_columns(stock_basic_fields),
            allow_empty=True,
            required=False,
            note="股票基础信息（name/industry/list_date等）",
        )
    )

    # 7) 名称变更
    namechange_fields = "ts_code,name,start_date,end_date,change_reason"
    jobs.append(
        FetchJob(
            key="namechange",
            fn=pro.namechange,
            kwargs={
                "start_date": (datetime.strptime(trade_date, "%Y%m%d") - pd.Timedelta(days=30)).strftime("%Y%m%d"),
                "end_date": trade_date,
                "fields": namechange_fields,
            },
            columns=_fields_to_columns(namechange_fields),
            allow_empty=True,
            required=False,
            note="名称变更（用于ST/更名等过滤）",
        )
    )

    # 8) 龙虎榜（可选）
    if hasattr(pro, "top_list"):
        jobs.append(
            FetchJob(
                key="top_list",
                fn=pro.top_list,
                kwargs={"trade_date": trade_date},
                columns=schema_min_code_date,
                allow_empty=True,
                required=False,
                note="龙虎榜",
            )
        )

    # 9) 沪深港通资金流向（可选）
    if hasattr(pro, "moneyflow_hsgt"):
        jobs.append(
            FetchJob(
                key="moneyflow_hsgt",
                fn=pro.moneyflow_hsgt,
                kwargs={"trade_date": trade_date},
                columns=schema_min_code_date,
                allow_empty=True,
                required=False,
                note="沪深港通资金流向",
            )
        )

    return jobs


# =========================
# 派生：热门板块/核心板块标签（日频、低算力）
# =========================
def derive_hot_board_tags(
    trade_date: str,
    base_raw: Path,
    base_latest: Path,
) -> Dict[str, Any]:
    topn = int(os.getenv("HOT_BOARD_TOPN", "10"))

    limit_path = base_latest / "limit_list_d.csv"
    basic_path = base_latest / "stock_basic.csv"
    namechg_path = base_latest / "namechange.csv"

    limit_df = load_csv(limit_path)
    basic_df = load_csv(basic_path)
    namechg_df = load_csv(namechg_path)

    if limit_df.empty or basic_df.empty:
        empty_hot = pd.DataFrame(columns=["trade_date", "industry", "limit_up_count", "rank"])
        empty_tags = pd.DataFrame(
            columns=[
                "trade_date",
                "ts_code",
                "name",
                "industry",
                "is_hot_board",
                "board_rank",
                "board_limit_up_count",
                "is_st_like",
            ]
        )
        save_df(empty_hot, base_raw / "hot_boards.csv", columns=list(empty_hot.columns))
        save_df(empty_tags, base_raw / "limit_up_tags.csv", columns=list(empty_tags.columns))
        save_df(empty_hot, base_latest / "hot_boards.csv", columns=list(empty_hot.columns))
        save_df(empty_tags, base_latest / "limit_up_tags.csv", columns=list(empty_tags.columns))
        return {"hot_board_topn": topn, "hot_boards": 0, "tagged": 0}

    limit_df = limit_df.copy()
    if "ts_code" not in limit_df.columns:
        return {"hot_board_topn": topn, "hot_boards": 0, "tagged": 0, "warn": "limit_list_d missing ts_code"}

    basic_df = basic_df.copy()
    keep_cols = [c for c in ["ts_code", "name", "industry"] if c in basic_df.columns]
    basic_df = basic_df[keep_cols].drop_duplicates(subset=["ts_code"])

    merged = limit_df.merge(basic_df, on="ts_code", how="left", suffixes=("", "_basic"))

    if "industry" not in merged.columns:
        merged["industry"] = ""
    if "name" not in merged.columns:
        merged["name"] = ""

    merged["industry"] = merged["industry"].fillna("").astype(str)
    merged["name"] = merged["name"].fillna("").astype(str)
    merged["ts_code"] = merged["ts_code"].fillna("").astype(str)

    # ST识别（轻量）
    st_like = set()
    try:
        if not namechg_df.empty and "ts_code" in namechg_df.columns:
            tmp = namechg_df.copy()
            tmp["ts_code"] = tmp["ts_code"].fillna("").astype(str)

            reason_col = "change_reason" if "change_reason" in tmp.columns else None
            if reason_col:
                tmp[reason_col] = tmp[reason_col].fillna("").astype(str)
                hit = tmp[tmp[reason_col].str.contains(r"ST|\*ST|退市|整理", regex=True, na=False)]
                st_like.update(hit["ts_code"].tolist())
    except Exception:
        pass

    ind_stat = (
        merged[merged["industry"] != ""]
        .groupby("industry", as_index=False)["ts_code"]
        .nunique()
        .rename(columns={"ts_code": "limit_up_count"})
        .sort_values(["limit_up_count", "industry"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ind_stat["rank"] = ind_stat.index + 1
    ind_stat = ind_stat.head(topn).copy()
    ind_stat.insert(0, "trade_date", trade_date)

    hot_industries = set(ind_stat["industry"].astype(str).tolist())
    rank_map = {row["industry"]: int(row["rank"]) for _, row in ind_stat.iterrows()}
    cnt_map = {row["industry"]: int(row["limit_up_count"]) for _, row in ind_stat.iterrows()}

    tags = merged[["ts_code", "name", "industry"]].drop_duplicates(subset=["ts_code"]).copy()
    tags.insert(0, "trade_date", trade_date)

    tags["is_hot_board"] = tags["industry"].astype(str).isin(hot_industries).astype(int)
    tags["board_rank"] = tags["industry"].astype(str).map(rank_map).fillna("")
    tags["board_limit_up_count"] = tags["industry"].astype(str).map(cnt_map).fillna("")

    name_has_st = tags["name"].fillna("").astype(str).str.contains(r"ST|\*ST", regex=True, na=False)
    code_in_st_like = tags["ts_code"].fillna("").astype(str).isin(st_like)
    tags["is_st_like"] = (name_has_st | code_in_st_like).astype(int)

    save_df(ind_stat, base_raw / "hot_boards.csv", columns=list(ind_stat.columns))
    save_df(tags, base_raw / "limit_up_tags.csv", columns=list(tags.columns))
    save_df(ind_stat, base_latest / "hot_boards.csv", columns=list(ind_stat.columns))
    save_df(tags, base_latest / "limit_up_tags.csv", columns=list(tags.columns))

    return {"hot_board_topn": topn, "hot_boards": int(len(ind_stat)), "tagged": int(len(tags))}


# =========================
# 主程序：抓取并落地快照
# =========================
def _postprocess_limit_tables(df: pd.DataFrame, key: str, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    补齐我们系统需要的“稳定字段名”：
    - seal_amount：封单额（映射自 tushare 的 fd_amount）
    - open_times ：开板/炸板次数（若缺列，补空列）
    """
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    def _ensure_col(col: str):
        if col not in df.columns:
            df[col] = pd.NA

    if key in ("limit_list_d", "limit_break_d"):
        _ensure_col("open_times")

        # fd_amount -> seal_amount（兼容字段缺失）
        if "seal_amount" not in df.columns:
            if "fd_amount" in df.columns:
                df["seal_amount"] = df["fd_amount"]
            else:
                df["seal_amount"] = pd.NA

        # 记录一下映射策略，便于排查
        meta.setdefault("derived", {})
        meta["derived"].setdefault("limit_fields", {})
        meta["derived"]["limit_fields"][key] = {
            "has_open_times": int("open_times" in df.columns),
            "has_fd_amount": int("fd_amount" in df.columns),
            "has_seal_amount": int("seal_amount" in df.columns),
            "has_limit_type": int("limit_type" in df.columns),
            "has_up_limit": int("up_limit" in df.columns),
            "has_down_limit": int("down_limit" in df.columns),
            "has_limit_times": int("limit_times" in df.columns),
            "has_up_stat": int("up_stat" in df.columns),
        }

    return df


def _enforce_close_limit_up_pool(
    trade_date: str,
    limit_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    stk_limit_df: pd.DataFrame,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    系统级硬定义（你确认的口径）：
    limit_list_d.csv = 当日“收盘真实涨停”的股票集合
    判定：close == up_limit（允许极小容差）
    同时回填：up_limit / down_limit / limit_type='U'
    """
    meta.setdefault("derived", {})
    info: Dict[str, Any] = {
        "policy": "CLOSE_EQ_UP_LIMIT",
        "trade_date": trade_date,
        "input_rows": int(0),
        "output_rows": int(0),
        "used_daily_close": 0,
        "used_stk_limit": 0,
        "filled_up_limit": 0,
        "filled_down_limit": 0,
        "warn": "",
    }

    if limit_df is None:
        limit_df = pd.DataFrame()
    if daily_df is None:
        daily_df = pd.DataFrame()
    if stk_limit_df is None:
        stk_limit_df = pd.DataFrame()

    df = limit_df.copy()
    info["input_rows"] = int(len(df))

    # 必备：ts_code
    if df.empty or "ts_code" not in df.columns:
        info["warn"] = "limit_list_d missing ts_code or empty"
        meta["derived"]["limit_list_d_policy"] = info
        return pd.DataFrame(columns=list(df.columns) if not df.empty else None)

    df["ts_code"] = df["ts_code"].map(_norm_ts_code)

    # 1) close：优先用 limit_list_d.close；缺则从 daily.close 回填
    if ("close" not in df.columns) or df["close"].isna().all():
        if (not daily_df.empty) and ("ts_code" in daily_df.columns) and ("close" in daily_df.columns):
            tmp = daily_df.copy()
            tmp["ts_code"] = tmp["ts_code"].map(_norm_ts_code)
            tmp = tmp[["ts_code", "close"]].drop_duplicates(subset=["ts_code"])
            df = df.merge(tmp, on="ts_code", how="left", suffixes=("", "_daily"))
            info["used_daily_close"] = 1
        else:
            info["warn"] = "missing close and daily.close not available"
            meta["derived"]["limit_list_d_policy"] = info
            return pd.DataFrame(columns=list(df.columns))

    # 2) up_limit/down_limit：优先用 df 自带；缺则从 stk_limit 回填
    need_stk = False
    if ("up_limit" not in df.columns) or df["up_limit"].isna().all():
        need_stk = True
    if ("down_limit" not in df.columns) or df["down_limit"].isna().all():
        need_stk = True

    if need_stk:
        if (not stk_limit_df.empty) and ("ts_code" in stk_limit_df.columns):
            tmp = stk_limit_df.copy()
            tmp["ts_code"] = tmp["ts_code"].map(_norm_ts_code)
            cols = ["ts_code"]
            if "up_limit" in tmp.columns:
                cols.append("up_limit")
            if "down_limit" in tmp.columns:
                cols.append("down_limit")
            tmp = tmp[cols].drop_duplicates(subset=["ts_code"])
            df = df.merge(tmp, on="ts_code", how="left", suffixes=("", "_stk"))
            info["used_stk_limit"] = 1

            # 合并后如果出现 up_limit_stk/down_limit_stk 这类列，兜底回填
            if "up_limit" in df.columns and "up_limit_stk" in df.columns:
                m = df["up_limit"].isna() | (df["up_limit"].astype(str).str.strip() == "")
                df.loc[m, "up_limit"] = df.loc[m, "up_limit_stk"]
                info["filled_up_limit"] = int(m.sum())
                df = df.drop(columns=["up_limit_stk"], errors="ignore")

            if "down_limit" in df.columns and "down_limit_stk" in df.columns:
                m = df["down_limit"].isna() | (df["down_limit"].astype(str).str.strip() == "")
                df.loc[m, "down_limit"] = df.loc[m, "down_limit_stk"]
                info["filled_down_limit"] = int(m.sum())
                df = df.drop(columns=["down_limit_stk"], errors="ignore")
        else:
            info["warn"] = "up_limit/down_limit missing and stk_limit not available"
            meta["derived"]["limit_list_d_policy"] = info
            return pd.DataFrame(columns=list(df.columns))

    # 3) 数值判定：close == up_limit（容差）
    if ("close" not in df.columns) or ("up_limit" not in df.columns):
        info["warn"] = "missing close or up_limit after fill"
        meta["derived"]["limit_list_d_policy"] = info
        return pd.DataFrame(columns=list(df.columns))

    close_num = _to_num(df["close"])
    up_num = _to_num(df["up_limit"])

    # 允许极小容差：绝对差 <= 1e-6 或相对差 <= 1e-6
    eps_abs = float(os.getenv("LIMIT_UP_EPS_ABS", "1e-6"))
    eps_rel = float(os.getenv("LIMIT_UP_EPS_REL", "1e-6"))

    valid = close_num.notna() & up_num.notna()
    diff = (close_num - up_num).abs()
    rel = diff / up_num.abs().replace(0, pd.NA)
    is_limit_up = valid & ((diff <= eps_abs) | (rel <= eps_rel))

    out = df[is_limit_up].copy()

    # 4) 强制写 limit_type='U'
    if "limit_type" not in out.columns:
        out["limit_type"] = "U"
    else:
        out["limit_type"] = "U"

    info["output_rows"] = int(len(out))
    meta["derived"]["limit_list_d_policy"] = info
    return out


# =========================
# 盘中升级：集合竞价 + 分钟 + 分时特征
# =========================
def safe_query(pro, api_name: str, *, fields: str = "", **params) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tushare 新接口兼容查询封装。
    任何失败都只返回结构化错误，不打印 token，不影响日频主链路。
    """
    info: Dict[str, Any] = {
        "ok": False,
        "api_name": api_name,
        "rows": 0,
        "columns": [],
        "error": "",
    }
    try:
        kwargs = dict(params)
        if fields:
            kwargs["fields"] = fields

        fn = getattr(pro, api_name, None)
        if callable(fn):
            df = fn(**kwargs)
        elif hasattr(pro, "query"):
            df = pro.query(api_name, **kwargs)
        else:
            raise RuntimeError(f"Tushare api not available: {api_name}")

        if df is None:
            df = pd.DataFrame()
        info["ok"] = True
        info["rows"] = int(len(df))
        info["columns"] = [str(c) for c in df.columns]
        print(f"[SAFE-QUERY] {api_name} params={params} rows={len(df)}")
        return df, info
    except Exception as e:
        info["error"] = repr(e)
        print(f"[SAFE-QUERY-FAILED] {api_name} params={params} err={repr(e)}")
        return pd.DataFrame(), info


def fetch_limit_step_optional(pro, trade_date: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tushare 原生连板/天梯源。

    只做可选增强：接口或字段不可用时返回空表，不影响主数据链路。
    """
    fields = "trade_date,ts_code,name,limit_times,up_stat"
    df, info = safe_query(pro, "limit_step", fields=fields, trade_date=trade_date)
    if info.get("ok"):
        return df, info

    fallback, fallback_info = safe_query(pro, "limit_step", trade_date=trade_date)
    fallback_info["fallback_from_fields"] = info
    return fallback, fallback_info


def _stage_text_from_limit_times(v: Any) -> str:
    try:
        n = int(float(str(v).strip()))
    except Exception:
        return ""
    if n <= 0:
        return ""
    return f"{n}→{n + 1}"


def _stage_prior_from_limit_times(v: Any) -> float:
    try:
        n = int(float(str(v).strip()))
    except Exception:
        return float("nan")
    priors = {
        1: 0.16,
        2: 0.35,
        3: 0.43,
        4: 0.515,
        5: 0.535,
        6: 0.44,
    }
    return float(priors.get(n, 0.35 if n >= 7 else float("nan")))


def _stage_quality_weight_from_limit_times(v: Any) -> float:
    try:
        n = int(float(str(v).strip()))
    except Exception:
        return 1.0
    # 用户交易口径：3→4、4→5 是晋阶质量顶点，两边自然滑落。
    weights = {
        1: 0.78,
        2: 0.92,
        3: 1.10,
        4: 1.10,
        5: 1.00,
        6: 0.88,
    }
    return float(weights.get(n, 0.72 if n >= 7 else 1.0))


def _stage_risk_weight_from_limit_times(v: Any) -> float:
    try:
        n = int(float(str(v).strip()))
    except Exception:
        return 0.0
    risks = {
        1: 0.035,
        2: 0.015,
        3: 0.000,
        4: 0.005,
        5: 0.045,
        6: 0.095,
    }
    return float(risks.get(n, 0.160 if n >= 7 else 0.0))


def derive_limit_stage(
    trade_date: str,
    limit_list_df: pd.DataFrame,
    limit_step_df: pd.DataFrame,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    生成晋阶因子。

    主原则：limit_times 必须来自 Tushare 原生字段，不用 daily/stk_limit 自行推导。
    """
    meta.setdefault("derived", {})
    info: Dict[str, Any] = {
        "source": "",
        "limit_list_rows": int(len(limit_list_df)) if isinstance(limit_list_df, pd.DataFrame) else 0,
        "limit_step_rows": int(len(limit_step_df)) if isinstance(limit_step_df, pd.DataFrame) else 0,
        "rows": 0,
        "limit_times_nonnull": 0,
        "missing_reason": "",
    }

    if limit_list_df is None:
        limit_list_df = pd.DataFrame()
    if limit_step_df is None:
        limit_step_df = pd.DataFrame()

    base_cols = ["trade_date", "ts_code", "name", "limit_times", "up_stat"]
    if limit_list_df.empty or "ts_code" not in limit_list_df.columns:
        info["missing_reason"] = "limit_list_d_empty_or_missing_ts_code"
        out = pd.DataFrame(columns=_limit_stage_columns())
        meta["derived"]["limit_stage"] = info
        return out, info

    out = pd.DataFrame()
    out["trade_date"] = (
        limit_list_df["trade_date"].astype(str) if "trade_date" in limit_list_df.columns else pd.Series([trade_date] * len(limit_list_df))
    )
    out["ts_code"] = limit_list_df["ts_code"].map(_norm_ts_code)
    out["name"] = limit_list_df["name"].astype(str) if "name" in limit_list_df.columns else ""

    source = "missing"
    if "limit_times" in limit_list_df.columns and limit_list_df["limit_times"].notna().any():
        out["limit_times"] = limit_list_df["limit_times"]
        out["up_stat"] = limit_list_df["up_stat"] if "up_stat" in limit_list_df.columns else pd.NA
        source = "limit_list_d.limit_times"
    elif (not limit_step_df.empty) and "ts_code" in limit_step_df.columns and "limit_times" in limit_step_df.columns:
        step = limit_step_df.copy()
        step["ts_code"] = step["ts_code"].map(_norm_ts_code)
        keep = ["ts_code", "limit_times"]
        if "up_stat" in step.columns:
            keep.append("up_stat")
        if "name" in step.columns:
            keep.append("name")
        step = step[keep].drop_duplicates(subset=["ts_code"], keep="first")
        out = out.merge(step, on="ts_code", how="left", suffixes=("", "_step"))
        if "name_step" in out.columns:
            name_blank = out["name"].astype(str).str.strip().eq("")
            out.loc[name_blank, "name"] = out.loc[name_blank, "name_step"]
            out = out.drop(columns=["name_step"], errors="ignore")
        if "up_stat" not in out.columns:
            out["up_stat"] = pd.NA
        source = "limit_step.limit_times"
    else:
        out["limit_times"] = pd.NA
        out["up_stat"] = limit_list_df["up_stat"] if "up_stat" in limit_list_df.columns else pd.NA
        info["missing_reason"] = "tushare_limit_times_missing"

    out["advance_stage"] = out["limit_times"].map(_stage_text_from_limit_times)
    out["晋阶"] = out["advance_stage"]
    out["stage_quality_weight"] = out["limit_times"].map(_stage_quality_weight_from_limit_times)
    out["stage_risk_weight"] = out["limit_times"].map(_stage_risk_weight_from_limit_times)
    out["stage_prior"] = out["limit_times"].map(_stage_prior_from_limit_times)
    out["stage_source"] = source

    info["source"] = source
    info["rows"] = int(len(out))
    info["limit_times_nonnull"] = int(pd.to_numeric(out.get("limit_times"), errors="coerce").notna().sum()) if len(out) else 0
    if info["limit_times_nonnull"] <= 0 and not info["missing_reason"]:
        info["missing_reason"] = "limit_times_all_empty"

    meta["derived"]["limit_stage"] = info
    cols = _limit_stage_columns()
    return out.reindex(columns=cols, fill_value=""), info


def _read_symbol_priority(path: Path) -> List[str]:
    df = load_csv(path)
    if df.empty or "ts_code" not in df.columns:
        return []
    return [_norm_ts_code(x) for x in df["ts_code"].tolist() if _norm_ts_code(x)]


def build_intraday_universe(day_dir: Path) -> List[str]:
    """
    候选池优先级：
    wp_pre_candidates -> limit_list_d -> limit_break_d -> top_list，去重后保持顺序。
    """
    out: List[str] = []
    seen = set()
    for name in ("wp_pre_candidates.csv", "limit_list_d.csv", "limit_break_d.csv", "top_list.csv"):
        for code in _read_symbol_priority(day_dir / name):
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    return out


def build_wp_pre_candidates(
    trade_date: str,
    base_raw: Path,
    base_latest: Path,
    dfs: Dict[str, pd.DataFrame],
    *,
    min_pct_chg: float = 6.0,
) -> List[str]:
    daily = dfs.get("daily", pd.DataFrame())
    limit_list = dfs.get("limit_list_d", pd.DataFrame())
    if daily is None or daily.empty or "ts_code" not in daily.columns:
        save_df(pd.DataFrame(columns=["ts_code", "trade_date", "pct_chg"]), base_raw / "wp_pre_candidates.csv", columns=["ts_code", "trade_date", "pct_chg"])
        save_df(pd.DataFrame(columns=["ts_code", "trade_date", "pct_chg"]), base_latest / "wp_pre_candidates.csv", columns=["ts_code", "trade_date", "pct_chg"])
        return []
    work = daily.copy()
    work["ts_code"] = work["ts_code"].map(_norm_ts_code)
    pct = _to_num(work["pct_chg"]) if "pct_chg" in work.columns else pd.Series([0] * len(work), index=work.index)
    current_limit = set()
    if limit_list is not None and not limit_list.empty and "ts_code" in limit_list.columns:
        current_limit = set(limit_list["ts_code"].map(_norm_ts_code).tolist())
    out = work.loc[(pct > min_pct_chg) & (~work["ts_code"].isin(current_limit)), ["ts_code"]].copy()
    out.insert(1, "trade_date", trade_date)
    out["pct_chg"] = pct.loc[out.index]
    out = out.drop_duplicates("ts_code")
    save_df(out, base_raw / "wp_pre_candidates.csv", columns=["ts_code", "trade_date", "pct_chg"])
    save_df(out, base_latest / "wp_pre_candidates.csv", columns=["ts_code", "trade_date", "pct_chg"])
    return out["ts_code"].tolist()


def fetch_auction(pro, trade_date: str, symbols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    fields = "ts_code,trade_date,vol,price,amount"
    errors: List[Dict[str, str]] = []

    df, info = safe_query(pro, "stk_auction", fields=fields, trade_date=trade_date)
    if info["ok"] and not df.empty:
        return df, info

    # 部分权限/版本可能不支持全市场查询，退化到候选池逐只查询。
    frames: List[pd.DataFrame] = []
    for ts_code in (symbols or []):
        one, one_info = safe_query(pro, "stk_auction", fields=fields, trade_date=trade_date, ts_code=ts_code)
        if one_info["ok"] and not one.empty:
            frames.append(one)
        elif one_info["error"]:
            errors.append({"ts_code": ts_code, "error": one_info["error"]})

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=_fields_to_columns(fields))

    return out, {
        "ok": True,
        "api_name": "stk_auction",
        "rows": int(len(out)),
        "columns": [str(c) for c in out.columns],
        "error": info.get("error", ""),
        "fallback_symbol_errors": errors[:20],
    }


def build_auction_features(
    auction_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    columns = _auction_feature_columns()
    if auction_df is None or auction_df.empty or "ts_code" not in auction_df.columns:
        return pd.DataFrame(columns=columns)
    auction = auction_df.copy()
    auction["ts_code"] = auction["ts_code"].map(_norm_ts_code)
    if "trade_date" not in auction.columns:
        auction["trade_date"] = trade_date
    auction["auction_price"] = _to_num(auction["price"]) if "price" in auction.columns else pd.NA
    auction["auction_vol"] = _to_num(auction["vol"]) if "vol" in auction.columns else 0
    auction["auction_amount"] = _to_num(auction["amount"]) if "amount" in auction.columns else 0

    base = pd.DataFrame()
    if daily_df is not None and not daily_df.empty and "ts_code" in daily_df.columns:
        keep = [c for c in ["ts_code", "pre_close", "amount"] if c in daily_df.columns]
        base = daily_df[keep].copy()
        base["ts_code"] = base["ts_code"].map(_norm_ts_code)
    out = auction[["ts_code", "trade_date", "auction_price", "auction_vol", "auction_amount"]].drop_duplicates("ts_code")
    if not base.empty:
        out = out.merge(base.drop_duplicates("ts_code"), on="ts_code", how="left")
    pre_close = _to_num(out["pre_close"]) if "pre_close" in out.columns else pd.Series([0] * len(out), index=out.index)
    day_amount = _to_num(out["amount"]) if "amount" in out.columns else pd.Series([0] * len(out), index=out.index)
    out["auction_pct_chg"] = (out["auction_price"] / pre_close.replace(0, pd.NA) - 1) * 100
    out["auction_amount_ratio"] = out["auction_amount"] / day_amount.replace(0, pd.NA)
    out["auction_strength_score"] = (
        out["auction_pct_chg"].fillna(0).clip(-5, 10) * 6
        + out["auction_amount_ratio"].fillna(0).clip(0, 0.25) * 180
    ).clip(0, 100)
    return out.reindex(columns=columns)


def fetch_minute_for_symbol(
    pro,
    ts_code: str,
    trade_date: str,
    freq: str = "1min",
    end_dt: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    day = _trade_date_dash(trade_date)
    if end_dt is None:
        end_text = f"{day} 15:00:00"
    else:
        end_text = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    fields = "ts_code,trade_time,open,high,low,close,vol,amount"
    params = {
        "ts_code": ts_code,
        "start_date": f"{day} 09:30:00",
        "end_date": end_text,
        "freq": freq,
    }
    df, info = safe_query(pro, "stk_mins", fields=fields, **params)
    if df.empty:
        df = pd.DataFrame(columns=_fields_to_columns(fields))
        if "ts_code" in df.columns:
            df["ts_code"] = pd.Series(dtype="str")
    return df, info


def fetch_market_minutes(
    pro,
    trade_date: str,
    freq: str = "1min",
    end_dt: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    day = _trade_date_dash(trade_date)
    if end_dt is None:
        end_text = f"{day} 15:00:00"
    else:
        end_text = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    fields = "ts_code,trade_time,open,high,low,close,vol,amount"
    return safe_query(
        pro,
        "stk_mins",
        fields=fields,
        start_date=f"{day} 09:30:00",
        end_date=end_text,
        freq=freq,
    )


def _limit_price_map(stk_limit_df: pd.DataFrame) -> Dict[str, float]:
    if stk_limit_df is None or stk_limit_df.empty or "ts_code" not in stk_limit_df.columns or "up_limit" not in stk_limit_df.columns:
        return {}
    tmp = stk_limit_df.copy()
    tmp["ts_code"] = tmp["ts_code"].map(_norm_ts_code)
    tmp["up_limit"] = _to_num(tmp["up_limit"])
    tmp = tmp.dropna(subset=["up_limit"]).drop_duplicates(subset=["ts_code"])
    return {str(row["ts_code"]): float(row["up_limit"]) for _, row in tmp.iterrows()}


def _minute_time_series(df: pd.DataFrame) -> pd.Series:
    for col in ("trade_time", "datetime", "time", "trade_date"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series([pd.NaT] * len(df), index=df.index)


def _score_reseal_speed(avg_minutes: Optional[float], reseal_count: int) -> float:
    if reseal_count <= 0 or avg_minutes is None or pd.isna(avg_minutes):
        return 0.0
    if avg_minutes <= 3:
        return _clip_score(100 - (avg_minutes - 1) * 5)
    if avg_minutes <= 8:
        return _clip_score(90 - (avg_minutes - 4) * 4)
    if avg_minutes <= 15:
        return _clip_score(70 - (avg_minutes - 9) * 3)
    return _clip_score(50 - min(30, avg_minutes - 15))


def _empty_intraday_row(ts_code: str, trade_date: str, minute_freq: str) -> Dict[str, Any]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "minute_freq": minute_freq,
        "minute_rows": 0,
        "has_minute_data": 0,
        "first_limit_time": "",
        "last_limit_time": "",
        "limit_touch_count": 0,
        "open_board_count": 0,
        "max_drawdown_after_limit": "",
        "reseal_count": 0,
        "reseal_minutes_avg": "",
        "reseal_speed_score": 0.0,
        "reseal_acceptance_score": 0.0,
        "intraday_vwap": "",
        "intraday_vwap_position": "",
        "late_volume_ratio": "",
        "late_price_change_pct": "",
        "late_price_weakness": "",
        "late_limit_hold_minutes": 0,
        "late_withdraw_score": 0.0,
        "limitup_path_score": 0.0,
        "limitup_quality_score": 0.0,
        "intraday_risk_score": 100.0,
        "intraday_tag": "missing_minute_data",
    }


def _build_one_intraday_feature(
    ts_code: str,
    df: pd.DataFrame,
    up_limit: Optional[float],
    trade_date: str,
    minute_freq: str,
) -> Dict[str, Any]:
    row = _empty_intraday_row(ts_code, trade_date, minute_freq)
    if df is None or df.empty:
        return row
    if up_limit is None or pd.isna(up_limit) or up_limit <= 0:
        row["minute_rows"] = int(len(df))
        row["has_minute_data"] = 1
        row["intraday_tag"] = "missing_limit_price"
        return row

    work = df.copy()
    times = _minute_time_series(work)
    work["_dt"] = times
    work = work.dropna(subset=["_dt"]).sort_values("_dt")
    if work.empty or "close" not in work.columns:
        return row

    close = _to_num(work["close"])
    high = _to_num(work["high"]) if "high" in work.columns else close
    low = _to_num(work["low"]) if "low" in work.columns else close
    vol = _to_num(work["vol"]) if "vol" in work.columns else pd.Series([0] * len(work), index=work.index)
    amount = _to_num(work["amount"]) if "amount" in work.columns else pd.Series([0] * len(work), index=work.index)
    last_close = float(close.dropna().iloc[-1]) if not close.dropna().empty else float("nan")
    total_vol = float(vol.fillna(0).sum())
    total_amount = float(amount.fillna(0).sum())
    intraday_vwap = total_amount / total_vol if total_vol > 0 and total_amount > 0 else float("nan")
    if pd.isna(intraday_vwap) or intraday_vwap <= 0:
        typical = ((high + low + close) / 3).dropna()
        intraday_vwap = float(typical.mean()) if not typical.empty else float("nan")
    row["intraday_vwap"] = round(intraday_vwap, 4) if not pd.isna(intraday_vwap) else ""
    row["intraday_vwap_position"] = round((last_close / intraday_vwap - 1) * 100, 4) if intraday_vwap and not pd.isna(intraday_vwap) and intraday_vwap > 0 else ""

    late_mask_all = work["_dt"].dt.strftime("%H:%M:%S") >= "14:30:00"
    late_all = work[late_mask_all].copy()
    late_vol_all = float(_to_num(late_all["vol"]).fillna(0).sum()) if (not late_all.empty and "vol" in late_all.columns) else 0.0
    row["late_volume_ratio"] = round(late_vol_all / total_vol, 6) if total_vol > 0 else 0.0
    if not late_all.empty and "close" in late_all.columns:
        late_close_series = _to_num(late_all["close"]).dropna()
        if not late_close_series.empty:
            first_late = float(late_close_series.iloc[0])
            last_late = float(late_close_series.iloc[-1])
            row["late_price_change_pct"] = round((last_late / first_late - 1) * 100, 4) if first_late > 0 else ""

    eps_abs = float(os.getenv("LIMIT_UP_EPS_ABS", "1e-6"))
    eps_rel = float(os.getenv("LIMIT_UP_EPS_REL", "1e-6"))
    limit_hit = (high - up_limit).abs().le(eps_abs) | ((high - up_limit).abs() / abs(up_limit) <= eps_rel) | (high >= up_limit)
    limit_close = (close - up_limit).abs().le(eps_abs) | ((close - up_limit).abs() / abs(up_limit) <= eps_rel) | (close >= up_limit)

    hit_positions = [i for i, v in enumerate(limit_hit.fillna(False).tolist()) if v]
    row["minute_rows"] = int(len(work))
    row["has_minute_data"] = 1
    if not hit_positions:
        row["late_price_weakness"] = ""
        row["intraday_risk_score"] = 80.0
        row["intraday_tag"] = "no_limit_touch"
        return row

    first_i = hit_positions[0]
    last_i = hit_positions[-1]
    first_time = work.iloc[first_i]["_dt"]
    last_time = work.iloc[last_i]["_dt"]
    row["first_limit_time"] = first_time.strftime("%H:%M:%S")
    row["last_limit_time"] = last_time.strftime("%H:%M:%S")

    touch_states = limit_hit.fillna(False).tolist()
    limit_touch_count = 0
    prev_touch = False
    for cur in touch_states:
        if cur and not prev_touch:
            limit_touch_count += 1
        prev_touch = cur

    states = limit_close.fillna(False).tolist()
    open_board_count = 0
    reseal_durations: List[float] = []
    open_start: Optional[pd.Timestamp] = None
    prev = False
    seen_limit = False
    for i, cur in enumerate(states):
        ts = work.iloc[i]["_dt"]
        if cur and not prev:
            if seen_limit and open_start is not None:
                reseal_durations.append(max(0.0, (ts - open_start).total_seconds() / 60.0))
                open_start = None
            seen_limit = True
        if seen_limit and prev and not cur:
            open_board_count += 1
            open_start = ts
        prev = cur

    after_first_low = low.iloc[first_i:]
    max_drawdown_after_limit = 0.0
    if after_first_low.notna().any():
        max_drawdown_after_limit = max(0.0, (up_limit - float(after_first_low.min())) / up_limit * 100.0)

    late_mask = work["_dt"].dt.strftime("%H:%M:%S") >= "14:30:00"
    late = work[late_mask].copy()
    late_vol = float(_to_num(late["vol"]).fillna(0).sum()) if (not late.empty and "vol" in late.columns) else 0.0
    late_volume_ratio = late_vol / total_vol if total_vol > 0 else 0.0
    late_last_close = float(_to_num(late["close"]).dropna().iloc[-1]) if (not late.empty and "close" in late.columns and not _to_num(late["close"]).dropna().empty) else float(close.dropna().iloc[-1])
    late_price_weakness = max(0.0, (up_limit - late_last_close) / up_limit * 100.0)
    late_close = limit_close[late_mask]
    late_limit_hold_minutes = int(late_close.fillna(False).sum())
    late_open_events = 0
    prev_late = True
    for cur in late_close.fillna(False).tolist():
        if prev_late and not cur:
            late_open_events += 1
        prev_late = cur

    reseal_count = len(reseal_durations)
    reseal_avg = sum(reseal_durations) / reseal_count if reseal_count else None
    reseal_speed = _score_reseal_speed(reseal_avg, reseal_count)
    reseal_acceptance = _clip_score(100 - open_board_count * 12 - max_drawdown_after_limit * 6 - (100 - reseal_speed) * 0.35)

    late_withdraw = _clip_score(late_price_weakness * 15 + max(0.0, late_volume_ratio - 0.18) * 180 + late_open_events * 12)
    if late_limit_hold_minutes == 0:
        late_withdraw = max(late_withdraw, 70.0)

    first_minutes = first_time.hour * 60 + first_time.minute
    early_bonus = 25 if first_minutes <= 10 * 60 else 12 if first_minutes <= 11 * 60 else 0
    late_penalty = 25 if first_minutes >= 14 * 60 + 30 else 0
    path_score = _clip_score(65 + early_bonus - open_board_count * 10 - (100 - reseal_speed) * 0.2 - late_penalty - late_withdraw * 0.25)
    quality_score = _clip_score(path_score * 0.45 + reseal_acceptance * 0.3 + (100 - late_withdraw) * 0.25)

    open_board_risk = _clip_score(open_board_count * 22)
    reseal_slow_risk = _clip_score(100 - reseal_speed if open_board_count else 15)
    path_weakness_risk = _clip_score(100 - path_score)
    intraday_risk = _clip_score(
        0.35 * open_board_risk
        + 0.25 * reseal_slow_risk
        + 0.25 * late_withdraw
        + 0.15 * path_weakness_risk
    )

    if late_withdraw >= 65:
        tag = "late_withdraw_risk"
    elif open_board_count >= 3:
        tag = "high_open_board_risk"
    elif quality_score >= 75 and intraday_risk <= 35:
        tag = "healthy_limitup"
    elif quality_score >= 55:
        tag = "neutral_limitup"
    else:
        tag = "weak_limitup"

    row.update(
        {
            "limit_touch_count": int(limit_touch_count),
            "open_board_count": int(open_board_count),
            "max_drawdown_after_limit": round(max_drawdown_after_limit, 4),
            "reseal_count": int(reseal_count),
            "reseal_minutes_avg": round(reseal_avg, 4) if reseal_avg is not None else "",
            "reseal_speed_score": round(reseal_speed, 4),
            "reseal_acceptance_score": round(reseal_acceptance, 4),
            "late_volume_ratio": round(late_volume_ratio, 6),
            "late_price_change_pct": row.get("late_price_change_pct", ""),
            "late_price_weakness": round(late_price_weakness, 4),
            "late_limit_hold_minutes": int(late_limit_hold_minutes),
            "late_withdraw_score": round(late_withdraw, 4),
            "limitup_path_score": round(path_score, 4),
            "limitup_quality_score": round(quality_score, 4),
            "intraday_risk_score": round(intraday_risk, 4),
            "intraday_tag": tag,
        }
    )
    return row


def build_intraday_features(
    minute_frames: Dict[str, pd.DataFrame],
    limit_price_map: Dict[str, float],
    trade_date: str,
    minute_freq: str,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    codes = symbols or sorted(set(minute_frames.keys()) | set(limit_price_map.keys()))
    rows = [
        _build_one_intraday_feature(
            ts_code=code,
            df=minute_frames.get(code, pd.DataFrame()),
            up_limit=limit_price_map.get(code),
            trade_date=trade_date,
            minute_freq=minute_freq,
        )
        for code in codes
    ]
    return pd.DataFrame(rows, columns=_intraday_feature_columns())


def _known_symbol_universe(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    out: List[str] = []
    seen = set()
    for key in ("stock_basic", "daily", "daily_basic", "stk_limit"):
        df = dfs.get(key, pd.DataFrame())
        if df is None or df.empty or "ts_code" not in df.columns:
            continue
        for code in df["ts_code"].map(_norm_ts_code).tolist():
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    return out


def fetch_realtime_quotes(symbols: List[str], trade_date: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"ok": False, "rows": 0, "chunks": 0, "error": ""}
    chunk_size = max(1, _env_int("REALTIME_QUOTE_CHUNK_SIZE", 800))
    max_symbols = _env_int("MAX_REALTIME_QUOTE_SYMBOLS", 6000)
    symbols = [s for s in symbols if s]
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    frames: List[pd.DataFrame] = []
    fn = getattr(ts, "realtime_quote", None)
    if not callable(fn):
        info["error"] = "tushare.realtime_quote not available"
        return pd.DataFrame(columns=_realtime_quote_columns()), info
    try:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            if not chunk:
                continue
            df = fn(ts_code=",".join(chunk))
            info["chunks"] += 1
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            info["ok"] = True
            return pd.DataFrame(columns=_realtime_quote_columns()), info
        raw = pd.concat(frames, ignore_index=True, sort=False)
        col_map = {str(c).strip().lower(): c for c in raw.columns}

        def pick(*names: str) -> pd.Series:
            for name in names:
                col = col_map.get(name.lower())
                if col is not None:
                    return raw[col]
            return pd.Series([pd.NA] * len(raw), index=raw.index)

        out = pd.DataFrame()
        out["ts_code"] = pick("ts_code", "code", "symbol").map(_norm_ts_code)
        out["trade_date"] = trade_date
        date_text = pick("date", "trade_date").fillna("").astype(str)
        time_text = pick("time", "trade_time").fillna("").astype(str)
        out["update_time"] = (date_text + " " + time_text).str.strip()
        out.loc[out["update_time"].str.strip().eq(""), "update_time"] = bj_now().strftime("%Y-%m-%d %H:%M:%S")
        out["name"] = pick("name")
        out["price"] = _to_num(pick("price", "close", "最新价"))
        out["open"] = _to_num(pick("open", "开盘价"))
        out["high"] = _to_num(pick("high", "最高价"))
        out["low"] = _to_num(pick("low", "最低价"))
        out["pre_close"] = _to_num(pick("pre_close", "preclose", "昨收"))
        pct = _to_num(pick("pct_chg", "pct_change", "change_pct", "涨跌幅"))
        calc_pct = (out["price"] / out["pre_close"].replace(0, pd.NA) - 1) * 100
        out["pct_chg"] = pct.where(pct.notna() & (pct != 0), calc_pct)
        out["vol"] = _to_num(pick("vol", "volume", "成交量"))
        out["amount"] = _to_num(pick("amount", "成交额"))
        out = out[out["ts_code"] != ""].drop_duplicates("ts_code")
        info["ok"] = True
        info["rows"] = int(len(out))
        return out.reindex(columns=_realtime_quote_columns()), info
    except Exception as e:
        info["error"] = repr(e)
        return pd.DataFrame(columns=_realtime_quote_columns()), info


def _previous_close_map(trade_date: str, dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    base: Dict[str, float] = {}
    daily = dfs.get("daily", pd.DataFrame())
    if daily is not None and not daily.empty and "ts_code" in daily.columns:
        tmp = daily.copy()
        tmp["ts_code"] = tmp["ts_code"].map(_norm_ts_code)
        price_col = "pre_close" if "pre_close" in tmp.columns and _to_num(tmp["pre_close"]).notna().any() else "close"
        if price_col in tmp.columns:
            tmp[price_col] = _to_num(tmp[price_col])
            tmp = tmp.dropna(subset=[price_col]).drop_duplicates("ts_code")
            base.update({str(row["ts_code"]): float(row[price_col]) for _, row in tmp.iterrows() if float(row[price_col]) > 0})

    raw_root = Path("data/raw") / trade_date[:4]
    if raw_root.exists():
        prev_dates = sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name < trade_date)
        for prev in reversed(prev_dates[-10:]):
            prev_daily = load_csv(raw_root / prev / "daily.csv")
            if prev_daily.empty or "ts_code" not in prev_daily.columns or "close" not in prev_daily.columns:
                continue
            prev_daily = prev_daily.copy()
            prev_daily["ts_code"] = prev_daily["ts_code"].map(_norm_ts_code)
            prev_daily["close"] = _to_num(prev_daily["close"])
            prev_daily = prev_daily.dropna(subset=["close"]).drop_duplicates("ts_code")
            for _, row in prev_daily.iterrows():
                code = str(row["ts_code"])
                close = float(row["close"])
                if code and close > 0 and code not in base:
                    base[code] = close
            if base:
                break
    return base


def build_realtime_snapshot(
    minute_frames: Dict[str, pd.DataFrame],
    trade_date: str,
    dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pre_close_map = _previous_close_map(trade_date, dfs)
    for ts_code, frame in minute_frames.items():
        if frame is None or frame.empty:
            continue
        work = frame.copy()
        if "ts_code" not in work.columns:
            work["ts_code"] = ts_code
        times = _minute_time_series(work)
        work["_dt"] = times
        work = work.dropna(subset=["_dt"]).sort_values("_dt")
        if work.empty or "close" not in work.columns:
            continue
        close = _to_num(work["close"])
        valid_close = close.dropna()
        if valid_close.empty:
            continue
        first = work.iloc[0]
        last = work.iloc[-1]
        price = float(valid_close.iloc[-1])
        pre_close = float(pre_close_map.get(ts_code, 0.0))
        pct_chg = (price / pre_close - 1) * 100 if pre_close > 0 else float("nan")
        high = _to_num(work["high"]).max() if "high" in work.columns else valid_close.max()
        low = _to_num(work["low"]).min() if "low" in work.columns else valid_close.min()
        open_price = _to_num(pd.Series([first.get("open", first.get("close", pd.NA))])).iloc[0]
        vol = _to_num(work["vol"]).fillna(0).sum() if "vol" in work.columns else 0
        amount = _to_num(work["amount"]).fillna(0).sum() if "amount" in work.columns else 0
        rows.append(
            {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "update_time": last["_dt"].strftime("%Y-%m-%d %H:%M:%S"),
                "price": price,
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "pre_close": pre_close if pre_close > 0 else pd.NA,
                "pct_chg": pct_chg,
                "vol": vol,
                "amount": amount,
                "realtime_source": "stk_mins",
            }
        )
    out = pd.DataFrame(rows, columns=_realtime_snapshot_columns())
    if not out.empty:
        out = out.sort_values(["pct_chg", "amount"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out


def build_quote_realtime_snapshot(quote_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if quote_df is None or quote_df.empty or "ts_code" not in quote_df.columns:
        return pd.DataFrame(columns=_realtime_snapshot_columns())
    out = quote_df.copy()
    out["ts_code"] = out["ts_code"].map(_norm_ts_code)
    out["trade_date"] = trade_date
    out["price"] = _to_num(out["price"]) if "price" in out.columns else 0
    out["close"] = out["price"]
    for col in ["open", "high", "low", "pre_close", "pct_chg", "vol", "amount"]:
        out[col] = _to_num(out[col]) if col in out.columns else pd.NA
    if "update_time" not in out.columns:
        out["update_time"] = bj_now().strftime("%Y-%m-%d %H:%M:%S")
    out["realtime_source"] = "realtime_quote_fallback"
    out = out[out["ts_code"] != ""].drop_duplicates("ts_code")
    out = out.sort_values(["pct_chg", "amount"], ascending=[False, False], na_position="last").reset_index(drop=True)
    return out.reindex(columns=_realtime_snapshot_columns())


def run_intraday_upgrade(
    pro,
    trade_date: str,
    base_raw: Path,
    base_latest: Path,
    dfs: Dict[str, pd.DataFrame],
    meta: Dict[str, Any],
) -> None:
    enable_auction = _env_bool("ENABLE_AUCTION", "1")
    enable_minute = _env_bool("ENABLE_MINUTE", "1")
    realtime_minute_only = _env_bool("REALTIME_MINUTE_ONLY", "1")
    enable_market_minute_scan = _env_bool("ENABLE_MARKET_MINUTE_SCAN", "0")
    minute_freq = os.getenv("MINUTE_FREQ", "1min").strip() or "1min"
    max_symbols = max(0, _env_int("MAX_MINUTE_SYMBOLS", 6000 if enable_market_minute_scan else 80))

    wp_symbols = build_wp_pre_candidates(
        trade_date,
        base_raw,
        base_latest,
        dfs,
        min_pct_chg=float(os.getenv("WP_INTRADAY_MIN_PCT", "6")),
    )
    priority_symbols = build_intraday_universe(base_raw)
    quote_df = pd.DataFrame(columns=_realtime_quote_columns())
    quote_info: Dict[str, Any] = {"ok": False, "rows": 0, "error": "disabled"}
    if enable_market_minute_scan:
        quote_df, quote_info = fetch_realtime_quotes(_known_symbol_universe(dfs), trade_date)
        save_df(quote_df, base_raw / "realtime_quote.csv", columns=_realtime_quote_columns())
        save_df(quote_df, base_latest / "realtime_quote.csv", columns=_realtime_quote_columns())
        quote_candidates: List[str] = []
        if not quote_df.empty and "pct_chg" in quote_df.columns:
            quote_candidates = quote_df.loc[_to_num(quote_df["pct_chg"]) > float(os.getenv("WP_INTRADAY_MIN_PCT", "6")), "ts_code"].map(_norm_ts_code).tolist()
        seen = set()
        symbols = []
        for code in priority_symbols + quote_candidates:
            if code and code not in seen:
                seen.add(code)
                symbols.append(code)
    else:
        symbols = priority_symbols
        save_df(pd.DataFrame(), base_raw / "realtime_quote.csv", columns=_realtime_quote_columns())
        save_df(pd.DataFrame(), base_latest / "realtime_quote.csv", columns=_realtime_quote_columns())
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    limit_map = _limit_price_map(dfs.get("stk_limit", pd.DataFrame()))
    now = bj_now()
    is_today = trade_date == now.strftime("%Y%m%d")
    before_minute_open = is_today and now.strftime("%H:%M:%S") < "09:30:00"
    minute_end_dt: Optional[datetime] = None
    if is_today and not before_minute_open:
        minute_end_dt = min(now, datetime.strptime(f"{trade_date}150000", "%Y%m%d%H%M%S").replace(tzinfo=BJ_TZ))

    meta["auction"] = {"enabled": enable_auction, "ok": False, "rows": 0, "columns": [], "error": ""}
    meta["minute"] = {
        "enabled": enable_minute,
        "realtime_only": realtime_minute_only,
        "market_minute_scan": enable_market_minute_scan,
        "realtime_quote": quote_info,
        "ok": False,
        "freq": minute_freq,
        "wp_pre_candidates": int(len(wp_symbols)),
        "minute_end": minute_end_dt.strftime("%Y-%m-%d %H:%M:%S") if minute_end_dt else "",
        "symbols_requested": int(len(symbols)) if enable_minute else 0,
        "symbols_success": 0,
        "symbols_failed": 0,
        "errors": [],
    }
    meta["intraday_features"] = {"ok": False, "rows": 0, "columns": [], "error": ""}
    meta["realtime_snapshot"] = {"ok": False, "rows": 0, "columns": _realtime_snapshot_columns(), "error": ""}

    if enable_auction:
        try:
            auction_df, auction_info = fetch_auction(pro, trade_date, symbols=symbols)
            save_df(auction_df, base_raw / "stk_auction.csv", columns=list(auction_df.columns) or _auction_columns())
            save_df(auction_df, base_latest / "stk_auction.csv", columns=list(auction_df.columns) or _auction_columns())
            auction_features = build_auction_features(auction_df, dfs.get("daily", pd.DataFrame()), trade_date)
            save_df(auction_features, base_raw / "auction_features.csv", columns=_auction_feature_columns())
            save_df(auction_features, base_latest / "auction_features.csv", columns=_auction_feature_columns())
            meta["auction"].update(
                {
                    "ok": bool(auction_info.get("ok", False)),
                    "rows": int(len(auction_df)),
                    "feature_rows": int(len(auction_features)),
                    "columns": [str(c) for c in auction_df.columns],
                    "error": auction_info.get("error", ""),
                }
            )
            if auction_info.get("fallback_symbol_errors"):
                meta["auction"]["fallback_symbol_errors"] = auction_info["fallback_symbol_errors"]
        except Exception as e:
            meta["auction"]["error"] = repr(e)
            save_df(pd.DataFrame(), base_raw / "stk_auction.csv", columns=_auction_columns())
            save_df(pd.DataFrame(), base_latest / "stk_auction.csv", columns=_auction_columns())
            save_df(pd.DataFrame(), base_raw / "auction_features.csv", columns=_auction_feature_columns())
            save_df(pd.DataFrame(), base_latest / "auction_features.csv", columns=_auction_feature_columns())
            print(f"[AUCTION-FAILED] err={repr(e)}")
            print(traceback.format_exc())
    else:
        save_df(pd.DataFrame(), base_raw / "stk_auction.csv", columns=_auction_columns())
        save_df(pd.DataFrame(), base_latest / "stk_auction.csv", columns=_auction_columns())
        save_df(pd.DataFrame(), base_raw / "auction_features.csv", columns=_auction_feature_columns())
        save_df(pd.DataFrame(), base_latest / "auction_features.csv", columns=_auction_feature_columns())

    minute_frames: Dict[str, pd.DataFrame] = {}
    raw_minute_dir = base_raw / "minute" / minute_freq
    latest_minute_dir = base_latest / "minute" / minute_freq
    ensure_dir(raw_minute_dir)
    _clear_csv_dir(latest_minute_dir)

    if enable_minute and realtime_minute_only and (not is_today or before_minute_open):
        meta["minute"]["ok"] = True
        if not is_today:
            meta["minute"]["skip_reason"] = f"REALTIME_MINUTE_ONLY enabled; trade_date {trade_date} is not today {now:%Y%m%d}"
        else:
            meta["minute"]["skip_reason"] = "REALTIME_MINUTE_ONLY enabled; skip stk_mins before 09:30 Beijing time"
        for ts_code in symbols:
            save_df(pd.DataFrame(columns=_minute_columns()), raw_minute_dir / f"{ts_code}.csv", columns=_minute_columns())
        _clear_csv_dir(latest_minute_dir)
    elif enable_minute and symbols:
        if enable_market_minute_scan and _env_bool("TRY_FULL_MARKET_MINUTE", "0"):
            try:
                market_df, market_info = fetch_market_minutes(pro, trade_date, minute_freq, end_dt=minute_end_dt)
                meta["minute"]["market_query"] = market_info
                if market_info.get("ok") and not market_df.empty and "ts_code" in market_df.columns:
                    market_df = market_df.copy()
                    market_df["ts_code"] = market_df["ts_code"].map(_norm_ts_code)
                    for ts_code, group in market_df.groupby("ts_code", sort=False):
                        if ts_code in symbols:
                            minute_frames[ts_code] = group.drop(columns=[], errors="ignore").copy()
                    meta["minute"]["symbols_success"] = int(len(minute_frames))
                else:
                    meta["minute"]["market_query_empty"] = True
            except Exception as e:
                meta["minute"]["market_query"] = {"ok": False, "error": repr(e)}
                print(f"[MARKET-MINUTE-FAILED] err={repr(e)}")

        fallback_symbols = [code for code in symbols if code not in minute_frames]
        if fallback_symbols:
            meta["minute"]["fallback_symbol_query_count"] = int(len(fallback_symbols))
        for ts_code in fallback_symbols:
            try:
                df, info = fetch_minute_for_symbol(pro, ts_code, trade_date, minute_freq, end_dt=minute_end_dt)
                minute_frames[ts_code] = df
                if info.get("ok") and not df.empty:
                    meta["minute"]["symbols_success"] += 1
                else:
                    meta["minute"]["symbols_failed"] += 1
                    if info.get("error"):
                        meta["minute"]["errors"].append({"ts_code": ts_code, "error": info["error"]})
            except Exception as e:
                minute_frames[ts_code] = pd.DataFrame(columns=_minute_columns())
                meta["minute"]["symbols_failed"] += 1
                meta["minute"]["errors"].append({"ts_code": ts_code, "error": repr(e)})
                print(f"[MINUTE-FAILED] ts_code={ts_code} err={repr(e)}")

        for ts_code, df in minute_frames.items():
            save_df(df, raw_minute_dir / f"{ts_code}.csv", columns=list(df.columns) or _minute_columns())
            save_df(df, latest_minute_dir / f"{ts_code}.csv", columns=list(df.columns) or _minute_columns())
        meta["minute"]["errors"] = meta["minute"]["errors"][:30]
        meta["minute"]["ok"] = True
    else:
        meta["minute"]["ok"] = True

    try:
        realtime_snapshot = build_realtime_snapshot(
            minute_frames=minute_frames,
            trade_date=trade_date,
            dfs=dfs,
        )
        if realtime_snapshot.empty and not quote_df.empty:
            realtime_snapshot = build_quote_realtime_snapshot(quote_df, trade_date)
            meta["realtime_snapshot"]["fallback"] = "realtime_quote"
        save_df(realtime_snapshot, base_raw / "realtime_snapshot.csv", columns=_realtime_snapshot_columns())
        save_df(realtime_snapshot, base_latest / "realtime_snapshot.csv", columns=_realtime_snapshot_columns())
        meta["realtime_snapshot"].update({"ok": True, "rows": int(len(realtime_snapshot)), "error": ""})
    except Exception as e:
        meta["realtime_snapshot"]["error"] = repr(e)
        save_df(pd.DataFrame(), base_raw / "realtime_snapshot.csv", columns=_realtime_snapshot_columns())
        save_df(pd.DataFrame(), base_latest / "realtime_snapshot.csv", columns=_realtime_snapshot_columns())
        print(f"[REALTIME-SNAPSHOT-FAILED] err={repr(e)}")
        print(traceback.format_exc())

    try:
        features = build_intraday_features(
            minute_frames=minute_frames,
            limit_price_map=limit_map,
            trade_date=trade_date,
            minute_freq=minute_freq,
            symbols=symbols,
        )
        save_df(features, base_raw / "intraday_features.csv", columns=_intraday_feature_columns())
        save_df(features, base_latest / "intraday_features.csv", columns=_intraday_feature_columns())
        meta["intraday_features"].update(
            {
                "ok": True,
                "rows": int(len(features)),
                "columns": [str(c) for c in features.columns],
                "error": "",
            }
        )
    except Exception as e:
        meta["intraday_features"]["error"] = repr(e)
        save_df(pd.DataFrame(), base_raw / "intraday_features.csv", columns=_intraday_feature_columns())
        save_df(pd.DataFrame(), base_latest / "intraday_features.csv", columns=_intraday_feature_columns())
        print(f"[INTRADAY-FEATURES-FAILED] err={repr(e)}")
        print(traceback.format_exc())


def main():
    requested_trade_date = os.getenv("TRADE_DATE", "").strip()

    pro = get_pro()
    trade_date = resolve_trade_date(pro, requested_trade_date)

    year = trade_date[:4]
    base_raw = Path("data/raw") / year / trade_date
    base_latest = Path("data/latest")
    ensure_dir(base_raw)
    ensure_dir(base_latest)

    retry_cfg = RetryConfig(
        max_retry=int(os.getenv("MAX_RETRY", "10")),
        base_sleep_sec=float(os.getenv("BASE_SLEEP_SEC", "2")),
        max_sleep_sec=float(os.getenv("MAX_SLEEP_SEC", "20")),
        jitter_sec=float(os.getenv("JITTER_SEC", "0.8")),
    )

    meta: Dict[str, Any] = {
        "requested_trade_date": requested_trade_date or None,
        "resolved_trade_date": trade_date,
        "generated_at_bj": bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": [],
        "derived": {},
    }

    jobs = build_jobs(pro, trade_date)

    any_required_failed = False
    dfs: Dict[str, pd.DataFrame] = {}          # 原始 df（后面可二次加工）
    job_columns: Dict[str, List[str]] = {}     # 每个 job 的列契约（用于 save_df）

    # 先抓取所有表（不急着对 limit_list_d 做最终过滤）
    for job in jobs:
        out_csv = base_raw / f"{job.key}.csv"
        out_latest = base_latest / f"{job.key}.csv"

        job_columns[job.key] = job.columns

        job_record: Dict[str, Any] = {
            "key": job.key,
            "note": job.note,
            "allow_empty": job.allow_empty,
            "required": job.required,
            "kwargs": job.kwargs,
            "status": "unknown",
            "rows": None,
            "error": None,
        }

        try:
            df = call_with_retry(
                job.fn,
                retry=retry_cfg,
                allow_empty=job.allow_empty,
                empty_ok_after_retry=True,
                **job.kwargs,
            )

            df = _postprocess_limit_tables(df, job.key, meta)

            dfs[job.key] = df

            # 先按原始抓取结果落盘（limit_list_d 之后会被“强制推导版”覆盖一次）
            save_df(df, out_csv, columns=job.columns)
            save_df(df, out_latest, columns=job.columns)

            job_record["status"] = "ok" if (df is not None and not df.empty) else "ok_empty"
            job_record["rows"] = int(len(df)) if df is not None else 0

        except Exception as e:
            job_record["status"] = "failed"
            job_record["error"] = repr(e)

            print(f"[JOB-FAILED] {job.key} err={repr(e)}")
            print(traceback.format_exc())

            dfs[job.key] = pd.DataFrame()

            try:
                save_df(pd.DataFrame(), out_csv, columns=job.columns)
                save_df(pd.DataFrame(), out_latest, columns=job.columns)
            except Exception:
                pass

            if job.required:
                any_required_failed = True

        meta["jobs"].append(job_record)

    # =========================
    # 系统级强制约束（根治点）：
    # limit_list_d.csv 最终必须是“收盘真实涨停池”
    # 依据：close == up_limit（用 daily + stk_limit 推导与回填）
    # =========================
    try:
        limit_df = dfs.get("limit_list_d", pd.DataFrame())
        daily_df = dfs.get("daily", pd.DataFrame())
        stk_df = dfs.get("stk_limit", pd.DataFrame())

        enforced = _enforce_close_limit_up_pool(
            trade_date=trade_date,
            limit_df=limit_df,
            daily_df=daily_df,
            stk_limit_df=stk_df,
            meta=meta,
        )

        # 覆盖写出（raw + latest）
        out_csv = base_raw / "limit_list_d.csv"
        out_latest = base_latest / "limit_list_d.csv"

        cols = job_columns.get("limit_list_d") or (list(enforced.columns) if not enforced.empty else [])
        save_df(enforced, out_csv, columns=cols)
        save_df(enforced, out_latest, columns=cols)

        # 同步内存
        dfs["limit_list_d"] = enforced

        print(
            f"[LIMIT_LIST_ENFORCED] trade_date={trade_date} rows={len(enforced)} "
            f"policy={meta.get('derived', {}).get('limit_list_d_policy', {}).get('policy')}"
        )

    except Exception as e:
        # 不让整个 workflow 因此挂掉，但会在 meta 中记录失败
        meta.setdefault("derived", {})
        meta["derived"]["limit_list_d_policy"] = {"status": "failed", "error": repr(e)}
        print(f"[LIMIT_LIST_ENFORCED-FAILED] err={repr(e)}")
        print(traceback.format_exc())

    try:
        limit_step_df, limit_step_info = fetch_limit_step_optional(pro, trade_date)
        meta["limit_step"] = limit_step_info
        limit_step_cols = list(limit_step_df.columns) if not limit_step_df.empty else ["trade_date", "ts_code", "name", "limit_times", "up_stat"]
        save_df(limit_step_df, base_raw / "limit_step.csv", columns=limit_step_cols)
        save_df(limit_step_df, base_latest / "limit_step.csv", columns=limit_step_cols)
        dfs["limit_step"] = limit_step_df

        limit_stage_df, limit_stage_info = derive_limit_stage(
            trade_date=trade_date,
            limit_list_df=dfs.get("limit_list_d", pd.DataFrame()),
            limit_step_df=limit_step_df,
            meta=meta,
        )
        save_df(limit_stage_df, base_raw / "limit_stage.csv", columns=_limit_stage_columns())
        save_df(limit_stage_df, base_latest / "limit_stage.csv", columns=_limit_stage_columns())
        dfs["limit_stage"] = limit_stage_df
        print(
            f"[LIMIT_STAGE] trade_date={trade_date} rows={len(limit_stage_df)} "
            f"source={limit_stage_info.get('source')} nonnull={limit_stage_info.get('limit_times_nonnull')}"
        )
    except Exception as e:
        meta.setdefault("derived", {})
        meta["derived"]["limit_stage"] = {"status": "failed", "error": repr(e)}
        try:
            save_df(pd.DataFrame(), base_raw / "limit_stage.csv", columns=_limit_stage_columns())
            save_df(pd.DataFrame(), base_latest / "limit_stage.csv", columns=_limit_stage_columns())
        except Exception:
            pass
        print(f"[LIMIT_STAGE-FAILED] err={repr(e)}")
        print(traceback.format_exc())

    safe_json_dump(meta, base_raw / "_meta.json")
    safe_json_dump(meta, base_latest / "_meta.json")

    try:
        derived_info = derive_hot_board_tags(trade_date, base_raw, base_latest)
        meta["derived"]["hot_board_tags"] = derived_info
        safe_json_dump(meta, base_raw / "_meta.json")
        safe_json_dump(meta, base_latest / "_meta.json")
    except Exception as e:
        meta["derived"]["hot_board_tags"] = {"status": "failed", "error": repr(e)}
        safe_json_dump(meta, base_raw / "_meta.json")
        safe_json_dump(meta, base_latest / "_meta.json")
        print(f"[DERIVED-FAILED] hot_board_tags err={repr(e)}")
        print(traceback.format_exc())

    try:
        run_intraday_upgrade(
            pro=pro,
            trade_date=trade_date,
            base_raw=base_raw,
            base_latest=base_latest,
            dfs=dfs,
            meta=meta,
        )
        safe_json_dump(meta, base_raw / "_meta.json")
        safe_json_dump(meta, base_latest / "_meta.json")
    except Exception as e:
        meta["auction"] = meta.get("auction", {"enabled": _env_bool("ENABLE_AUCTION", "1")})
        meta["minute"] = meta.get("minute", {"enabled": _env_bool("ENABLE_MINUTE", "1")})
        meta["intraday_features"] = {"ok": False, "rows": 0, "columns": [], "error": repr(e)}
        safe_json_dump(meta, base_raw / "_meta.json")
        safe_json_dump(meta, base_latest / "_meta.json")
        print(f"[INTRADAY-UPGRADE-FAILED] err={repr(e)}")
        print(traceback.format_exc())

    if any_required_failed:
        raise RuntimeError("Some required jobs failed. Check data/raw/.../_meta.json for details.")

    print("[DONE] snapshots saved.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
