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

    # 1) limit_list_d（注意：该接口在不同权限下可能不给 up_limit/limit_type，这里先抓原始表，后面用 daily+stk_limit 推导“收盘真实涨停池”）
    limit_list_fields = (
        "trade_date,ts_code,name,limit_type,close,up_limit,down_limit,"
        "open_times,fd_amount,first_time,last_time"
    )
    jobs.append(
        FetchJob(
            key="limit_list_d",
            fn=pro.limit_list_d,
            kwargs={"trade_date": trade_date, "fields": limit_list_fields},
            columns=_fields_to_columns(limit_list_fields) or schema_min_code_date,
            allow_empty=True,
            required=False,
            note="涨停池（日）（会在后处理中强制推导为“收盘真实涨停”并回填 up_limit/down_limit/limit_type）",
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
    daily_fields = "ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
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

    if any_required_failed:
        raise RuntimeError("Some required jobs failed. Check data/raw/.../_meta.json for details.")

    print("[DONE] snapshots saved.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
