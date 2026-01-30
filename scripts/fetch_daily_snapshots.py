import os
import time
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List

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
        # 可选：让期望列排在前面，其他列跟在后面
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


# =========================
# 交易日处理
# =========================
def resolve_trade_date(pro, requested_trade_date: str) -> str:
    """
    - 如果指定了 TRADE_DATE：优先用它；但若它不是交易日，则回退最近交易日
    - 如果未指定：默认用北京时间今天；若今天非交易日则回退最近交易日
    """
    target = requested_trade_date.strip() if requested_trade_date else bj_today_yyyymmdd()

    # 为了“最近交易日”回退，需要向前多取一些天
    end_date = target
    start_date = (datetime.strptime(target, "%Y%m%d") - pd.Timedelta(days=30)).strftime("%Y%m%d")

    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
    if cal is None or cal.empty:
        # 极端情况：拿不到日历就原样返回
        return target

    cal = cal.sort_values("cal_date")
    # target 当天是否开市
    row = cal[cal["cal_date"] == target]
    if not row.empty and int(row.iloc[0]["is_open"]) == 1:
        return target

    # 回退到 <= target 的最近开市日
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

                # 不允许空：触发重试
                raise RuntimeError(f"empty dataframe (not allowed): {name} kwargs={kwargs}")

            print(f"[OK] {name} kwargs={kwargs} -> rows={len(df)}")
            return df

        except Exception as e:
            last_err = e

            # 指数退避 + 抖动
            sleep = min(retry.max_sleep_sec, retry.base_sleep_sec * (2 ** (i - 1)))
            sleep = sleep + random.random() * retry.jitter_sec
            print(f"[RETRY {i}/{retry.max_retry}] {name} kwargs={kwargs} err={repr(e)} sleep={sleep:.1f}s")
            time.sleep(sleep)

    # 重试耗尽
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
    key: str                    # 输出文件名用
    fn: Callable                # pro 接口函数
    kwargs: Dict[str, Any]      # 参数
    columns: List[str]          # 期望输出表头（哪怕没数据也要写出来）
    allow_empty: bool = True    # 是否允许空
    required: bool = False      # 是否关键任务（关键任务失败是否要让整体失败）
    note: str = ""              # 备注


def _fields_to_columns(fields: Optional[str]) -> List[str]:
    if not fields:
        return []
    cols = [c.strip() for c in fields.split(",") if c.strip()]
    return cols


def build_jobs(pro, trade_date: str) -> List[FetchJob]:
    """
    日频数据仓库（打板Top10系统）所需的最小核心数据：
    1) limit_list_d      涨停/跌停列表（日）
    2) limit_break_d     炸板/开板（日）——用于情绪过滤炸板率
    3) daily             日线OHLCV+amount ——用于振幅/收盘接近最高/量能等代理因子
    4) stk_limit         涨跌停价 ——用于触板/一字等代理判断
    5) daily_basic       换手/市值 ——用于结构过滤与排序因子
    可选：
    6) stock_basic       股票基础信息（name/industry/list_date等）
    7) namechange        名称变更（用于ST/更名等过滤标记）
    8) top_list          龙虎榜（可选）
    9) moneyflow_hsgt    北向资金（可选）
    """
    jobs: List[FetchJob] = []

    # --- schemas（保证空表也有表头）---
    # 对于没指定 fields 的接口：至少保证 ts_code、trade_date 这两个关键列存在
    schema_min_code_date = ["ts_code", "trade_date"]

    # 1) 涨跌停列表（日）
    jobs.append(
        FetchJob(
            key="limit_list_d",
            fn=pro.limit_list_d,
            kwargs={"trade_date": trade_date},
            columns=schema_min_code_date,
            allow_empty=True,
            required=False,
            note="涨跌停列表（日）",
        )
    )

    # 2) 炸板/开板（日）
    if hasattr(pro, "limit_break_d"):
        jobs.append(
            FetchJob(
                key="limit_break_d",
                fn=pro.limit_break_d,
                kwargs={"trade_date": trade_date},
                columns=schema_min_code_date,  # 下游至少需要 ts_code 不然会炸
                allow_empty=True,
                required=False,
                note="炸板/开板（日）",
            )
        )

    # 3) 日线行情（日频核心）
    daily_fields = "ts_code,trade_date,open,high,low,close,vol,amount,pct_chg"
    jobs.append(
        FetchJob(
            key="daily",
            fn=pro.daily,
            kwargs={
                "trade_date": trade_date,
                "fields": daily_fields,
            },
            columns=_fields_to_columns(daily_fields),
            allow_empty=True,
            required=False,
            note="日线行情（OHLCV+amount）",
        )
    )

    # 4) 涨跌停价（用于触板/一字等代理指标）
    stk_limit_fields = "ts_code,trade_date,up_limit,down_limit"
    jobs.append(
        FetchJob(
            key="stk_limit",
            fn=pro.stk_limit,
            kwargs={
                "trade_date": trade_date,
                "fields": stk_limit_fields,
            },
            columns=_fields_to_columns(stk_limit_fields),
            allow_empty=True,
            required=False,
            note="涨跌停价（限制价格）",
        )
    )

    # 5) 每日指标（换手/市值等）
    daily_basic_fields = "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,total_mv,float_mv"
    jobs.append(
        FetchJob(
            key="daily_basic",
            fn=pro.daily_basic,
            kwargs={
                "trade_date": trade_date,
                "fields": daily_basic_fields,
            },
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
            kwargs={
                "exchange": "",
                "list_status": "L",
                "fields": stock_basic_fields,
            },
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
    """
    用“行业(=板块代理)”来做当日热门板块：
    - 统计当日涨停股在各行业的数量
    - 取 TopN 行业作为“热门板块”
    - 对每只涨停股打标签：是否热门板块、板块排名、板块涨停数
    """
    topn = int(os.getenv("HOT_BOARD_TOPN", "10"))

    limit_path = base_latest / "limit_list_d.csv"
    basic_path = base_latest / "stock_basic.csv"
    namechg_path = base_latest / "namechange.csv"

    limit_df = load_csv(limit_path)
    basic_df = load_csv(basic_path)
    namechg_df = load_csv(namechg_path)

    # 若没有涨停数据或基础信息，仍落空文件（但带表头）
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

    # 必须有 ts_code
    limit_df = limit_df.copy()
    if "ts_code" not in limit_df.columns:
        return {"hot_board_topn": topn, "hot_boards": 0, "tagged": 0, "warn": "limit_list_d missing ts_code"}

    # 股票基础信息：ts_code -> name/industry
    basic_df = basic_df.copy()
    keep_cols = [c for c in ["ts_code", "name", "industry"] if c in basic_df.columns]
    basic_df = basic_df[keep_cols].drop_duplicates(subset=["ts_code"])

    merged = limit_df.merge(basic_df, on="ts_code", how="left", suffixes=("", "_basic"))
    # 保底字段
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

    # 行业统计（热门板块）
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

    # 对涨停股打标签（去重 ts_code）
    tags = merged[["ts_code", "name", "industry"]].drop_duplicates(subset=["ts_code"]).copy()
    tags.insert(0, "trade_date", trade_date)

    tags["is_hot_board"] = tags["industry"].astype(str).isin(hot_industries).astype(int)
    tags["board_rank"] = tags["industry"].astype(str).map(rank_map).fillna("")
    tags["board_limit_up_count"] = tags["industry"].astype(str).map(cnt_map).fillna("")

    name_has_st = tags["name"].fillna("").astype(str).str.contains(r"ST|\*ST", regex=True, na=False)
    code_in_st_like = tags["ts_code"].fillna("").astype(str).isin(st_like)
    tags["is_st_like"] = (name_has_st | code_in_st_like).astype(int)

    # 落盘（保证表头）
    save_df(ind_stat, base_raw / "hot_boards.csv", columns=list(ind_stat.columns))
    save_df(tags, base_raw / "limit_up_tags.csv", columns=list(tags.columns))
    save_df(ind_stat, base_latest / "hot_boards.csv", columns=list(ind_stat.columns))
    save_df(tags, base_latest / "limit_up_tags.csv", columns=list(tags.columns))

    return {
        "hot_board_topn": topn,
        "hot_boards": int(len(ind_stat)),
        "tagged": int(len(tags)),
    }


# =========================
# 主程序：抓取并落地快照
# =========================
def main():
    # 允许手动回补：workflow_dispatch / 本地运行时可传入
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

    for job in jobs:
        out_csv = base_raw / f"{job.key}.csv"
        out_latest = base_latest / f"{job.key}.csv"

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
                empty_ok_after_retry=True,   # 核心：即便空也不让 workflow 挂
                **job.kwargs,
            )

            save_df(df, out_csv, columns=job.columns)
            save_df(df, out_latest, columns=job.columns)

            job_record["status"] = "ok" if not df.empty else "ok_empty"
            job_record["rows"] = int(len(df))

        except Exception as e:
            job_record["status"] = "failed"
            job_record["error"] = repr(e)

            print(f"[JOB-FAILED] {job.key} err={repr(e)}")
            print(traceback.format_exc())

            # 即使失败，也落一个“带表头”的空文件，保证下游不炸
            try:
                save_df(pd.DataFrame(), out_csv, columns=job.columns)
                save_df(pd.DataFrame(), out_latest, columns=job.columns)
            except Exception:
                pass

            if job.required:
                any_required_failed = True

        meta["jobs"].append(job_record)

    # 先写 meta（不影响稳定性）
    safe_json_dump(meta, base_raw / "_meta.json")
    safe_json_dump(meta, base_latest / "_meta.json")

    # 派生：热门板块标签（纯本地计算，几乎无算力）
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

    # 只有“required=True”的任务失败才让整体失败
    if any_required_failed:
        raise RuntimeError("Some required jobs failed. Check data/raw/.../_meta.json for details.")

    print("[DONE] snapshots saved.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
