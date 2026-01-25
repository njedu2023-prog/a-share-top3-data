import os, json, time
from pathlib import Path
from datetime import datetime
from dateutil import tz

import tushare as ts
import pandas as pd

BJ_TZ = tz.gettz("Asia/Shanghai")

# =========================
# 基础工具
# =========================
def bj_now():
    return datetime.now(BJ_TZ)

def bj_today_yyyymmdd():
    return bj_now().strftime("%Y%m%d")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, out_csv: Path):
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def get_pro():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing env TUSHARE_TOKEN. Set it in GitHub Secrets or local env.")
    ts.set_token(token)
    return ts.pro_api()

def call_with_retry(fn, *, max_retry=10, sleep_sec=8, **kwargs):
    last_err = None
    for i in range(1, max_retry + 1):
        try:
            df = fn(**kwargs)
            if df is None or df.empty:
                raise RuntimeError(f"Empty dataframe for {fn.__name__} kwargs={kwargs}")
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"Failed after {max_retry} retries: {last_err}")

# =========================
# 主程序：抓取并落地快照
# =========================
def main():
    # 允许手动回补：workflow_dispatch / 本地运行时可传入
    trade_date = os.getenv("TRADE_DATE", "").strip()
    if not trade_date:
        trade_date = bj_today_yyyymmdd()

    year = trade_date[:4]
    base_raw = Path("data/raw") / year / trade_date
    base_latest = Path("data/latest")
    ensure_dir(base_raw)
    ensure_dir(base_latest)

    pro = get_pro()

    # --- 核心四表（冻结主链路） ---
    df_limit_d = call_with_retry(pro.limit_list_d, trade_date=trade_date)
    df_limit_ths = call_with_retry(pro.limit_list_ths, trade_date=trade_date)
    df_limit_step = call_with_retry(pro.limit_step, trade_date=trade_date)
    df_limit_cpt = call_with_retry(pro.limit_cpt_list, trade_date=trade_date)

    # 龙虎榜（可选，不阻塞主链路）
    enable_dragon = os.getenv("ENABLE_DRAGON_TIGER", "0").strip() == "1"
    df_top_list = None
    if enable_dragon:
        df_top_list = call_with_retry(pro.top_list, trade_date=trade_date)

    # --- 写 raw 快照 ---
    raw_files = {}
    raw_files["limit_list_d"] = base_raw / "limit_list_d.csv"
    raw_files["limit_list_ths"] = base_raw / "limit_list_ths.csv"
    raw_files["limit_step"] = base_raw / "limit_step.csv"
    raw_files["limit_cpt_list"] = base_raw / "limit_cpt_list.csv"
    raw_files["top_list"] = base_raw / "top_list.csv"

    save_df(df_limit_d, raw_files["limit_list_d"])
    save_df(df_limit_ths, raw_files["limit_list_ths"])
    save_df(df_limit_step, raw_files["limit_step"])
    save_df(df_limit_cpt, raw_files["limit_cpt_list"])
    if df_top_list is not None:
        save_df(df_top_list, raw_files["top_list"])

    # --- 同步 latest（覆盖） ---
    save_df(df_limit_d, base_latest / "limit_list_d.csv")
    save_df(df_limit_ths, base_latest / "limit_list_ths.csv")
    save_df(df_limit_step, base_latest / "limit_step.csv")
    save_df(df_limit_cpt, base_latest / "limit_cpt_list.csv")
    if df_top_list is not None:
        save_df(df_top_list, base_latest / "top_list.csv")

    # --- 元信息（审计/复盘用） ---
    meta = {
        "trade_date": trade_date,
        "generated_at_bj": bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "enable_dragon_tiger": enable_dragon,
        "frozen_sources_v0_1": {
            "limit_list_d": "封板时间(first_time/last_time)、炸板次数(open_times)、封单金额(fd_amount)、连板数(limit_times)",
            "limit_list_ths": "涨停池/首板/题材原因(lu_desc)/形态标签(tag)",
            "limit_step": "连板天梯（高度结构）",
            "limit_cpt_list": "最强题材/轮动结构",
            "top_list": "龙虎榜（可选开关）",
        },
        "raw_dir": str(base_raw),
        "latest_dir": str(base_latest),
    }

    (base_raw / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (base_latest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK:", json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
