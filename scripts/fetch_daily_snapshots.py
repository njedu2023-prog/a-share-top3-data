import os
import json
import time
from pathlib import Path
from datetime import datetime
from dateutil import tz

import tushare as ts
import pandas as pd

BJ_TZ = tz.gettz("Asia/Shanghai")

# =========================
# 基础工具
# =========================
def bj_now() -> datetime:
    return datetime.now(BJ_TZ)

def bj_today_yyyymmdd() -> str:
    return bj_now().strftime("%Y%m%d")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, out_csv: Path):
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def save_json(obj: dict, out_json: Path):
    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def get_pro():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing env TUSHARE_TOKEN. Set it in GitHub Secrets or local env.")
    ts.set_token(token)
    return ts.pro_api()

def fn_name(fn) -> str:
    """
    兼容 functools.partial / pro.xxx 等对象，避免访问 __name__ 报错
    """
    name = getattr(fn, "__name__", None)
    if name:
        return name
    # partial / bound method / callable object
    return getattr(getattr(fn, "func", None), "__name__", None) or repr(fn)

def call_with_retry(fn, *, max_retry=10, sleep_sec=8, **kwargs) -> pd.DataFrame:
    last_err = None
    name = fn_name(fn)

    for i in range(1, max_retry + 1):
        try:
            df = fn(**kwargs)
            if df is None or df.empty:
                raise RuntimeError(f"Empty dataframe for {name} kwargs={kwargs}")
            return df
        except Exception as e:
            last_err = e
            # 最后一次就不 sleep 了
            if i < max_retry:
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

    # 你当前脚本里用到的：limit_list_d（涨跌停列表-日）
    df_limit_d = call_with_retry(pro.limit_list_d, trade_date=trade_date)

    # 落地：按日期目录
    out_csv = base_raw / "limit_list_d.csv"
    save_df(df_limit_d, out_csv)

    # 同时维护 latest 软链接式“最新文件”（用复制实现，兼容 GitHub Pages/Windows）
    out_latest_csv = base_latest / "limit_list_d.csv"
    save_df(df_limit_d, out_latest_csv)

    # 额外写一个 meta，方便你后面核验/前端展示
    meta = {
        "trade_date": trade_date,
        "generated_at_bj": bj_now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": int(len(df_limit_d)),
        "files": {
            "raw_csv": str(out_csv.as_posix()),
            "latest_csv": str(out_latest_csv.as_posix()),
        },
    }
    save_json(meta, base_raw / "meta.json")
    save_json(meta, base_latest / "meta.json")

    print(f"[OK] trade_date={trade_date} rows={len(df_limit_d)}")
    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_latest_csv}")

if __name__ == "__main__":
    main()
