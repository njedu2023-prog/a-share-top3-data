#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_limit_list_d.py

用途：
- 读取 limit_list_d.csv（或同结构 CSV）
- 只保留“涨停 / 封板”相关记录（尽量不误伤；若无法判断则可选择不过滤）
- 输出到指定路径（默认覆盖写出）

判定优先级（从强到弱）：
1) 若存在 limit_type 字段：匹配包含 Up/涨停/封板 等关键词
2) 若存在 close 与 up_limit：close >= up_limit（允许极小误差）
3) 若存在 pct_chg：>= 9.8（兼容 10% 涨停的粗判；可自行调整）
4) 以上都没有：默认不做过滤（避免误伤成空）

用法示例：
python tools/filter_limit_list_d.py \
  --in data/raw/2026/20260130/limit_list_d.csv \
  --out data/raw/2026/20260130/limit_list_d.csv \
  --mode only_limit_up
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


UP_KEYWORDS = [
    "U", "UP", "LIMIT_UP", "涨停", "封板", "涨停板", "涨停价", "一字板", "封死",
]


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _has_cols(df: pd.DataFrame, *cols: str) -> bool:
    s = set(df.columns)
    return all(c in s for c in cols)


def _parse_float(v) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def is_limit_up_row(row: pd.Series, eps: float = 1e-6) -> bool:
    # 1) limit_type 强判
    if "limit_type" in row.index:
        lt = _safe_str(row.get("limit_type"))
        if lt:
            lt_up = lt.upper()
            # 兼容中文/英文/单字母等
            if any(k in lt_up for k in ["UP", "U", "LIMIT_UP"]) or any(k in lt for k in ["涨停", "封板"]):
                return True

    # 2) close 与 up_limit
    if "close" in row.index and "up_limit" in row.index:
        close = _parse_float(row.get("close"))
        up_limit = _parse_float(row.get("up_limit"))
        if close is not None and up_limit is not None:
            if close + eps >= up_limit:
                return True

    # 3) pct_chg 粗判（可按需改成 9.9 / 9.95 / 19.8 等分档）
    if "pct_chg" in row.index:
        pct = _parse_float(row.get("pct_chg"))
        if pct is not None and pct >= 9.8:
            return True

    return False


def filter_limit_list(
    df: pd.DataFrame,
    mode: str = "only_limit_up",
) -> pd.DataFrame:
    mode = (mode or "").strip().lower()

    if df.empty:
        return df

    if mode in ("none", "no_filter", "keep_all"):
        return df

    if mode not in ("only_limit_up",):
        raise ValueError(f"Unknown mode: {mode}")

    # 若完全没有任何可判定字段，则不过滤，避免误伤为全空
    judgeable = any(c in df.columns for c in ("limit_type", "close", "up_limit", "pct_chg"))
    if not judgeable:
        return df

    mask = df.apply(is_limit_up_row, axis=1)
    out = df.loc[mask].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input csv path")
    ap.add_argument("--out", dest="out_path", required=True, help="output csv path")
    ap.add_argument("--mode", default="only_limit_up", help="only_limit_up | none")
    ap.add_argument("--encoding", default="utf-8-sig", help="csv encoding (default utf-8-sig)")
    ap.add_argument("--eps", type=float, default=1e-6, help="epsilon for close>=up_limit compare")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, dtype=str)  # 先全按 str 读，避免脏数据导致失败

    # 将数值列尽量转回数值（不强制）
    for c in ("close", "up_limit", "down_limit", "pct_chg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 保证表头字段存在时按原样输出；若缺字段也不强行补（避免破坏上游结构）
    out_df = filter_limit_list(df, mode=args.mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding=args.encoding)

    print(
        f"[OK] mode={args.mode} in_rows={len(df)} out_rows={len(out_df)} "
        f"in={in_path} out={out_path}"
    )


if __name__ == "__main__":
    main()
