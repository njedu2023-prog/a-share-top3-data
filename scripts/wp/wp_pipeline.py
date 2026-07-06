from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WP_ROOT = ROOT / "data" / "wp"
LATEST = WP_ROOT / "latest"

SCHEMA = [
    "trade_date", "update_time", "ts_code", "name", "price", "open", "high", "low",
    "close", "pre_close", "pct_chg", "amount", "volume", "turnover_rate",
    "volume_ratio", "sector_name", "sector_rank", "sector_limitup_count",
    "sector_gt6_count", "sector_amount_ratio", "pre_day_limitup", "today_limitup",
    "today_limit_up_price", "prev_limit_up_price", "ret_5d", "ret_20d",
]


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


def load_source() -> pd.DataFrame:
    source = os.environ.get("WP_SOURCE_CSV", "")
    if source and Path(source).exists():
        return pd.read_csv(source)
    return pd.DataFrame(columns=SCHEMA)


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
