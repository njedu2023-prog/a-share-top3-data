#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import tushare as ts


BASE_FIELDS = "trade_date,ts_code,name,limit_type,close,up_limit,down_limit,open_times,fd_amount,first_time,last_time"
EXT_FIELDS = BASE_FIELDS + ",limit_times,up_stat,industry,turnover_ratio,amount,float_mv,total_mv"


def stage_text(v: object) -> str:
    try:
        if pd.isna(v):
            return ""
        n = int(float(v))
    except Exception:
        return ""
    return f"{n}进{n + 1}" if n > 0 else ""


def quality_weight(v: object) -> float:
    try:
        n = int(float(v))
    except Exception:
        return float("nan")
    if n <= 1:
        return 0.78
    if n == 2:
        return 0.92
    if n in (3, 4):
        return 1.10
    if n == 5:
        return 1.00
    if n == 6:
        return 0.88
    return 0.72


def risk_weight(v: object) -> float:
    try:
        n = int(float(v))
    except Exception:
        return float("nan")
    if n <= 1:
        return 0.035
    if n == 2:
        return 0.015
    if n == 3:
        return 0.000
    if n == 4:
        return 0.005
    if n == 5:
        return 0.045
    if n == 6:
        return 0.095
    return 0.160


def read_trade_date(root: Path) -> str:
    env = str(os.environ.get("TRADE_DATE", "")).strip()
    if env:
        return env
    meta_path = root / "data/latest/_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ("resolved_trade_date", "requested_trade_date", "trade_date"):
                v = str(meta.get(k, "")).strip()
                if len(v) == 8 and v.isdigit():
                    return v
        except Exception:
            pass
    raise SystemExit("cannot resolve trade_date for limit stage")


def safe_call(func, **kwargs) -> pd.DataFrame:
    try:
        df = func(**kwargs)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:
        print(f"[limit_stage] optional tushare call failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def main() -> int:
    root = Path.cwd()
    trade_date = read_trade_date(root)
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN is empty")
    pro = ts.pro_api(token)

    ll = safe_call(pro.limit_list_d, trade_date=trade_date, fields=EXT_FIELDS)
    if ll.empty or "limit_times" not in ll.columns:
        fallback = safe_call(pro.limit_list_d, trade_date=trade_date, fields=BASE_FIELDS)
        if not fallback.empty:
            ll = fallback

    step = safe_call(pro.limit_step, trade_date=trade_date, fields="trade_date,ts_code,name,limit_times,up_stat")
    if step.empty:
        step = safe_call(pro.limit_step, trade_date=trade_date)

    frames = []
    if not ll.empty:
        x = ll.copy()
        x["stage_source"] = "limit_list_d"
        frames.append(x)
    if not step.empty:
        y = step.copy()
        y["stage_source"] = "limit_step"
        frames.append(y)

    cols = ["trade_date", "ts_code", "name", "limit_times", "advance_stage", "晋阶", "stage_quality_weight", "stage_risk_weight", "stage_prior", "stage_source", "up_stat"]
    if not frames:
        out = pd.DataFrame(columns=cols)
    else:
        raw = pd.concat(frames, ignore_index=True, sort=False)
        raw["trade_date"] = raw.get("trade_date", trade_date).fillna(trade_date).astype(str)
        raw["ts_code"] = raw["ts_code"].astype(str).str.strip()
        raw["limit_times"] = pd.to_numeric(raw.get("limit_times"), errors="coerce")
        raw = raw.sort_values(["ts_code", "limit_times"], ascending=[True, False], na_position="last")
        out = raw.drop_duplicates("ts_code", keep="first").copy()
        out["advance_stage"] = out["limit_times"].map(stage_text)
        out["晋阶"] = out["advance_stage"]
        out["stage_quality_weight"] = out["limit_times"].map(quality_weight)
        out["stage_risk_weight"] = out["limit_times"].map(risk_weight)
        out["stage_prior"] = out["limit_times"].map(lambda v: {1: 0.16, 2: 0.35, 3: 0.43, 4: 0.515, 5: 0.535, 6: 0.44}.get(int(v), 0.35) if pd.notna(v) else float("nan"))
        for c in cols:
            if c not in out.columns:
                out[c] = pd.NA
        out = out[cols]

    latest = root / "data/latest"
    raw_dir = root / f"data/raw/{trade_date[:4]}/{trade_date}"
    latest.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(latest / "limit_stage.csv", index=False, encoding="utf-8-sig")
    out.to_csv(raw_dir / "limit_stage.csv", index=False, encoding="utf-8-sig")
    if not step.empty:
        step.to_csv(latest / "limit_step.csv", index=False, encoding="utf-8-sig")
        step.to_csv(raw_dir / "limit_step.csv", index=False, encoding="utf-8-sig")
    print(f"[limit_stage] trade_date={trade_date} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
