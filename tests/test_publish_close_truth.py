from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "wp" / "publish_close_truth.py"
SPEC = importlib.util.spec_from_file_location("publish_close_truth", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_validate_frame_accepts_complete_truth() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_code": "688506.SH",
                "trade_date": "20260716",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "pre_close": 1,
                "vol": 10,
                "amount": 20,
                "pct_chg": 100,
            }
        ]
    )
    result = MODULE.validate_frame(
        frame,
        label="daily",
        trade_date="20260716",
        required_columns=MODULE.DAILY_FIELDS,
        min_rows=1,
    )
    assert result["ts_code"].tolist() == ["688506.SH"]


def test_validate_frame_rejects_partial_truth() -> None:
    frame = pd.DataFrame([{"ts_code": "688506.SH", "trade_date": "20260716"}])
    with pytest.raises(RuntimeError, match="missing columns"):
        MODULE.validate_frame(
            frame,
            label="daily",
            trade_date="20260716",
            required_columns=MODULE.DAILY_FIELDS,
            min_rows=1,
        )


def test_validate_frame_rejects_too_few_rows() -> None:
    frame = pd.DataFrame(
        [{"ts_code": "688506.SH", "trade_date": "20260716", "up_limit": 10, "down_limit": 5}]
    )
    with pytest.raises(RuntimeError, match="minimum is 2"):
        MODULE.validate_frame(
            frame,
            label="stk_limit",
            trade_date="20260716",
            required_columns=MODULE.LIMIT_FIELDS,
            min_rows=2,
        )
