from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "wp"))

import wp_pipeline  # noqa: E402


class _Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        self.read_count += 1
        return self.body


class WpPipelineSourceTest(unittest.TestCase):
    def test_remote_csv_is_downloaded_only_once(self) -> None:
        response = _Response(b"ts_code,close\n000001.SZ,12.5\n")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(wp_pipeline, "ROOT", Path(directory)),
                patch.object(wp_pipeline, "urlopen", return_value=response) as mocked_urlopen,
            ):
                frame = wp_pipeline.read_csv_source("data/raw/2026/20260716/daily.csv")

        self.assertEqual(mocked_urlopen.call_count, 1)
        self.assertEqual(response.read_count, 1)
        self.assertEqual(frame.loc[0, "ts_code"], "000001.SZ")

    def test_previous_limitup_uses_latest_available_remote_trade_date(self) -> None:
        source = pd.DataFrame({"ts_code": ["000001.SZ", "688001.SH"]})
        with (
            patch.object(
                wp_pipeline,
                "available_raw_dates",
                return_value=["20260714", "20260715", "20260716"],
            ),
            patch.object(wp_pipeline, "read_csv_source", return_value=source) as read_source,
        ):
            codes = wp_pipeline.previous_limitup_codes("20260716")

        self.assertEqual(codes, {"000001.SZ", "688001.SH"})
        read_source.assert_called_once_with("data/raw/2026/20260715/limit_list_d.csv")

    def test_available_dates_cross_year_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            (raw_root / "2025" / "20251231").mkdir(parents=True)
            (raw_root / "2026" / "20260102").mkdir(parents=True)
            with patch.object(wp_pipeline, "RAW_ROOT", raw_root):
                dates = wp_pipeline.available_raw_dates("20260105", lookback_days=10)

        self.assertIn("20251231", dates)
        self.assertIn("20260102", dates)

    def test_local_current_date_does_not_hide_remote_history(self) -> None:
        remote = [
            {"name": "20260715", "type": "dir"},
            {"name": "20260716", "type": "dir"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            (raw_root / "2026" / "20260716").mkdir(parents=True)
            with (
                patch.object(wp_pipeline, "RAW_ROOT", raw_root),
                patch.object(wp_pipeline, "read_json_url", return_value=remote),
            ):
                dates = wp_pipeline.available_raw_dates("20260716")

        self.assertIn("20260715", dates)
        self.assertIn("20260716", dates)

    def test_history_quality_rejects_fallback_only_features(self) -> None:
        frame = pd.DataFrame(
            {
                "pct_chg": [8.2, 9.1, 10.0],
                "ret_5d": [8.2, 9.1, 10.0],
                "amount_ratio_5d": [1.0, 1.0, 1.0],
                "ma5_position": [0.0, 0.0, 0.0],
                "ma20_position": [0.0, 0.0, 0.0],
            }
        )

        quality = wp_pipeline.historical_feature_quality(frame)

        self.assertFalse(quality["history_features_ready"])
        self.assertEqual(quality["history_feature_coverage_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
