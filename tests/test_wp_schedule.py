from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "wp"))

from wp_schedule import CN_TZ, due_decision, latest_due_slot, scheduled_slots  # noqa: E402


class WpScheduleTest(unittest.TestCase):
    def test_business_slots_include_normal_tail_and_final_cadence(self) -> None:
        slots = [slot.strftime("%H:%M") for slot in scheduled_slots(datetime(2026, 7, 16).date())]
        self.assertIn("09:25", slots)
        self.assertIn("14:15", slots)
        self.assertIn("14:20", slots)
        self.assertIn("14:25", slots)
        self.assertIn("14:55", slots)
        self.assertIn("15:05", slots)
        self.assertIn("15:10", slots)
        self.assertNotIn("14:10", slots[slots.index("14:15") + 1 :])

    def test_late_start_catches_latest_slot(self) -> None:
        current = datetime(2026, 7, 16, 13, 29, tzinfo=CN_TZ)
        self.assertEqual(latest_due_slot(current).strftime("%H:%M"), "13:25")

    def test_manifest_deduplicates_a_covered_slot(self) -> None:
        current = datetime(2026, 7, 16, 13, 29, tzinfo=CN_TZ)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps({"generated_at": "2026-07-16 13:27:00", "source_trade_date": "20260716"}),
                encoding="utf-8",
            )
            decision = due_decision(current, manifest)
        self.assertFalse(decision.should_run)
        self.assertEqual(decision.target_slot.strftime("%H:%M"), "13:25")

    def test_stale_manifest_runs_latest_slot(self) -> None:
        current = datetime(2026, 7, 16, 14, 37, tzinfo=CN_TZ)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps({"generated_at": "2026-07-16 14:24:00", "source_trade_date": "20260716"}),
                encoding="utf-8",
            )
            decision = due_decision(current, manifest)
        self.assertTrue(decision.should_run)
        self.assertEqual(decision.target_slot.strftime("%H:%M"), "14:35")


if __name__ == "__main__":
    unittest.main()
