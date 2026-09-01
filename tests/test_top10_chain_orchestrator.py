import datetime as dt

import pytest

from orchestrator.top10_chain_orchestrator import ChainError, resolve_trade_date


def test_explicit_trade_date_wins_over_schedule_slot():
    current = dt.datetime(2026, 9, 2, 1, 58)
    assert resolve_trade_date("20260831", "19:10", current) == ("20260831", "explicit")


def test_delayed_schedule_before_next_slot_binds_previous_weekday():
    current = dt.datetime(2026, 9, 1, 1, 58)
    assert resolve_trade_date("", "19:10", current) == ("20260831", "schedule_slot_19:10")


def test_delayed_friday_schedule_crossing_weekend_binds_friday():
    current = dt.datetime(2026, 9, 7, 1, 0)
    assert resolve_trade_date("", "19:10", current) == ("20260904", "schedule_slot_19:10")


def test_schedule_at_or_after_slot_binds_current_weekday():
    current = dt.datetime(2026, 9, 1, 19, 10)
    assert resolve_trade_date("", "19:10", current) == ("20260901", "schedule_slot_19:10")


def test_invalid_schedule_slot_is_rejected():
    with pytest.raises(ChainError, match="schedule-slot"):
        resolve_trade_date("", "25:10", dt.datetime(2026, 9, 1, 1, 0))
