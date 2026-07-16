from datetime import datetime

import pandas as pd

from scripts.fetch_daily_snapshots import default_trade_date, resolve_trade_date


class FakeCalendar:
    def trade_cal(self, **kwargs):
        return pd.DataFrame(
            [
                {"cal_date": "20260715", "is_open": 1},
                {"cal_date": "20260716", "is_open": 1},
                {"cal_date": "20260717", "is_open": 1},
            ]
        )


def test_default_trade_date_uses_previous_day_before_market_data_window():
    assert default_trade_date(datetime(2026, 7, 17, 5, 35)) == "20260716"
    assert default_trade_date(datetime(2026, 7, 17, 9, 24, 59)) == "20260716"


def test_default_trade_date_uses_today_from_0925():
    assert default_trade_date(datetime(2026, 7, 17, 9, 25)) == "20260717"


def test_explicit_trade_date_is_not_shifted_before_market():
    assert resolve_trade_date(FakeCalendar(), "20260717") == "20260717"
