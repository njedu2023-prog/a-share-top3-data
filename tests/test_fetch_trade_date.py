from datetime import datetime

import pandas as pd
import pytest

from scripts.fetch_daily_snapshots import (
    FetchJob,
    build_jobs,
    default_trade_date,
    main as fetch_main,
    resolve_trade_date,
    select_intraday_symbols,
)


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


class FakePro:
    def limit_list_d(self, **kwargs):
        return pd.DataFrame()

    def daily(self, **kwargs):
        return pd.DataFrame()

    def stk_limit(self, **kwargs):
        return pd.DataFrame()

    def daily_basic(self, **kwargs):
        return pd.DataFrame()

    def stock_basic(self, **kwargs):
        return pd.DataFrame()

    def namechange(self, **kwargs):
        return pd.DataFrame()


def test_core_market_jobs_are_required_and_nonempty():
    jobs = {job.key: job for job in build_jobs(FakePro(), "20260901")}
    for key in ("daily", "daily_basic", "stk_limit"):
        assert jobs[key].required is True
        assert jobs[key].allow_empty is False
    assert jobs["limit_list_d"].required is False
    assert jobs["limit_list_d"].allow_empty is True


def test_required_failure_does_not_replace_latest(tmp_path, monkeypatch):
    latest = tmp_path / "data" / "latest"
    latest.mkdir(parents=True)
    (latest / "daily.csv").write_text(
        "ts_code,trade_date\nKEEP_DAILY,20260831\n", encoding="utf-8"
    )
    (latest / "stk_limit.csv").write_text(
        "ts_code,trade_date\nKEEP_LIMIT,20260831\n", encoding="utf-8"
    )
    (latest / "_meta.json").write_text('{"resolved_trade_date":"20260831"}\n', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in latest.iterdir()}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADE_DATE", "20260901")
    monkeypatch.setenv("MAX_RETRY", "1")
    monkeypatch.setenv("BASE_SLEEP_SEC", "0")
    monkeypatch.setenv("MAX_SLEEP_SEC", "0")
    monkeypatch.setenv("JITTER_SEC", "0")
    monkeypatch.setattr("scripts.fetch_daily_snapshots.get_pro", lambda: object())
    monkeypatch.setattr(
        "scripts.fetch_daily_snapshots.resolve_trade_date",
        lambda pro, requested: requested,
    )
    monkeypatch.setattr(
        "scripts.fetch_daily_snapshots.build_jobs",
        lambda pro, trade_date: [
            FetchJob(
                key="daily",
                fn=lambda **kwargs: pd.DataFrame(
                    [{"ts_code": "000001.SZ", "trade_date": trade_date}]
                ),
                kwargs={"trade_date": trade_date},
                columns=["ts_code", "trade_date"],
                allow_empty=False,
                required=True,
            ),
            FetchJob(
                key="stk_limit",
                fn=lambda **kwargs: pd.DataFrame(),
                kwargs={"trade_date": trade_date},
                columns=["ts_code", "trade_date"],
                allow_empty=False,
                required=True,
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="data/latest was not modified"):
        fetch_main()

    after = {path.name: path.read_bytes() for path in latest.iterdir()}
    assert after == before
    assert (tmp_path / "data" / "raw" / "2026" / "20260901" / "_meta.json").exists()


def test_intraday_selection_guarantees_limit_pool_beyond_configured_cap():
    required = [f"{600000 + i:06d}.SH" for i in range(93)]
    wp = [f"{300000 + i:06d}.SZ" for i in range(111)]

    selected, info = select_intraday_symbols(wp + required, required, max_symbols=80)

    assert selected == required + wp[:80]
    assert info == {
        "policy": "preserve_legacy_selection_plus_required_limit_list",
        "configured_cap": 80,
        "legacy_selected_count": 80,
        "required_count": 93,
        "required_selected": 93,
        "required_already_in_legacy": 0,
        "required_added_beyond_cap": 93,
        "preserved_nonrequired_count": 80,
        "selected_count": 173,
    }


def test_intraday_selection_fills_remaining_capacity_with_stable_optional_order():
    required = ["600001.SH", "600002.SH", "600003.SH"]
    candidates = [
        "300001.SZ",
        "600002.SH",
        "300002.SZ",
        "300001.SZ",
        "300003.SZ",
    ]

    selected, info = select_intraday_symbols(candidates, required, max_symbols=5)

    assert selected == required + ["300001.SZ", "300002.SZ", "300003.SZ"]
    assert info["required_selected"] == 3
    assert info["required_already_in_legacy"] == 1
    assert info["required_added_beyond_cap"] == 2
    assert info["preserved_nonrequired_count"] == 3
    assert info["selected_count"] == 6


def test_intraday_selection_preserves_legacy_cap_when_required_pool_is_empty():
    candidates = ["300001.SZ", "300002.SZ", "300001.SZ", "300003.SZ"]

    selected, info = select_intraday_symbols(candidates, [], max_symbols=2)

    assert selected == ["300001.SZ", "300002.SZ"]
    assert info["required_count"] == 0
    assert info["legacy_selected_count"] == 2
    assert info["preserved_nonrequired_count"] == 2
    assert info["selected_count"] == 2


def test_intraday_selection_keeps_unlimited_semantics_for_zero_cap():
    required = ["600001.SH", "600002.SH"]
    candidates = ["300001.SZ", "600001.SH", "300002.SZ"]

    selected, info = select_intraday_symbols(candidates, required, max_symbols=0)

    assert selected == required + ["300001.SZ", "300002.SZ"]
    assert info["legacy_selected_count"] == 3
    assert info["preserved_nonrequired_count"] == 2
    assert info["selected_count"] == 4
