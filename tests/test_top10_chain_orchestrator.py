import datetime as dt

import pytest

from orchestrator.top10_chain_orchestrator import (
    ChainError,
    WorkflowRunError,
    chain_outputs_status,
    decision_dashboard_data_status,
    followup_window_start,
    resolve_trade_date,
    wait_decision_followups,
    wait_followup_workflow,
)


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


def _run(run_id, workflow, *, event="workflow_run", created_at="2026-09-07T14:42:05Z"):
    return {
        "id": run_id,
        "run_number": run_id,
        "event": event,
        "created_at": created_at,
        "html_url": f"https://example.invalid/{workflow}/{run_id}",
    }


def test_followup_window_starts_near_decision_completion():
    run = {
        "created_at": "2026-09-07T13:00:00Z",
        "updated_at": "2026-09-07T14:42:00Z",
    }
    assert followup_window_start(run) == dt.datetime(
        2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc
    )


def test_wait_followup_limits_event_and_completion_window(monkeypatch):
    before_completion = _run(
        101, "premium", created_at="2026-09-07T14:40:00Z"
    )
    scheduled = _run(
        102, "premium", event="schedule", created_at="2026-09-07T14:43:00Z"
    )
    expected = _run(103, "premium", created_at="2026-09-07T14:42:05Z")
    observed = []

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.workflow_runs",
        lambda *args, **kwargs: [scheduled, before_completion, expected],
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.wait_run",
        lambda repo, workflow, token, run, label, timeout_s: observed.append(run) or run,
    )

    result = wait_followup_workflow(
        "token",
        dt.datetime(2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc),
        "run_premium.yml",
        "premium",
        timeout_s=30,
    )

    assert result == expected
    assert observed == [expected]


def test_cancelled_initial_followup_bubbles_without_polling(monkeypatch):
    cancelled = _run(101, "premium")
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.workflow_runs",
        lambda *args, **kwargs: [cancelled],
    )

    def cancelled_wait(repo, workflow, token, run, label, timeout_s):
        raise WorkflowRunError(label, "cancelled", run)

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.wait_run", cancelled_wait
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.time.sleep",
        lambda seconds: pytest.fail("cancelled run must not enter the polling loop again"),
    )

    with pytest.raises(WorkflowRunError, match="cancelled"):
        wait_followup_workflow(
            "token",
            dt.datetime(2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc),
            "run_premium.yml",
            "premium",
            timeout_s=30,
        )


@pytest.mark.parametrize(
    ("loser", "expected_workflow", "expected_inputs"),
    [
        ("premium", "run_premium.yml", {"cmd": "predict"}),
        ("decision_v12", "run_auction_v3.yml", {"signal_date": "20260907"}),
    ],
)
def test_followups_wait_for_survivor_then_recover_either_loser_once(
    monkeypatch, loser, expected_workflow, expected_inputs
):
    completed = {
        "premium": _run(201, "premium"),
        "decision_v12": _run(202, "v12"),
    }
    recovered = _run(301, loser, event="workflow_dispatch")
    order = []

    def fake_wait(token, since, workflow, label, timeout_s):
        key = "decision_v12" if workflow == "run_auction_v3.yml" else "premium"
        order.append(f"wait:{key}")
        if key == loser:
            raise WorkflowRunError(label, "cancelled", _run(100, key))
        return completed[key]

    def fake_execute(repo, workflow, token, inputs, label, timeout_s, **kwargs):
        order.append(f"recover:{loser}")
        assert workflow == expected_workflow
        assert inputs == expected_inputs
        assert kwargs["retry_count"] == 0
        return recovered, dt.datetime(2026, 9, 7, 14, 50, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.wait_followup_workflow", fake_wait
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.execute_workflow", fake_execute
    )

    premium, v12 = wait_decision_followups(
        "token",
        dt.datetime(2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc),
        "20260907",
        timeout_s=30,
    )

    assert order == [
        "wait:decision_v12",
        "wait:premium",
        f"recover:{loser}",
    ]
    assert (premium if loser == "premium" else v12) == recovered


def test_second_cancel_after_recovery_fails_fast(monkeypatch):
    calls = []

    def fake_wait(token, since, workflow, label, timeout_s):
        if workflow == "run_premium.yml":
            raise WorkflowRunError(label, "cancelled", _run(101, "premium"))
        return _run(202, "v12")

    def fake_execute(repo, workflow, token, inputs, label, timeout_s, **kwargs):
        calls.append(workflow)
        raise WorkflowRunError(label, "cancelled", _run(102, "premium-retry"))

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.wait_followup_workflow", fake_wait
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.execute_workflow", fake_execute
    )

    with pytest.raises(WorkflowRunError, match="cancelled"):
        wait_decision_followups(
            "token",
            dt.datetime(2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc),
            "20260907",
            timeout_s=30,
        )
    assert calls == ["run_premium.yml"]


def test_two_displaced_followups_fail_without_dispatch(monkeypatch):
    def cancelled_wait(token, since, workflow, label, timeout_s):
        raise WorkflowRunError(label, "cancelled", _run(101, workflow))

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.wait_followup_workflow",
        cancelled_wait,
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.execute_workflow",
        lambda *args, **kwargs: pytest.fail("ambiguous double recovery must not dispatch"),
    )

    with pytest.raises(ChainError, match="multiple Decision follow-ups"):
        wait_decision_followups(
            "token",
            dt.datetime(2026, 9, 7, 14, 41, 30, tzinfo=dt.timezone.utc),
            "20260907",
            timeout_s=30,
        )


def test_dashboard_data_rejects_stale_action_plan_signal_date(monkeypatch):
    def fake_get_text(url):
        if url.endswith("report_index.json"):
            return '{"latest_report_date": "20260908"}'
        return '{"signal_date": "20260904", "exec_date": "20260908"}'

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.get_text", fake_get_text
    )

    ok, detail = decision_dashboard_data_status("20260907")
    assert not ok
    assert detail == "action plan signal_date=20260904; expected 20260907"


def test_dashboard_data_rejects_report_index_exec_date_mismatch(monkeypatch):
    def fake_get_text(url):
        if url.endswith("report_index.json"):
            return '{"latest_report_date": "20260907"}'
        return '{"signal_date": "20260907", "exec_date": "20260908"}'

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.get_text", fake_get_text
    )

    ok, detail = decision_dashboard_data_status("20260907")
    assert not ok
    assert detail == (
        "report index latest_report_date=20260907; expected exec_date=20260908"
    )


def test_chain_outputs_accepts_dynamic_dashboard_data_contract(monkeypatch):
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.data_outputs_status",
        lambda trade_date: (True, "ok"),
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.url_ok", lambda url: True
    )
    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.decision_report_for_signal",
        lambda trade_date: "https://example.invalid/decision-report",
    )

    def fake_get_text(url):
        if url.endswith("report_index.json"):
            return '{"latest_report_date": "20260908"}'
        if url.endswith("action_plan_latest.json"):
            return '{"signal_date": "20260907", "exec_date": "20260908"}'
        return "current report 20260907"

    monkeypatch.setattr(
        "orchestrator.top10_chain_orchestrator.get_text", fake_get_text
    )

    ok, detail, report = chain_outputs_status("20260907")
    assert ok
    assert detail == "all chain outputs are already published"
    assert report == "https://example.invalid/decision-report"
