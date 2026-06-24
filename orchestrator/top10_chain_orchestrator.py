#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential GitHub Actions orchestrator for the A-share Top10 chain.

The script is intentionally external to the business systems. It dispatches the
existing workflows, waits for success, verifies published artifacts, and then
moves to the next repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

OWNER = "njedu2023-prog"
BRANCH = "main"
API = "https://api.github.com"
TZ = ZoneInfo("Asia/Shanghai")

DATA_REPO = f"{OWNER}/a-share-top3-data"
TOP10_REPO = f"{OWNER}/a-top10"
DECISION_REPO = f"{OWNER}/top10-decision"

DATA_WF = "daily_fetch.yml"
TOP10_WF = "run_top10.yml"
DECISION_WF = "run_decision_daily.yml"
PREMIUM_WF = "run_premium.yml"

A_TOP10_HOME = "https://njedu2023-prog.github.io/a-top10/"
DECISION_HOME = "https://njedu2023-prog.github.io/top10-decision/decision.html"
PREMIUM_HOME = "https://njedu2023-prog.github.io/top10-decision/docs/reports/premium_latest.html"


class ChainError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def request(url: str, *, method: str = "GET", token: str = "", payload: Optional[Dict[str, Any]] = None, raw: bool = False, timeout: int = 60) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "top10-chain-orchestrator"}
    if url.startswith(API):
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if raw:
            return body
        if not body:
            return {}
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype or url.startswith(API):
            return json.loads(body.decode("utf-8"))
        return body.decode("utf-8", errors="replace")


def gh(path: str, token: str) -> Any:
    return request(API + path, token=token)


def gh_post(path: str, token: str, payload: Dict[str, Any]) -> Any:
    return request(API + path, method="POST", token=token, payload=payload)


def raw_url(repo: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{path}"


def page_url(repo_slug: str, path: str) -> str:
    return f"https://njedu2023-prog.github.io/{repo_slug}/{path}"


def get_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "top10-chain-orchestrator"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def url_ok(url: str, timeout: int = 30) -> bool:
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": "top10-chain-orchestrator"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= int(resp.status) < 300
        except Exception:
            pass
    return False


def wait_url(url: str, label: str, timeout_s: int = 900, contains: str = "") -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            if contains:
                text = get_text(url)
                if contains in text:
                    log(f"[publish] {label}: OK {url}")
                    return
                last = f"missing text {contains!r}"
            elif url_ok(url):
                log(f"[publish] {label}: OK {url}")
                return
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(20)
    raise ChainError(f"publication timeout: {label} {url} {last}")


def tushare_trade_day(trade_date: str, token: str) -> Optional[bool]:
    if not token:
        return None
    payload = {
        "api_name": "trade_cal",
        "token": token,
        "params": {"exchange": "SSE", "start_date": trade_date, "end_date": trade_date},
        "fields": "cal_date,is_open",
    }
    try:
        res = request("https://api.tushare.pro", method="POST", payload=payload, timeout=30)
        fields = res.get("data", {}).get("fields", [])
        rows = res.get("data", {}).get("items", [])
        if not rows or "is_open" not in fields:
            return None
        return str(rows[0][fields.index("is_open")]) == "1"
    except Exception as exc:
        log(f"[calendar] tushare unavailable: {type(exc).__name__}: {exc}")
        return None


def snapshot_exists(trade_date: str) -> bool:
    return url_ok(raw_url(DATA_REPO, f"data/raw/{trade_date[:4]}/{trade_date}/_meta.json"))


def is_trade_day(trade_date: str, tushare_token: str) -> bool:
    d = dt.datetime.strptime(trade_date, "%Y%m%d").date()
    if d.weekday() >= 5:
        log(f"[calendar] {trade_date} is weekend; skip")
        return False
    cal = tushare_trade_day(trade_date, tushare_token)
    if cal is not None:
        log(f"[calendar] SSE is_open={cal} date={trade_date}")
        return cal
    if snapshot_exists(trade_date):
        log(f"[calendar] fallback snapshot exists; date={trade_date}")
        return True
    log(f"[calendar] no positive trading-day evidence; skip date={trade_date}")
    return False


def workflow_runs(repo: str, workflow: str, token: str, per_page: int = 30) -> List[Dict[str, Any]]:
    wf = urllib.parse.quote(workflow)
    data = gh(f"/repos/{repo}/actions/workflows/{wf}/runs?branch={BRANCH}&per_page={per_page}", token)
    return data.get("workflow_runs", [])


def active_run(repo: str, workflow: str, token: str, since: dt.datetime) -> Optional[Dict[str, Any]]:
    for run in workflow_runs(repo, workflow, token):
        if parse_time(run["created_at"]) >= since and run["status"] in {"queued", "in_progress", "waiting", "pending", "requested"}:
            return run
    return None


def recent_success(repo: str, workflow: str, token: str, since: dt.datetime) -> Optional[Dict[str, Any]]:
    for run in workflow_runs(repo, workflow, token, 20):
        if parse_time(run["created_at"]) >= since and run["status"] == "completed" and run.get("conclusion") == "success":
            return run
    return None


def dispatch(repo: str, workflow: str, token: str, inputs: Dict[str, str]) -> None:
    payload: Dict[str, Any] = {"ref": BRANCH}
    if inputs:
        payload["inputs"] = inputs
    gh_post(f"/repos/{repo}/actions/workflows/{urllib.parse.quote(workflow)}/dispatches", token, payload)


def trigger_or_attach(repo: str, workflow: str, token: str, inputs: Dict[str, str], label: str, since: dt.datetime) -> Dict[str, Any]:
    run = active_run(repo, workflow, token, since)
    if run:
        log(f"[{label}] attach active run #{run['run_number']} {run['html_url']}")
        return run
    log(f"[{label}] dispatch {repo}/{workflow} inputs={inputs}")
    dispatch(repo, workflow, token, inputs)
    deadline = time.time() + 180
    while time.time() < deadline:
        run = active_run(repo, workflow, token, since) or recent_success(repo, workflow, token, since)
        if run:
            log(f"[{label}] run found #{run['run_number']} {run['status']} {run.get('conclusion')} {run['html_url']}")
            return run
        time.sleep(6)
    raise ChainError(f"{label}: dispatch succeeded but no run found")


def failed_log_excerpt(repo: str, run_id: int, token: str) -> str:
    parts: List[str] = []
    try:
        jobs = gh(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", token).get("jobs", [])
        for job in jobs:
            if job.get("conclusion") == "failure":
                parts.append(f"job={job.get('name')} id={job.get('id')}")
                for step in job.get("steps", []):
                    if step.get("conclusion") == "failure":
                        parts.append(f"failed_step={step.get('number')} {step.get('name')}")
    except Exception as exc:
        parts.append(f"jobs_read_failed={type(exc).__name__}:{exc}")
    try:
        raw = request(f"{API}/repos/{repo}/actions/runs/{run_id}/logs", token=token, raw=True, timeout=120)
        zf = zipfile.ZipFile(io.BytesIO(raw))
        pat = re.compile(r"(Traceback|Error:|Exception|FAILED|failed|fatal:|exit code|NameError|TypeError|ValueError)", re.I)
        for name in zf.namelist():
            lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines):
                if pat.search(line):
                    block = "\n".join(lines[max(0, i - 4): min(len(lines), i + 10)])
                    parts.append(f"log={name}\n{block}")
                    if len(parts) >= 8:
                        return "\n".join(parts)
                    break
    except Exception as exc:
        parts.append(f"logs_read_failed={type(exc).__name__}:{exc}")
    return "\n".join(parts)


def wait_run(repo: str, workflow: str, token: str, run: Dict[str, Any], label: str, timeout_s: int = 3600) -> Dict[str, Any]:
    run_id = int(run["id"])
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        cur = gh(f"/repos/{repo}/actions/runs/{run_id}", token)
        state = f"{cur.get('status')}/{cur.get('conclusion')}"
        if state != last:
            log(f"[{label}] #{cur.get('run_number')} {state} {cur.get('html_url')}")
            last = state
        if cur.get("status") == "completed":
            if cur.get("conclusion") != "success":
                raise ChainError(f"{label} failed: {cur.get('conclusion')} {cur.get('html_url')}\n{failed_log_excerpt(repo, run_id, token)}")
            return cur
        time.sleep(20)
    raise ChainError(f"{label} timeout: #{run.get('run_number')} {run.get('html_url')}")


def wait_pages(repo: str, token: str, since: dt.datetime, label: str, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        runs = gh(f"/repos/{repo}/actions/runs?per_page=30", token).get("workflow_runs", [])
        for run in runs:
            if run.get("name") == "pages build and deployment" and parse_time(run["created_at"]) >= since:
                wait_run(repo, "pages-build-deployment", token, run, label, timeout_s)
                return
        time.sleep(10)
    log(f"[{label}] no Pages run observed; direct URL checks will decide")


def wait_premium(token: str, since: dt.datetime, timeout_s: int = 3000) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for run in workflow_runs(DECISION_REPO, PREMIUM_WF, token, 20):
            if parse_time(run["created_at"]) >= since:
                return wait_run(DECISION_REPO, PREMIUM_WF, token, run, "premium", timeout_s)
        time.sleep(15)
    log("[premium] no run observed after decision; continue")
    return None


def decision_report_for_signal(trade_date: str) -> Optional[str]:
    data = request(f"{API}/repos/{DECISION_REPO}/contents/outputs/decision?ref={BRANCH}")
    paths = sorted([x["path"] for x in data if re.search(r"decision_report_\d{8}\.md$", x.get("path", ""))], reverse=True)
    for path in paths[:12]:
        url = page_url("top10-decision", path)
        try:
            text = get_text(url)
        except Exception:
            continue
        if f"signal_date: **{trade_date}**" in text or f"trade_date: **{trade_date}**" in text:
            return url
    return None


def append_summary(lines: List[str]) -> None:
    text = "\n".join(lines) + "\n"
    print(text)
    path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-date", default="")
    ap.add_argument("--skip-calendar", action="store_true")
    args = ap.parse_args()

    token = os.getenv("ORCHESTRATOR_TOKEN", "").strip()
    if not token:
        raise ChainError("missing ORCHESTRATOR_TOKEN secret")
    tushare_token = os.getenv("TUSHARE_TOKEN", "").strip()
    trade_date = args.trade_date.strip() or dt.datetime.now(TZ).strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", trade_date):
        raise ChainError(f"invalid trade_date={trade_date!r}")

    started = now_utc()
    summary = ["# A-share Top10 Chain Orchestrator", "", f"- trade_date: `{trade_date}`", f"- started_at_utc: `{iso(started)}`"]

    if not args.skip_calendar and not is_trade_day(trade_date, tushare_token):
        summary.append("- status: `skipped_non_trading_day`")
        append_summary(summary)
        return 0

    data_run = trigger_or_attach(DATA_REPO, DATA_WF, token, {"trade_date": trade_date}, "data", started)
    data_done = wait_run(DATA_REPO, DATA_WF, token, data_run, "data", 3600)
    summary.append(f"- data: `success` [#{data_done['run_number']}]({data_done['html_url']})")
    y = trade_date[:4]
    wait_url(raw_url(DATA_REPO, f"data/raw/{y}/{trade_date}/_meta.json"), "data raw meta", 600)
    wait_url(raw_url(DATA_REPO, f"data/raw/{y}/{trade_date}/daily.csv"), "data raw daily", 600)
    wait_url(raw_url(DATA_REPO, "data/latest/_meta.json"), "data latest meta", 600, contains=trade_date)

    top10_start = now_utc()
    top10_run = trigger_or_attach(TOP10_REPO, TOP10_WF, token, {"trade_date": trade_date}, "a-top10", top10_start)
    top10_done = wait_run(TOP10_REPO, TOP10_WF, token, top10_run, "a-top10", 3600)
    summary.append(f"- a-top10: `success` [#{top10_done['run_number']}]({top10_done['html_url']})")
    wait_pages(TOP10_REPO, token, top10_start, "a-top10 pages", 900)
    wait_url(A_TOP10_HOME, "a-top10 home", 600)
    wait_url(page_url("a-top10", f"outputs/predict_top10_{trade_date}.md"), "a-top10 top10 md", 900)
    wait_url(raw_url(TOP10_REPO, f"outputs/decisio/pred_decisio_{trade_date}.csv"), "a-top10 pred csv", 900)
    summary.append(f"- a-top10_report: {A_TOP10_HOME}")

    decision_start = now_utc()
    decision_run = trigger_or_attach(DECISION_REPO, DECISION_WF, token, {"trade_date": trade_date}, "decision", decision_start)
    decision_done = wait_run(DECISION_REPO, DECISION_WF, token, decision_run, "decision", 4200)
    summary.append(f"- decision: `success` [#{decision_done['run_number']}]({decision_done['html_url']})")
    premium_done = wait_premium(token, decision_start, 3000)
    if premium_done:
        summary.append(f"- premium: `success` [#{premium_done['run_number']}]({premium_done['html_url']})")
    wait_pages(DECISION_REPO, token, decision_start, "top10-decision pages", 1200)
    wait_url(DECISION_HOME, "decision dashboard", 900)
    report = decision_report_for_signal(trade_date)
    if not report:
        raise ChainError(f"cannot find decision report for signal_date={trade_date}")
    wait_url(PREMIUM_HOME, "premium latest", 900)
    summary.extend([
        f"- decision_report: {DECISION_HOME}",
        f"- decision_report_md: {report}",
        f"- premium_report: {PREMIUM_HOME}",
        "- status: `success`",
        f"- finished_at_utc: `{iso(now_utc())}`",
    ])
    append_summary(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChainError as exc:
        log(f"ERROR: {exc}")
        append_summary(["# A-share Top10 Chain Orchestrator", "", "- status: `failed`", "", "```", str(exc), "```"])
        raise SystemExit(1)
