from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


CN_TZ = ZoneInfo("Asia/Shanghai")
MORNING_START = time(9, 25)
MORNING_END = time(11, 35)
AFTERNOON_START = time(12, 55)
AFTERNOON_END = time(15, 10)


@dataclass(frozen=True)
class DueDecision:
    should_run: bool
    reason: str
    target_slot: datetime | None = None
    last_generated_at: datetime | None = None


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def _slots(start: datetime, end: datetime, minutes: int) -> list[datetime]:
    result: list[datetime] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(minutes=minutes)
    return result


def scheduled_slots(day: date) -> list[datetime]:
    def at(value: time) -> datetime:
        return datetime.combine(day, value, CN_TZ)

    slots = [
        *_slots(at(MORNING_START), at(MORNING_END), 10),
        *_slots(at(AFTERNOON_START), at(time(14, 15)), 10),
        *_slots(at(time(14, 20)), at(time(14, 55)), 5),
        at(time(15, 5)),
        at(AFTERNOON_END),
    ]
    return sorted(set(slots))


def in_data_window(current: datetime) -> bool:
    local = current.astimezone(CN_TZ)
    return (
        MORNING_START <= local.time() <= MORNING_END
        or AFTERNOON_START <= local.time() <= AFTERNOON_END
    )


def latest_due_slot(current: datetime) -> datetime | None:
    local = current.astimezone(CN_TZ)
    if not in_data_window(local):
        return None
    due = [slot for slot in scheduled_slots(local.date()) if slot <= local]
    return due[-1] if due else None


def parse_datetime(value: Any) -> datetime | None:
    text_value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text_value[:19], fmt)
            return parsed.replace(tzinfo=CN_TZ)
        except ValueError:
            continue
    return None


def read_manifest(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def due_decision(current: datetime, manifest_path: str | Path) -> DueDecision:
    local = current.astimezone(CN_TZ)
    target = latest_due_slot(local)
    if target is None:
        return DueDecision(False, "outside A-share data window")

    manifest = read_manifest(manifest_path)
    generated_at = parse_datetime(manifest.get("generated_at"))
    source_trade_date = str(manifest.get("source_trade_date") or "")
    today = local.strftime("%Y%m%d")
    if source_trade_date != today:
        return DueDecision(True, f"source trade date {source_trade_date or 'missing'} != {today}", target, generated_at)
    if generated_at is None:
        return DueDecision(True, "manifest generated_at is missing", target, generated_at)
    if generated_at >= target:
        return DueDecision(False, "latest target slot already covered", target, generated_at)
    return DueDecision(True, "latest target slot is newer than manifest", target, generated_at)
