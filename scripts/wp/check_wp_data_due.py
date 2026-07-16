from __future__ import annotations

import json
import os
from pathlib import Path

from wp_schedule import due_decision, now_cn


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "wp" / "latest" / "wp_manifest.json"


def write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main() -> None:
    mode = os.environ.get("WP_DATA_RUN_MODE", "due").strip().lower()
    if mode in {"once", "session"}:
        should_run = True
        reason = f"explicit {mode} mode"
        target_slot = ""
    else:
        decision = due_decision(now_cn(), MANIFEST_PATH)
        should_run = decision.should_run
        reason = decision.reason
        target_slot = decision.target_slot.strftime("%Y-%m-%d %H:%M:%S") if decision.target_slot else ""

    write_output("should_run", str(should_run).lower())
    write_output("target_slot", target_slot)
    write_output("reason", reason)
    print(
        json.dumps(
            {
                "mode": mode,
                "should_run": should_run,
                "target_slot": target_slot,
                "reason": reason,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
