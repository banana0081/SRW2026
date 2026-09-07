"""Notification helpers for a calibration run.

The previous version watched an arbitrary PID. When it was pointed at a shell
wrapper instead of the Python process, the wrapper's exit was reported as a
crash while the real run kept going and finished normally. PID watching is
therefore gone: status is derived from the run directory, and a run is only
declared finished once the supervisor has written a verdict it validated
against `summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import requests

CONDITION_DIRECTORIES = ("initial", "draft")
VERDICT_NAME = "supervisor_verdict.json"


def condition_directories(run_root: Path) -> tuple[str, ...]:
    """Prefer recorded condition names; fall back to known released pair."""
    queries = run_root / "queries.json"
    if queries.exists():
        try:
            plan = json.loads(queries.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            plan = []
        names: list[str] = []
        if isinstance(plan, list) and plan:
            for condition in plan[0].get("condition_order") or []:
                folder = str(condition).lower()
                if folder not in names:
                    names.append(folder)
        if names:
            return tuple(names)
    found = tuple(
        sorted(
            path.name
            for path in run_root.iterdir()
            if path.is_dir() and list(path.glob("ToolBench_G3_DFS_*.jsonl"))
        )
    )
    return found or CONDITION_DIRECTORIES


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ[key] = value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            break
    return rows


def record_count(path: Path) -> int:
    """Count released-format records, tolerating a truncated trailing one."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    count = 0
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        try:
            _, consumed = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            break
        count += 1
        offset += consumed
    return count


def planned_queries(run_root: Path) -> int:
    path = run_root / "queries.json"
    if not path.exists():
        return 0
    plan = json.loads(path.read_text(encoding="utf-8"))
    return len(plan) if isinstance(plan, list) else 0


def cost_usd(run_root: Path) -> float:
    total = 0.0
    for name in ("usage_agent.jsonl", "usage_simulator.jsonl"):
        total += sum(
            float(row.get("cost_usd") or 0.0)
            for row in _read_jsonl(run_root / name)
        )
    return round(total, 6)


def snapshot(run_root: Path) -> dict[str, object]:
    total_planned = planned_queries(run_root)
    conditions: dict[str, object] = {}
    total_done = 0
    names = condition_directories(run_root)
    for name in names:
        directory = run_root / name
        files = sorted(directory.glob("ToolBench_G3_DFS_*.jsonl"))
        done = record_count(files[0]) if files else 0
        conditions[name] = {"done": done, "planned": total_planned}
        total_done += done
    agent = _read_jsonl(run_root / "usage_agent.jsonl")
    simulator = _read_jsonl(run_root / "usage_simulator.jsonl")
    return {
        "run_root": str(run_root),
        "conditions": conditions,
        "total_done": total_done,
        "total_planned": total_planned * max(1, len(conditions)),
        "agent_requests": len(agent),
        "simulator_requests": len(simulator),
        "total_cost_usd": cost_usd(run_root),
        "summary_exists": (run_root / "summary.json").exists(),
        "verdict_exists": (run_root / VERDICT_NAME).exists(),
    }


def format_status(title: str, data: dict[str, object], extra: str = "") -> str:
    conditions = data["conditions"]
    lines = [title]
    for name, payload in conditions.items():
        planned = payload.get("planned") or "?"
        lines.append(f"{name}: {payload.get('done')}/{planned}")
    lines.extend(
        [
            f"Agent requests: {data['agent_requests']}",
            f"Simulator requests: {data['simulator_requests']}",
            f"Cost: ${data['total_cost_usd']}",
        ]
    )
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def send_telegram(text: str) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return "telegram skipped: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.ok:
            return "telegram sent"
        return f"telegram failed: HTTP {response.status_code} {response.text[:200]}"
    except requests.RequestException as exc:
        return f"telegram failed: {exc}"


def send_ntfy(topic: str, title: str, text: str) -> str:
    if not topic:
        return "ntfy skipped"
    try:
        request = Request(
            f"https://ntfy.sh/{topic}",
            data=text.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Tags": "microscope",
            },
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            response.read()
        return f"ntfy sent to https://ntfy.sh/{topic}"
    except Exception as exc:  # noqa: BLE001
        return f"ntfy failed: {exc}"


def send_windows_toast(title: str, text: str) -> str:
    escaped = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        "$n.Visible = $true; "
        f"$n.ShowBalloonTip(8000, '{title}', '{escaped}', "
        "[System.Windows.Forms.ToolTipIcon]::Info); "
        "Start-Sleep -Seconds 8; $n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "windows toast shown"
    except OSError as exc:
        return f"windows toast failed: {exc}"


def notify(env_path: Path, title: str, text: str, ntfy_topic: str) -> list[str]:
    load_env(env_path)
    return [
        send_telegram(f"{title}\n{text}"),
        send_ntfy(ntfy_topic, title, text),
        send_windows_toast(title, text.replace("\n", " | ")),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report calibration-run status from the run directory."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/results/g3_calibration_v2"),
    )
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--ntfy-topic", default="nir-tooldoc-calibration")
    parser.add_argument(
        "--mode",
        choices=["notify-only", "await-verdict"],
        default="notify-only",
        help=(
            "notify-only reports the current directory state; await-verdict "
            "waits for the supervisor's validated verdict before notifying."
        ),
    )
    parser.add_argument("--title", default="NIR: calibration status")
    parser.add_argument("--text", default="")
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "notify-only":
        text = args.text.strip() or format_status(
            args.title, snapshot(args.run_root)
        )
        results = notify(args.env_path, args.title, text, args.ntfy_topic)
        print(text)
        print("\n".join(results))
        return 0

    verdict_path = args.run_root / VERDICT_NAME
    log_path = args.run_root / "watcher.log"
    while not verdict_path.exists():
        data = snapshot(args.run_root)
        log_path.write_text(
            format_status("calibration still running", data) + "\n",
            encoding="utf-8",
        )
        time.sleep(args.poll_seconds)

    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    title = f"NIR: calibration {verdict.get('status', 'unknown')}"
    text = format_status(
        title,
        snapshot(args.run_root),
        str(verdict.get("detail") or ""),
    )
    results = notify(args.env_path, title, text, args.ntfy_topic)
    log_path.write_text(
        text + "\n" + "\n".join(results) + "\n", encoding="utf-8"
    )
    print(text)
    print("\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
