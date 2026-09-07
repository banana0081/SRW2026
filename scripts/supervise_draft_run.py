"""Own a calibration run end to end and report a validated verdict.

The supervisor spawns the runner as its own direct child, so the process it
watches is the Python process rather than a shell wrapper. It restarts only
after an infrastructure failure, and only when the previous attempt actually
advanced, so a permanent failure stops instead of looping. Notifications are
sent from the verdict, which is derived from `summary.json`, never from the
observation that some process exited.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from watch_run_notify import (  # noqa: E402
    VERDICT_NAME,
    format_status,
    notify,
    snapshot,
)

EXIT_COMPLETE = 0
EXIT_COST_CAP = 2
EXIT_INFRASTRUCTURE = 3
EXIT_PROVENANCE = 4

LOCK_NAME = "supervisor.lock"


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def process_alive(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return str(pid) in result.stdout and "No tasks" not in result.stdout


def acquire_lock(run_root: Path, log_path: Path) -> Path:
    lock = run_root / LOCK_NAME
    if lock.exists():
        try:
            holder = int(json.loads(lock.read_text(encoding="utf-8"))["pid"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            holder = -1
        if holder > 0 and process_alive(holder):
            raise SystemExit(
                f"{lock} is held by live pid {holder}. Refusing to run two "
                "supervisors against one output root."
            )
        log(log_path, f"clearing stale lock from pid {holder}")
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()}) + "\n",
        encoding="utf-8",
    )
    return lock


def summary_verdict(run_root: Path, cap_usd: float) -> tuple[bool, str]:
    """Accept a run only if the saved summary says it is clean and complete."""
    path = run_root / "summary.json"
    if not path.exists():
        return False, "summary.json is missing"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"summary.json is unreadable: {exc}"
    problems: list[str] = []
    planned = int(summary.get("planned_queries") or 0)
    if planned <= 0:
        problems.append("no planned queries")
    if summary.get("pending_positions"):
        problems.append(
            f"{len(summary['pending_positions'])} queries were never attempted"
        )
    for name, condition in (summary.get("conditions") or {}).items():
        records = int(condition.get("records") or 0)
        excluded = len(condition.get("excluded_queries") or {})
        if records + excluded != planned:
            problems.append(
                f"{name} has {records} records and {excluded} exclusions "
                f"for {planned} queries"
            )
        if int(condition.get("rolled_back_records") or 0):
            problems.append(
                f"{name} rolled back "
                f"{condition['rolled_back_records']} record(s)"
            )
    spent = float((summary.get("usage") or {}).get("cost_usd") or 0.0)
    if spent > cap_usd:
        problems.append(f"spent ${spent:.6f} above cap ${cap_usd:.2f}")
    stop_reason = str(summary.get("stop_reason") or "")
    if stop_reason != "completed":
        problems.append(f"stop_reason={stop_reason}")
    if problems:
        return False, "; ".join(problems)
    scored = int(summary.get("scored_queries") or 0)
    excluded = len(summary.get("excluded_positions") or [])
    return True, (
        f"{scored}/{planned} queries scored, {excluded} excluded, "
        f"spent ${spent:.6f} of ${cap_usd:.2f}"
    )


def build_command(args: argparse.Namespace, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tooldoc_nir.draft_agent_reproduction",
        "--start",
        str(args.start),
        "--limit",
        str(args.limit),
        "--retrieval-num",
        str(args.retrieval_num),
        "--seed",
        str(args.seed),
        "--agent-model",
        args.agent_model,
        "--simulator-model",
        args.simulator_model,
        "--simulator-mode",
        args.simulator_mode,
        "--max-cost-usd",
        str(args.max_cost_usd),
        "--query-attempts",
        str(args.query_attempts),
        "--max-driver-failure-rate",
        str(args.max_driver_failure_rate),
        "--conditions",
        *list(args.conditions),
        "--output-root",
        str(args.run_root),
    ]
    if args.agent_provider:
        command += ["--agent-provider", args.agent_provider]
    extra = getattr(args, "extra_documentation", None) or []
    for item in extra:
        command += ["--extra-documentation", item]
    if getattr(args, "schema_gate", False):
        command.append("--schema-gate")
    backend = getattr(args, "backend", "stable")
    if backend and backend != "stable":
        command += ["--backend", backend]
    if not resume:
        command.append("--no-resume")
    return command


def write_verdict(
    run_root: Path,
    *,
    status: str,
    detail: str,
    attempts: int,
    exit_code: int,
) -> dict[str, object]:
    verdict = {
        "status": status,
        "detail": detail,
        "attempts": attempts,
        "runner_exit_code": exit_code,
        "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshot": snapshot(run_root),
    }
    (run_root / VERDICT_NAME).write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate one paired Raw/DRAFT calibration."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/results/g3_calibration_v2"),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--retrieval-num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agent-model", default="openai/gpt-4o-mini-2024-07-18"
    )
    parser.add_argument("--agent-provider", default=None)
    parser.add_argument(
        "--simulator-model", default="openai/gpt-4o-mini-2024-07-18"
    )
    parser.add_argument(
        "--simulator-mode", choices=["simulate", "replay"], default="simulate"
    )
    parser.add_argument("--max-cost-usd", type=float, default=0.15)
    parser.add_argument("--query-attempts", type=int, default=2)
    parser.add_argument(
        "--max-driver-failure-rate",
        type=float,
        default=0.15,
        help=(
            "Forwarded to the runner. DeepSeek Flash fails JSON more often "
            "than GPT-4o-mini; raise this so a noisy prefix does not abort "
            "a resume of an otherwise valid run."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Initial", "DRAFT"],
    )
    parser.add_argument(
        "--extra-documentation",
        action="append",
        default=[],
        help="Condition=path pairs, e.g. Ours=artifacts/documentation/Ours.json",
    )
    parser.add_argument(
        "--schema-gate",
        action="store_true",
        help="Reject ToolEnv-invalid calls before cache or simulation.",
    )
    parser.add_argument(
        "--backend",
        choices=["stable", "live"],
        default="stable",
        help=(
            "stable uses the ToolEnv cache and simulator. live calls the "
            "subscribed RapidAPI hosts."
        ),
    )
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=20.0)
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume", default=True
    )
    parser.add_argument("--env-path", type=Path, default=Path(".env"))
    parser.add_argument("--ntfy-topic", default="nir-tooldoc-calibration")
    parser.add_argument(
        "--no-notify", action="store_false", dest="notify", default=True
    )
    args = parser.parse_args()

    run_root: Path = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    supervisor_log = run_root / "supervisor.log"
    child_log = run_root / "supervisor_child.log"
    (run_root / VERDICT_NAME).unlink(missing_ok=True)
    lock = acquire_lock(run_root, supervisor_log)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    status = "incomplete"
    detail = "the runner never started"
    exit_code = -1
    attempt = 0
    try:
        previous_progress = -1
        for attempt in range(1, args.max_restarts + 1):
            before = snapshot(run_root)
            log(supervisor_log, f"attempt {attempt} state={before}")
            # Only the first attempt may discard existing output; a restart
            # after an infrastructure failure must resume the same run.
            command = build_command(args, resume=args.resume or attempt > 1)
            with child_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n===== attempt {attempt} at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
                )
                handle.flush()
                child = subprocess.Popen(
                    command,
                    cwd=str(Path.cwd()),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                log(supervisor_log, f"runner pid={child.pid} command={command}")
                exit_code = child.wait()
            after = snapshot(run_root)
            log(
                supervisor_log,
                f"runner pid={child.pid} exit={exit_code} state={after}",
            )

            if exit_code == EXIT_COMPLETE:
                accepted, note = summary_verdict(run_root, args.max_cost_usd)
                status = "complete" if accepted else "incomplete"
                detail = note
                break
            if exit_code == EXIT_COST_CAP:
                status = "cost_cap"
                detail = summary_verdict(run_root, args.max_cost_usd)[1]
                break
            if exit_code == EXIT_PROVENANCE:
                status = "provenance_mismatch"
                detail = "the manifest no longer matches this output root"
                break

            done = int(after["total_done"])
            if exit_code != EXIT_INFRASTRUCTURE:
                status = "runner_error"
                detail = f"unexpected exit code {exit_code}"
                break
            if done <= previous_progress:
                status = "infrastructure"
                detail = (
                    "the runner failed twice without advancing; not restarting"
                )
                break
            previous_progress = done
            if attempt >= args.max_restarts:
                status = "infrastructure"
                detail = f"gave up after {attempt} attempts"
                break
            log(
                supervisor_log,
                f"infrastructure failure, retrying in {args.sleep_seconds}s",
            )
            time.sleep(args.sleep_seconds)
    finally:
        lock.unlink(missing_ok=True)

    verdict = write_verdict(
        run_root,
        status=status,
        detail=detail,
        attempts=attempt,
        exit_code=exit_code,
    )
    title = f"NIR: calibration {status}"
    text = format_status(title, snapshot(run_root), detail)
    log(supervisor_log, f"verdict {status}: {detail}")
    if args.notify:
        for line in notify(args.env_path, title, text, args.ntfy_topic):
            log(supervisor_log, line)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
