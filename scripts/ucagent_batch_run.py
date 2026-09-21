#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import pty
import select

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "ucagent", "setting.yaml")
DEFAULT_RESULTS_LOG = os.path.join(REPO_ROOT, "ucagent_batch_results.txt")

DEFAULT_DUTS = ["IntegerDivider"]
DEFAULT_PHASES = [
    {
        "enable_all": "true",
        "enable_data_collection": "true",
        "enable_rerank": "true",
        "enable_structured_summary": "true",
        "enable_hierarchical_summary": "true",
        "enable_compact_test_output": "true",
        "enable_long_term_memory": "true",
        "enable_long_term_memory_embed": "true",
        "enable_failure_aware_context": "true",
        "enable_failure_taxonomy": "true",
        "enable_failure_online_metrics": "true",
    },
]


def parse_phase_kv(phase_str: str):
    overrides = {}
    if not phase_str:
        return overrides
    parts = [p.strip() for p in phase_str.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid override '{part}', expected key=value")
        k, v = part.split("=", 1)
        overrides[k.strip()] = v.strip()
    return overrides


def update_context_upgrade(config_path: str, overrides: dict):
    if not overrides:
        return
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_block = False
    block_indent = None
    updated = set()

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not in_block:
            if stripped.startswith("context_upgrade:"):
                in_block = True
                block_indent = len(line) - len(stripped)
            continue

        # inside block
        if stripped.strip() == "" or stripped.lstrip().startswith("#"):
            continue

        indent = len(line) - len(stripped)
        if indent <= block_indent:
            in_block = False
            continue

        # match key: value [#comment]
        if ":" not in stripped:
            continue
        key_part, rest = stripped.split(":", 1)
        key = key_part.strip()
        if key in overrides:
            # preserve indentation, inline comment, and optional YAML anchor
            comment = ""
            if "#" in rest:
                value_part, comment = rest.split("#", 1)
                comment = "#" + comment.rstrip("\n")
            else:
                value_part = rest
            value_part = value_part.strip()
            anchor = ""
            if value_part.startswith("&"):
                anchor_token = value_part.split()[0]
                anchor = anchor_token + " "
            new_value = overrides[key]
            new_line = " " * indent + f"{key}: {anchor}{new_value}"
            if comment:
                new_line += " " + comment
            new_line += "\n"
            lines[i] = new_line
            updated.add(key)

    missing = [k for k in overrides.keys() if k not in updated]
    if missing:
        raise RuntimeError(f"Keys not found under context_upgrade: {', '.join(missing)}")

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def load_ucagent_status(status_path: str):
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_completed(status_path: str, start_time: float):
    if not os.path.exists(status_path):
        return False
    try:
        mtime = os.path.getmtime(status_path)
    except OSError:
        return False
    if mtime < start_time:
        return False
    status = load_ucagent_status(status_path)
    if not status:
        return False
    return bool(status.get("all_completed", False))

def reset_status_file(status_path: str):
    if os.path.exists(status_path):
        try:
            os.remove(status_path)
        except Exception:
            pass


def clean_output(output_dir: str, keep_dirs=None):
    if keep_dirs is None:
        keep_dirs = {".qwen", ".ucagent_memory"}
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name in keep_dirs:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            subprocess.run(["rm", "-rf", path], check=False)
        else:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def get_conda_activate_cmd(env_name: str):
    try:
        conda_base = subprocess.check_output(
            ["bash", "-lc", "conda info --base"], text=True
        ).strip()
        conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        return f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(env_name)}"
    except Exception:
        return f"conda activate {shlex.quote(env_name)}"

def stop_process(proc, name: str):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


def run_make_test(dut: str, env_name: str, repo_root: str, make_args: str, status_path: str, timeout_seconds: int, complete_grace_seconds: int, max_no_generation_errors: int):
    conda_cmd = get_conda_activate_cmd(env_name)
    cmd = f"{conda_cmd} && make test_{shlex.quote(dut)} {make_args}".strip()
    shell_cmd = f"bash -lc {shlex.quote(cmd)}"

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        shell_cmd,
        shell=True,
        cwd=repo_root,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        preexec_fn=os.setsid,
    )
    os.close(slave_fd)

    exit_sent = False
    completed_at = None
    no_generation_errors = 0
    error_token = b"No generations found in stream."
    timed_out = False
    start_time = time.time()
    last_check = 0.0

    try:
        while True:
            if proc.poll() is not None:
                break
            if timeout_seconds > 0 and (time.time() - start_time) > timeout_seconds:
                timed_out = True
                break

            now = time.time()
            if now - last_check > 2.0:
                last_check = now
                if is_completed(status_path, start_time):
                    if completed_at is None:
                        completed_at = now
                    # send quit periodically to ensure TUI exits
                    for _ in range(2):
                        os.write(master_fd, b"q\n")
                        time.sleep(0.5)
                    exit_sent = True
                if completed_at is not None and complete_grace_seconds > 0:
                    if now - completed_at > complete_grace_seconds:
                        # force exit after grace period
                        break

            rlist, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in rlist:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                if max_no_generation_errors > 0 and error_token in data:
                    no_generation_errors += data.count(error_token)
                    if no_generation_errors >= max_no_generation_errors:
                        break

        # drain remaining output
        while True:
            rlist, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd not in rlist:
                break
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    if no_generation_errors >= max_no_generation_errors and max_no_generation_errors > 0:
        return 125
    if completed_at is not None and complete_grace_seconds > 0:
        # treat completed-but-stuck runs as success
        return 0
    if timed_out:
        return 124
    return proc.returncode


def append_result(log_path: str, line: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")

def load_success_runs(log_path: str):
    if not log_path or not os.path.exists(log_path):
        return set()
    success = set()
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "SUCCESS" not in line:
                    continue
                parts = line.strip().split()
                kv = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        kv[k] = v
                dut = kv.get("dut")
                phase = kv.get("phase")
                run = kv.get("run")
                if dut and phase and run:
                    try:
                        success.add((dut, int(phase), int(run)))
                    except ValueError:
                        continue
    except Exception:
        return set()
    return success

def migrate_results_log_if_needed(target_log: str):
    legacy_log = os.path.join(DEFAULT_OUTPUT_DIR, "ucagent_batch_results.txt")
    if os.path.exists(target_log):
        return
    if os.path.exists(legacy_log):
        os.makedirs(os.path.dirname(target_log), exist_ok=True)
        try:
            os.rename(legacy_log, target_log)
        except Exception:
            # fallback copy
            try:
                with open(legacy_log, "r", encoding="utf-8") as src, open(target_log, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                os.remove(legacy_log)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch run UCAgent for multiple DUTs with repeat runs and optional context_upgrade phases."
        )
    )
    parser.add_argument("duts", nargs="*", help="DUT names (default: Adder uart_tx IntegerDivider)")
    parser.add_argument("--repeat", type=int, default=3, help="Runs per DUT per phase (default: 3)")
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        help=(
            "Optional override phases: comma-separated key=value list. "
            "If omitted, uses built-in 4 phases."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Config file to edit (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--env",
        default="uc",
        help="Conda env name to activate (default: uc)",
    )
    parser.add_argument(
        "--make-args",
        default="",
        help="Extra args appended to make command (e.g., ARGS='--exit-on-completion')",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=2.0,
        help="Timeout per run in hours (default: 2)",
    )
    parser.add_argument(
        "--complete-grace-seconds",
        type=int,
        default=120,
        help="Grace period to wait after completion before force-exit (default: 120)",
    )
    parser.add_argument(
        "--max-no-generation-errors",
        type=int,
        default=10,
        help="Max 'No generations found in stream' errors before aborting attempt (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for a single run attempt (default: 3)",
    )
    parser.add_argument(
        "--results-log",
        default=DEFAULT_RESULTS_LOG,
        help=f"Results log path (default: {DEFAULT_RESULTS_LOG})",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue even if a run fails",
    )

    args = parser.parse_args()

    phases = [parse_phase_kv(p) for p in args.phase] if args.phase else DEFAULT_PHASES

    status_path = os.path.join(args.output_dir, ".ucagent_info.json")

    duts = args.duts if args.duts else DEFAULT_DUTS
    timeout_seconds = int(args.timeout_hours * 3600)
    migrate_results_log_if_needed(args.results_log)
    completed_runs = load_success_runs(args.results_log)

    for dut in duts:
        for phase_index, overrides in enumerate(phases, start=1):
            if overrides:
                update_context_upgrade(args.config, overrides)
                print(f"Applied context_upgrade overrides for phase {phase_index}: {overrides}")
            else:
                print(f"Phase {phase_index}: using existing context_upgrade settings")

            for run_idx in range(1, args.repeat + 1):
                if (dut, phase_index, run_idx) in completed_runs:
                    print(f"Skipping completed run: DUT {dut}, phase {phase_index}, run {run_idx}")
                    continue
                print(f"Running DUT {dut}, phase {phase_index}, run {run_idx}/{args.repeat}...")
                attempt = 1
                success = False
                while attempt <= args.max_retries:
                    reset_status_file(status_path)
                    rc = run_make_test(
                            dut=dut,
                            env_name=args.env,
                            repo_root=REPO_ROOT,
                            make_args=args.make_args,
                            status_path=status_path,
                            timeout_seconds=timeout_seconds,
                            complete_grace_seconds=args.complete_grace_seconds,
                            max_no_generation_errors=args.max_no_generation_errors,
                        )
                    print(f"Run finished with code {rc} (attempt {attempt}/{args.max_retries})")
                    clean_output(args.output_dir)
                    if rc == 0:
                        append_result(
                            args.results_log,
                            f"SUCCESS dut={dut} phase={phase_index} run={run_idx} attempt={attempt}",
                        )
                        success = True
                        break
                    if rc == 124:
                        append_result(
                            args.results_log,
                            f"TIMEOUT dut={dut} phase={phase_index} run={run_idx} attempt={attempt}",
                        )
                    elif rc == 125:
                        append_result(
                            args.results_log,
                            f"NOGEN dut={dut} phase={phase_index} run={run_idx} attempt={attempt}",
                        )
                    else:
                        append_result(
                            args.results_log,
                            f"FAIL dut={dut} phase={phase_index} run={run_idx} attempt={attempt} rc={rc}",
                        )
                    attempt += 1

                if not success:
                    append_result(
                        args.results_log,
                        f"SKIP dut={dut} phase={phase_index} run={run_idx} reason=max_retries",
                    )
                    if not args.keep_going:
                        print("Stopping due to repeated failures. Use --keep-going to continue.")
                        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
