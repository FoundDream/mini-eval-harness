from __future__ import annotations

import argparse
from pathlib import Path

from mini_eval_harness.agent.replay import find_record, render_replay
from mini_eval_harness.runner import main as run_eval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mini-eval",
        description="Run and inspect mini eval harness experiments.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an eval config.")
    run_parser.add_argument("config", nargs="?", help="Path to a YAML config.")
    run_parser.add_argument("--config", dest="config_flag", help="Path to a YAML config.")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay an agent eval trajectory.",
    )
    replay_parser.add_argument("results", nargs="?", help="Result JSONL path.")
    replay_parser.add_argument("task_id", nargs="?", help="Task/sample id.")
    replay_parser.add_argument("--results", dest="results_flag", help="Result JSONL path.")
    replay_parser.add_argument("--task-id", dest="task_id_flag", help="Task/sample id.")

    configs_parser = subparsers.add_parser("configs", help="List local config files.")
    configs_parser.add_argument(
        "--dir",
        default="configs",
        help="Directory containing YAML config files.",
    )

    args, extra_args = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        forwarded_args: list[str] = []
        config_path = args.config_flag or args.config
        if config_path is not None:
            forwarded_args.extend(["--config", config_path])
        forwarded_args.extend(extra_args)
        run_eval(forwarded_args)
        return

    if args.command == "replay":
        if extra_args:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        results_path = args.results_flag or args.results
        task_id = args.task_id_flag or args.task_id
        if results_path is None:
            replay_parser.error("results path is required")
        if task_id is None:
            replay_parser.error("task_id is required")
        record = find_record(Path(results_path), task_id)
        print(render_replay(record))
        return

    if args.command == "configs":
        if extra_args:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
        config_dir = Path(args.dir)
        if not config_dir.exists():
            raise SystemExit(f"Config directory not found: {config_dir}")
        for path in sorted(config_dir.glob("*.yaml")):
            print(path)
        return

    parser.error(f"Unsupported command: {args.command}")
