"""CLI entry points for configuration, fixture smoke and production sandbox wiring."""

import argparse
import asyncio
import json
from dataclasses import asdict

from lagrl.config import Config, load_config


async def serve(config: Config) -> None:
    from lagrl.sandbox.gvisor import GVisorExecutor
    from lagrl.sandbox.lifecycle import SessionManager
    from lagrl.sandbox.transport import running_server

    backend = SessionManager(GVisorExecutor())
    try:
        async with running_server(backend, config.sandbox) as target:
            print(f"Sandbox transport at {target}; lifecycle T08/isolation T09 unfinished.")
            await asyncio.Event().wait()
    finally:
        await backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lagrl", description="Unfinished RL research scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="fixture wiring only; no learning or commands")
    smoke.add_argument("--backend", choices=["fake"], required=True)
    smoke.add_argument("--config", default="configs/cpu.yaml")
    check = commands.add_parser("config", help="validate and print configuration")
    check.add_argument("--config", default="configs/cpu.yaml")
    sandbox = commands.add_parser("sandbox", help="serve production wiring with TODO lifecycle")
    sandbox.add_argument("--config", default="configs/cpu.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "config":
        print(json.dumps(asdict(config), sort_keys=True))
    elif args.command == "smoke":
        from lagrl.testing.smoke import run_smoke

        asyncio.run(run_smoke(config))
    else:
        asyncio.run(serve(config))
    return 0
