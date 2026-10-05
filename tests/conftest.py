"""Explicit opt-in gates; unavailable dependencies never become skips or xfails."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-isolation", action="store_true", help="opt in to real-isolation tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if any(item.get_closest_marker("isolation") for item in items):
        if not config.getoption("--run-isolation"):
            raise pytest.UsageError("isolation exercises require explicit --run-isolation")
