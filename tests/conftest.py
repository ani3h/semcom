import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests that need network access"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow (downloads data)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
