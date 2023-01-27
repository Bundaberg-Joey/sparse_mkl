import pytest


def pytest_addoption(parser):
    """Adding extra flags to specify if pytest should run certain tests or not.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Adding bespoke marker for test decoration to prevent / enable certain tests to run.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Logic to be performed in the presence / abscence of bespoke flags / markers as defined.
    """
    if config.getoption("--runslow"):  # run slow tests so do not skip
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
