import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--show-plots", action="store_true", help="display plots (True/False)"
    )


@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show-plots")
