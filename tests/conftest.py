import pytest


@pytest.fixture(autouse=True)
def settings():
    from tests.settings import settings
    return settings
