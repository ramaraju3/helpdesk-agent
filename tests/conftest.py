import pytest


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Give every test its own fresh copy of the fixture database."""
    path = tmp_path / "helpdesk.db"
    monkeypatch.setenv("HELPDESK_DB", str(path))
    return path
