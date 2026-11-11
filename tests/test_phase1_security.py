"""Small smoke tests for Optivolt Phase 1 hardening."""
from pathlib import Path

import config
from dashboard import database


def test_production_defaults():
    assert config.ENVIRONMENT == "production"
    assert not config.DEMO_MODE
    assert config.ALLOW_SELF_SIGNUP == (config.ENABLE_PUBLIC_DEMO_REGISTRATION or config._signup_opt_in())


def test_fresh_database_has_no_demo_accounts(tmp_path: Path):
    path = tmp_path / "fresh.sqlite"
    with database.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0


def test_create_and_authenticate_owner(tmp_path: Path):
    path = tmp_path / "owner.sqlite"
    user = database.create_organisation(
        config.PLATFORM_ORGANISATION, "owner@example.com", "Owner",
        "LongUniquePassword123!", platform_admin=True, path=path,
    )
    assert user["platform_admin"] is True
    assert database.authenticate("owner@example.com", "LongUniquePassword123!", path=path)
    assert database.authenticate("owner@example.com", "incorrect", path=path) is None


def test_bearer_token_not_written_to_url():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    login = (Path(__file__).resolve().parents[1] / "dashboard/pages/login.py").read_text()
    assert 'st.query_params["session"] =' not in app + login
    assert "st.query_params.get(\"session\")" not in app + login
