"""Navigation and account-registration regression checks."""
from pathlib import Path

from dashboard import database

ROOT = Path(__file__).resolve().parents[1]


def test_public_home_is_default_and_registration_is_linked():
    app = (ROOT / "app.py").read_text()
    login = (ROOT / "dashboard/pages/login.py").read_text()
    landing = (ROOT / "dashboard/pages/landing.py").read_text()
    assert 'login_or_landing("landing"), title="Home"' in app
    assert 'url_path="welcome", default=True' in app
    assert 'url_path="create-account"' in app
    assert 'args=("landing",)' in login
    assert '_sign_in_user(user, remember=True)' not in login
    assert 'args=("signup",)' in landing


def test_registration_creates_account_and_sign_in(tmp_path):
    path = tmp_path / "registration.sqlite"
    user = database.create_organisation(
        "Demo EV Fleet", "new@example.com", "Test User", "SecureTestPassword2026!", path=path
    )
    assert user["role"] == "admin"
    assert database.authenticate("new@example.com", "SecureTestPassword2026!", path=path)


def test_public_demo_registration_visible_even_if_platform_was_closed(tmp_path):
    import config
    assert config.ENABLE_PUBLIC_DEMO_REGISTRATION
    path = tmp_path / "demo-signup.sqlite"
    database.set_signup_allowed(False, path=path)
    assert database.signup_allowed(path=path)
    user = database.create_organisation(
        "Example EV Depot", "temporary-test@example.com", "Test Operator",
        "TestOnlyPassword2026!", path=path,
    )
    assert database.authenticate("temporary-test@example.com", "TestOnlyPassword2026!", path=path)["org_id"] == user["org_id"]
