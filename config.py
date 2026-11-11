"""Central configuration: paths, defaults and constants shared by the whole project.

Deployment settings come from environment variables (see README, "Deployment"):

    DEPOTWATT_ENV         "demo" (default) seeds demo accounts; "production" does not
    DEPOTWATT_STATE_DIR   folder for the database and uploaded depot data (a persistent volume)
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- brand
PRODUCT_NAME = "Optivolt EV Depot Intelligence Portal"
PRODUCT_TAGLINE = "EV depot demand forecasting and capacity management"
DEFAULT_ORGANISATION = "Demo Fleet Operator"

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DATA_PATH = DATA_DIR / "sample_processed_data.csv"

MODELS_DIR = ROOT / "models"
ACTIVE_MODEL_DIR = MODELS_DIR / "global_model"
MODEL_ARCHIVE_DIR = MODELS_DIR / "archive"
MODEL_CANDIDATE_DIR = MODELS_DIR / "candidates"

OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "report_exports"
LATEST_FORECAST_PATH = OUTPUT_DIR / "latest_forecast.csv"
# Everything customers create (database, uploaded depot data) lives under STATE_DIR.
STATE_DIR = Path(os.environ.get("DEPOTWATT_STATE_DIR") or OUTPUT_DIR / "app_state")
DATABASE_PATH = STATE_DIR / "depotwatt.db"  # SQLite: organisations, users, settings, history
UPLOAD_DIR = STATE_DIR / "uploads"          # one sub-folder per organisation
ASSETS_DIR = ROOT / "dashboard" / "assets"

# ------------------------------------------------------------------ model package
MODEL_FILE = "xgboost_model.json"
METADATA_FILE = "model_metadata.json"
FEATURE_CONFIG_FILE = "feature_config.json"
METRICS_FILE = "model_metrics.csv"
TEST_PREDICTIONS_FILE = "test_predictions.csv"  # optional, used by Model Performance
REQUIRED_PACKAGE_FILES = (MODEL_FILE, METADATA_FILE, FEATURE_CONFIG_FILE, METRICS_FILE)

# What this version of the application knows how to serve.
SUPPORTED_MODEL_TYPES = ("xgboost.XGBRegressor",)
SUPPORTED_TARGETS = ("demand_ratio_next_step",)
SUPPORTED_FEATURE_SCHEMA = 1

# ------------------------------------------------------------- canonical data schema
COL_TIME = "timestamp"
COL_DEPOT = "depot_id"
COL_DEMAND = "demand_kw"

# ------------------------------------------------------------------- training setup
SAMPLING_MINUTES = 15
DATA_TIMEZONE = "Europe/London"
# Chronological split shared by every depot (local time, end-exclusive).
DATA_START = "2021-07-01"   # first month with >95% meter coverage at every depot
TRAIN_END = "2022-04-01"
VALID_END = "2022-05-10"
TEST_END = "2022-07-26"
EVAL_HORIZON_HOURS = 24
EVAL_ORIGIN_HOUR = 0  # one forecast per day, issued at midnight local time
# Planning limit used when a depot has no configured capacity: this quantile of the
# depot's daily peak demand over the training period.
DEFAULT_CAPACITY_QUANTILE = 0.75

# -------------------------------------------------------------------- app defaults
DEFAULT_WARNING_ALPHA = 0.80
DEFAULT_HORIZON_HOURS = 24
HORIZON_CHOICES = (6, 12, 24, 48, 72)
DEFAULT_HISTORY_WINDOW_HOURS = 48
REPLAY_DAYS = 14  # how many past days are replayed for History / Alerts

# ------------------------------------------------------------------ deployment
ENVIRONMENT = os.environ.get("DEPOTWATT_ENV", "production").strip().lower()
# Demo accounts require TWO explicit opt-ins. Never seed public deployments automatically.
DEMO_MODE = (ENVIRONMENT == "demo" and os.environ.get("DEPOTWATT_ENABLE_DEMO_SEED") == "1")
# Temporary PUBLIC DEMO switch. True makes Create account available on Streamlit
# Community Cloud without configuring Secrets. This is NOT a production-safe
# registration system: no email verification, durable account storage, or
# signup rate limiting is provided. Do not collect real credentials/data.
# Set False again before using Optivolt with real organisations.
ENABLE_PUBLIC_DEMO_REGISTRATION = True

# The registration page is available but public registration is OFF by default.
# Set DEPOTWATT_ALLOW_SIGNUP=true explicitly for a demo environment. Do not
# enable for real users without email verification, rate limiting, and durable
# user storage. The platform owner may also control this in Platform > Sign-up.
def _signup_opt_in() -> bool:
    """Explicit opt-in from environment or Streamlit Cloud secrets; off by default."""
    flag = os.environ.get("DEPOTWATT_ALLOW_SIGNUP")
    if flag is None:
        try:
            import streamlit as st
            flag = st.secrets.get("DEPOTWATT_ALLOW_SIGNUP", "false")
        except (FileNotFoundError, KeyError, OSError, AttributeError, ImportError):
            flag = "false"
    return str(flag).strip().lower() in ("1", "true", "yes", "on")

ALLOW_SELF_SIGNUP = ENABLE_PUBLIC_DEMO_REGISTRATION or _signup_opt_in()
PLATFORM_ORGANISATION = "Optivolt Solutions Limited"

# Demo accounts, created on first run in demo mode only. In production no account is seeded:
# create the platform owner with  python scripts/create_owner.py
DEMO_OWNER_EMAIL = "owner@depotwatt.demo"
DEMO_OWNER_PASSWORD = os.environ.get("DEPOTWATT_DEMO_OWNER_PASSWORD", "")
DEMO_ADMIN_EMAIL = "admin@depotwatt.demo"
DEMO_ADMIN_PASSWORD = os.environ.get("DEPOTWATT_DEMO_ADMIN_PASSWORD", "")
DEMO_VIEWER_EMAIL = "viewer@depotwatt.demo"
DEMO_VIEWER_PASSWORD = os.environ.get("DEPOTWATT_DEMO_VIEWER_PASSWORD", "")

# Bundled real depots: UK Power Networks "Optimise Prime" WS2 depot trial (Royal Mail
# depots, table 27 "depot load data"). Keys are the dataset's numeric depot_id; names
# were matched against the dataset's named hourly load profiles (table 32).
DATA_SOURCE_NAME = "UK Power Networks Optimise Prime (WS2 depot trial)"
DATA_SOURCE_URL = "https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-optimise-prime"
BUNDLED_DEPOTS = {
    "1": "Bexleyheath",
    "2": "Dartford",
    "3": "Islington",
    "4": "Mount Pleasant",
    "5": "Orpington",
    "6": "Premier Park",
    "7": "Whitechapel",
    "8": "Camden",
    "9": "Victoria",
}
