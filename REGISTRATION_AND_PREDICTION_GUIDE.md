# Optivolt registration and LSTM prediction improvements

## Why the form said "sign-ups closed"

Public registration is **off by default** for security. The registration page itself worked; access was disabled by configuration. To temporarily enable it for a **demo only**:

1. Open the existing Streamlit Cloud app > **Manage app** > **Settings** > **Secrets**.
2. Add the TOML key below, save, and reboot the app:

```toml
DEPOTWATT_ALLOW_SIGNUP = "true"
```

3. Go to **Home > Register**, then enter a fictional/demo organisation, name, and password.
4. Sign in and test dashboard navigation. The first user becomes the organisation administrator.

If the page remains closed, the existing SQLite `platform.allow_signup` setting may override the environment flag. In that case a platform owner should open the **Platform Administration** page and enable sign-up explicitly. Do not wipe the existing database just to change this setting.

**Streamlit Cloud warning:** Its local SQLite database is not durable storage. Newly registered accounts may disappear after an app restart/redeployment. Use an external persistent database and implement email verification, anti-abuse/rate limits and account recovery before accepting real registrations. Do not use sensitive production passwords in the demo.

## What changed on the public Home page

- The hero chart shows **observed demand vs previous-day baseline vs LSTM forecast** on an actual held-out day.
- A second interactive chart lets viewers select a historical held-out day and a hypothetical depot planning limit.
- It plots a separate **illustrative capacity-capped demand** line and estimates how much demand would need rescheduling.
- The app reports the MAE improvement over the previous-day baseline, computed from `heldout_predictions_all_models.csv`.
- The green capacity-capped line is **not** a second trained AI model and **not** a valid charging schedule; it is an illustrative clipped curve. It does not shift deferred energy. The app explicitly explains this.

LSTM improves prediction, but actual charging optimisation would need a separate scheduling algorithm with constraints (fleet departure times, charger power, battery SoC, prices, transformer capacity). Do not claim a percent reduction in electricity consumption or grid violations solely from better prediction.

## GitHub update

Upload only the changed files to the existing `charlyclassy/uptivolt` repository, preserving their paths:

- `dashboard/pages/landing.py`
- `dashboard/pages/login.py`
- `config.py`
- `REGISTRATION_AND_PREDICTION_GUIDE.md`

Also verify the existing repository contains `heldout_predictions_all_models.csv` at its root. That historical data file is required for the new LSTM comparison. Do not overwrite your model files.

After deployment, test **Home > Register > Sign in > Dashboard** and select different historical days and capacity limits on Home.
