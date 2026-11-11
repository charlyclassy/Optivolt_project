# Optivolt — complete GitHub upload

This is the **complete source project** combining the latest public LSTM prediction demonstration and the demonstration-account registration fix. It is intended to update your **existing** GitHub repository `charlyclassy/uptivolt`, not create another project.

## Upload

1. Back up your existing GitHub repository and any database or user records.
2. Extract this ZIP; upload **the contents of the extracted folder**, not the ZIP file or an extra enclosing folder.
3. Keep `app.py`, `config.py`, `requirements.txt` at the repository root. Preserve nested `dashboard/`, `dashboard/pages/`, `models/`, `data/`, `scripts/`, and `.streamlit/` paths.
4. Do **not** upload `.env` or `*.db`; the included `.env.example` is a sample only.
5. Commit changes to `main`. Streamlit Community Cloud should redeploy the existing app automatically.
6. Test Home → Create account → Sign in → Dashboard, and the public LSTM comparison page.

## Registration limits

`config.py` contains `ENABLE_PUBLIC_DEMO_REGISTRATION = True` to display the form **without Streamlit Secrets**. Anyone with your public app URL can register. Use fictional data and a disposable password only. To disable public signup, set this constant to `False` and redeploy.

Accounts are stored in local SQLite which is not reliably persistent on Streamlit Community Cloud. This is **not** a production authentication deployment. Do not use real customer data.

## Forecast illustration

The public Home page uses `heldout_predictions_all_models.csv` for historical held-out predictions and compares LSTM against a previous-day baseline. The capacity limit demonstration is an illustration, not a mathematically optimised charging schedule.

## Checks

Python syntax and available automated tests were checked on this assembled project. **The live Streamlit login and model pages still require testing on deployment.**

## Troubleshooting

If `ImportError` reports a missing `dashboard` module, check that `dashboard/__init__.py`, `dashboard/database.py`, and `dashboard/services.py` are uploaded in the correct directory. If TensorFlow/model loading fails, inspect Streamlit Cloud logs and verify all `models/` files and `requirements.txt`.
