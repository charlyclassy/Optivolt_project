# Optivolt — enable demo registration without Streamlit Secrets

This update is intended **only for a public classroom/demo deployment**. It does **not** create production-ready user registration.

## Replace these files in your existing GitHub repository

- `config.py` (repository root)
- `dashboard/database.py`
- `dashboard/pages/login.py`

The ZIP also contains optional regression tests in `tests/`.

## How to upload

1. Extract the ZIP on your computer.
2. In your existing `charlyclassy/uptivolt` GitHub repository, open each corresponding file.
3. Click the pencil icon and replace its contents with the matching file from this ZIP. Commit each change. Alternatively upload the files using GitHub's folder paths; **preserve the nested folders**.
4. Do not upload the outer ZIP as a repository file.
5. Open the same Streamlit Community Cloud app; reboot it via **Manage app** if necessary.
6. Navigate Home → Create account. The signup fields should now be visible.
7. Use a fictional organisation name and a **unique, disposable test password**, then verify that the app signs you in.

## Why the form was blocked

The app checks both a config flag and a persisted `platform.allow_signup` value. The code-level demo toggle in `config.py` is now `ENABLE_PUBLIC_DEMO_REGISTRATION = True`. `dashboard/database.py` makes this demo override take precedence over old `allow_signup=false` values in the local database.

## Security limitation — important

- This opens public registration **to anyone with the app URL**; there is no email verification, signup rate limit, or persistent remote database.
- Accounts and passwords are stored in local SQLite (passwords salted+hashed). Those records may disappear on restart or redeployment. **Do not reuse a personal password or provide real customer data.**
- To disable public registration again, set `ENABLE_PUBLIC_DEMO_REGISTRATION = False` in `config.py`, commit, and reboot. When disabled, existing `platform.allow_signup` state controls availability.
- Streamlit Secrets do not need to be edited for this demo registration.
- Do not treat this app as production-ready until durable storage, stronger abuse protection, email verification, and secure account recovery have been added.

## Verified

All seven Phase 1 and public registration automated checks passed. The live Streamlit deployment still requires browser testing after the GitHub update.
