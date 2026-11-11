# Optivolt: Public home, registration and navigation

## Changes
- Visitors open the **Home** introduction before the sign-in page.
- Home explains LSTM demand forecasting, charging/grid planning limits, and links to Sign in and Register.
- **Back to home** now targets the registered Home page.
- **Create an account** is available when registration is enabled, with organisation name, full name, work email, password and confirmation.
- Corrected a sign-up crash: `_sign_in_user(user, remember=True)` passed an invalid argument.
- Public navigation sidebar is hidden; the landing page and account screens have their own navigation buttons.

## Deployment instructions for GitHub / Streamlit Cloud
1. Back up the project and any user database.
2. Upload the full project, maintaining `dashboard/` and `models/` folders. Do **not** upload only the small patch files to an empty repository.
3. Commit changes to your existing `uptivolt` repository, retaining your current Streamlit app and main file `app.py`.
4. The default public view now shows `/welcome`; `/sign-in` and `/create-account` are also available.
5. To enable registration **for a demonstration only**, open Streamlit Community Cloud > Manage app > Settings > Secrets, add `DEPOTWATT_ALLOW_SIGNUP = "true"` (TOML syntax), save and reboot. This is a non-secret feature flag. To disable registration, set `DEPOTWATT_ALLOW_SIGNUP = "false"`. Production defaults to registration OFF until you have persistent user storage, email verification, anti-bot controls and appropriate account-abuse protection.
6. On Streamlit Community Cloud, the SQLite database is **ephemeral**. New accounts may disappear after app restart, redeployment or migration. Registration is suitable for demonstration **only**, not production customers. For real registrations, migrate to persistent hosted storage and managed identity.
7. A new SQLite database has no default platform owner. Create one using `scripts/create_owner.py` on a server with persistent storage before production use.

## Acceptance checks
- Open the site in a new private window. It should land on the home page.
- Click Sign in, then Back to home, then Register.
- Complete registration with a fictitious test organisation and test email (not real customer data).
- Verify password confirmation and duplicate account errors.
- Confirm successful registration opens a signed-in workspace.
- Sign out and confirm protected dashboard pages are no longer accessible.
- Test that accounts survive a process restart **only on a persistent server**.

This patch does not claim to provide production-ready identity management or verified UK grid connection limits.
