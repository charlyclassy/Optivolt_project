# Optivolt — Phase 1 deployment and security changes

## What changed
- Production is now the default environment. Demo users are no longer automatically seeded.
- Public workspace registration is closed by default, unless explicitly enabled by the operator.
- Removed URL-query-parameter bearer session tokens. Sign-in is held in Streamlit server-side session state; users sign in again on a fresh session.
- Added one-time, interactive platform owner bootstrap, prompting for a password without displaying it in the shell command history.
- Docker now includes the missing `scripts/`, `.streamlit/`, sample data, saved models, historical depot CSVs, model evaluation CSVs and forecasting module.
- Startup seeds a named persistent model volume with packaged models on first launch. SQLite and uploads use a separate persistent state volume.
- `.dockerignore` and `.gitignore` reduce accidental inclusion of local secrets and databases.

## Install on a server
1. Make backups of your existing application, SQLite database and uploaded data before replacing files. **Do not delete existing Docker volumes.**
2. Extract the package to your project. Keep the folder structure.
3. Copy `.env.example` to `.env`, edit `DEPOTWATT_DOMAIN` to a DNS hostname you own, and point DNS at your server. Do not commit `.env`.
4. Install Docker Engine and Docker Compose, then run `docker compose build` and `docker compose up -d`.
5. **On a new installation only**, run:
   `docker compose exec app python scripts/create_owner.py --email owner@example.com --name "Platform Owner"`
   Enter a unique strong password at the hidden prompt. An existing platform owner prevents another bootstrap.
6. Open `https://<your-domain>`, sign in, and verify that the dashboard and models load.
7. Set up encrypted, tested backups of `depotwatt_state` and `depotwatt_models` volumes. Apply HTTPS and firewall rules.

## Existing installations
- Existing SQLite users and uploads in the mounted state volume are preserved. Existing URL session links will stop working and users must sign in again.
- Existing demo accounts **are not removed** from an existing database by changing the configuration. Audit them and disable or reset their passwords before going public.
- Existing Docker model volume contents are preserved. If the model volume is empty on the first launch, the image supplies the bundled models.
- Don't change `.env` from demo to production without checking existing seeded users and the persistent database.

## Security limits before production use
- Streamlit session state is not a secure persistent-cookie login service. This update eliminates URL-borne tokens but **does not** provide seamless cross-refresh sign-in. Consider a professionally managed OIDC identity provider for production.
- No external penetration test or tenant-isolation assessment has been performed. Review multi-organisation access checks, upload validation and rate limits before onboarding real customers.
- SQLite on one persistent host is for small deployments; Cloud platforms with ephemeral filesystem storage need an external persistent database / storage migration.
- This update has not been Docker-built or exercised end-to-end in this environment. Run the acceptance checks below before deployment.

## Quick acceptance checks
- Confirm a fresh production database has **zero** demo users.
- Run owner bootstrap once; a second run must reject another platform owner.
- Log in and confirm `?session=...` is **not** present in the address bar.
- Log out, confirm the dashboard is no longer accessible and reload the page.
- Create a test workspace only if you intentionally set `DEPOTWATT_ALLOW_SIGNUP=true`.
- Generate a forecast, restart containers, verify the model, account, upload and history remain.
- Verify HTTPS and that model/data volumes are backed up.
