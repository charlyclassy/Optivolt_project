# Depot-specific model comparison development update

This development ZIP adds an opt-in four-model experiment to the Model Performance page. The experiment uses the active depot's own historical time series, 96-reading consecutive input windows, a chronological 70/15/15 split, and writes depot-specific results under `data/depot_comparisons/<depot>/`. It does not replace the saved production model. It adds an explicit built-in-depot model borrowing selector to Fleet Growth when a selected depot's active pipeline is unavailable, but the current application's top-level pipeline gate still blocks uploaded depots without an active LSTM before the planning page can render. That end-to-end route is **not complete**.

The experimental comparison is one-step-ahead, not a recursive multi-day evaluation. Review dataset preprocessing, deployment persistence, CPU/memory limits, and access control before production use. The app has not been end-to-end tested in this environment.
