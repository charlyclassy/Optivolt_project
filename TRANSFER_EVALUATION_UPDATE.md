# Depot transfer-model evaluation update

Only `optivolt_lstm.py` was modified in the application. The Home / Overview page, logo, sidebar and navigation were not edited.

For a newly uploaded depot without an active trained model, Fleet Growth & Capacity Planning now has an explicit **Evaluate borrowed LSTM on this depot's readings** action. It evaluates the selected saved source model on up to 384 recent valid, consecutive 96-step windows of the target depot's own readings, without fitting weights or mixing source-depot readings. It reports MAE, RMSE, a last-reading persistence baseline MAE, breach recall where defined, and an actual-versus-predicted plot. It invalidates the session result if the depot, source model selection or target readings change. This diagnostic is not a held-out evaluation of a model trained on the target depot, and is not production approval.

Validation: `python -m py_compile optivolt_lstm.py` passed. `PYTHONPATH=. python -m pytest -q tests` returned 6 passed, 1 failed (existing authentication source-inspection test). Full Streamlit/TensorFlow integration was not run. Do not deploy without checking the borrowed-model workflow against representative uploaded data.
