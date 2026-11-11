# Transfer forecast freshness safeguard

This incremental update only modifies `optivolt_lstm.py` and adds this note.

When the selected depot borrows another depot's saved LSTM, the in-session transfer evaluation and planning forecast now record a signature of the source model and scaler artifacts. Replacing either artifact invalidates the displayed cached transfer result, rather than showing an earlier result as if it came from the new source model. Transfer evaluation is also invalidated when the selected depot's grid capacity changes, because breach recall depends on that threshold.

No Home / Overview, logo, sidebar, navigation, authentication, or other pages were modified.

Validation: Python compileall passed. Existing automated tests: 6 passed, 1 failed (`test_bearer_token_not_written_to_url`); this pre-existing failure concerns session tokens in `app.py` and was deliberately not changed because it is outside the requested pages. Full Streamlit/TensorFlow inference testing has not been performed. Do not deploy as a verified final release.
