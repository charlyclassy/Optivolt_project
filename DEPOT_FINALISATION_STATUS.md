# Optivolt depot-specific development status — 10 November 2026

## Changes in this revision
- Uploaded depots without a production LSTM can now open Model Performance directly, train the four comparison models explicitly, select ten metrics (five defaults), inspect saved results and residual plots, and download the depot-specific evidence. Switching depots does not start training.
- Uploaded depots without a production LSTM now have the eight original operational planning controls: fleet growth, fixed load, coincidence uplift, managed charging reduction, flexible charging share, time shift, temporary surge, and peak load cap. These modify a borrowed-model baseline generated from the selected depot's own readings. Added energy, headroom and capacity-duration summaries.
- Prior safeguards remain: explicit source-model choice, 96 consecutive readings, source-model shape/scaler validation, depot-specific experiment versioning, data-fingerprint invalidation, and no Bexleyheath metrics substituted for another depot.

## Important limits
- The uploaded-depot planning controls share the original concepts but are still presented in a dedicated route, not the full built-in-depot multi-step page. The simplified workflow does not include all original date/time selection and report elements.
- The borrowed model's accuracy on the target depot is unverified until separately evaluated. Four-model experiments measure one-step-ahead held-out performance, not 24–72-hour recursive forecasting.
- The environment has not run the Streamlit UI or TensorFlow training. Python syntax and source checks were run; this is not deployment certification.
- Broad navy card redesign across every requested screen is not complete.

Do not replace the production deployment without an end-to-end run using the intended Python and model dependencies.
