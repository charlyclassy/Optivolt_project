# Dedicated 30-day LSTM — setup

This update adds a **second, independently trainable LSTM**. It does not replace the existing 72-hour LSTM or the hybrid forecast.

## Files added/changed
- `train_long_term_lstm.py` — training, chronological holdout evaluation, saved model and metadata.
- `long_term_lstm.py` — 60-day history inference returning up to 30 days of 15-minute predictions.
- `depot_updates.py` — optional forecast method selector, visible only when a trained artefact exists.

## Train a depot

Install dependencies from `requirements.txt` and make sure TensorFlow is available. Then run locally (not during a Streamlit page request):

```bash
python train_long_term_lstm.py --depot Bexleyheath --epochs 35
```

For depots with spaces, quote the name, e.g. `--depot "Mount Pleasant"`.

The model and evaluation report will be written to `models/long_term_lstm/bexleyheath/`.
Repeat for each depot you want to enable. **Commit the resulting `model.keras` and `metadata.json` files** if their size and your repository policies permit, then redeploy Streamlit.

## Important limitations

- **Nine depot models were trained in this delivery using Keras 3 with the PyTorch backend.** The Bexleyheath model was trained for up to 20 epochs; the other eight for up to 5 epochs. TensorFlow-backed inference on Streamlit has not been verified here. Review `models/long_term_lstm/*/metadata.json` for held-out evaluation results.
- The model uses 60 complete days (60 × 96 readings) and predicts 30 days (30 × 96 readings) directly.
- Training examples overlap, and approximately one year of data is a small sample for this task. Examine test MAE/RMSE against the 30-day persistence baseline before presenting forecasts as reliable.
- This first integration is for **Forecast after latest observations** only, anchored at the latest complete day ending 23:45. Future-date scenarios continue using the existing hybrid method.
- The main `optivolt_lstm.py` module and saved 72-hour models have not been changed.
- Use this as an experimental model; the application must not claim it is validated until results are reviewed.

## Trained model evaluation

All nine depots have saved models. The table reports MAE on overlapping held-out 30-day forecast windows (kW). The persistence baseline repeats the previous 30 days. These are experimental results, not a guarantee of future performance.

| Depot | LSTM MAE (kW) | Persistence MAE (kW) |
|---|---:|---:|
| Bexleyheath | 2.69 | 3.59 |
| Camden | 2.80 | 3.19 |
| Dartford | 5.42 | 7.60 |
| Islington | 2.79 | 3.24 |
| Mount_Pleasant | 26.56 | 36.56 |
| Orpington | 2.23 | 2.39 |
| Premier_Park | 4.94 | 6.67 |
| Victoria | 4.99 | 7.64 |
| Whitechapel | 5.62 | 7.71 |
