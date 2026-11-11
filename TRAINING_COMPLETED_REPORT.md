# Dedicated 30-day LSTM training report



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
