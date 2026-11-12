# ⚡ Optivolt — EV Depot Demand Forecasting & Grid Capacity Planning

## Project Overview

**Optivolt** is an interactive machine-learning application developed using Python and Streamlit to forecast electricity demand at electric vehicle (EV) depots and support grid-capacity planning.

The project combines historical electricity-demand data, LSTM neural networks, capacity-risk assessment and interactive visualisations to help users understand electricity consumption patterns and evaluate potential infrastructure constraints.


🌐 [Launch the Live Optivolt Application](https://optivoltproject-5dedagmozeh9mojotdd2kd.streamlit.app/)

💻 [View the GitHub Repository](https://github.com/charlyclassy/Optivolt_project)


## Project Objectives

- Analyse historical electricity-demand patterns across EV depots.
- Develop and evaluate machine-learning forecasting models.
- Predict future electricity demand using LSTM neural networks.
- Identify periods of potential grid-capacity risk.
- Evaluate the impact of EV fleet expansion.
- Provide accessible, interactive dashboards for electricity-demand planning.

## Key Features

### 1. Electricity Demand Forecasting

Optivolt uses **two trained LSTM models across three forecasting methods**, supporting prediction horizons from 24 hours to 30 days.

| Forecasting Method | Horizon | Approach |
|---|---|---|
| LSTM 1 — Short-term forecast | 24, 48 and 72 hours | Recursive forecasting using a trained short-term LSTM |
| LSTM 1 — Hybrid forecast | 7 and 14 days | First 72 hours predicted by LSTM 1; remaining days estimated using historical demand profiles |
| LSTM 2 — Long-term forecast | 30 days | Direct multi-step forecasting using a separately trained long-term LSTM |

The short-term model uses 96 previous readings, equivalent to 24 hours of electricity-demand observations at 15-minute intervals.

The long-term model uses a longer historical input window to generate a direct 30-day forecast.

Optivolt selects the appropriate forecasting method according to the chosen horizon and model availability. Where an alternative method is required, the application should identify the method actually used.

### 2. Grid Capacity Risk Assessment

The application compares predicted electricity demand against configurable grid-capacity limits.

Forecast results include:

- Highest predicted electricity demand (kW).
- Configured grid capacity (kW).
- Minimum spare grid capacity (kW).
- Total predicted energy consumption (kWh).
- Estimated duration of capacity exceedances.
- Busiest predicted demand period.
- Demand and grid-capacity visualisations.

These indicators help identify periods when electricity demand may approach or exceed the available capacity.

### 3. EV Fleet Growth Scenarios

Optivolt allows users to investigate how EV fleet expansion may affect electricity demand.

Users can apply demand-growth assumptions to explore potential future electricity requirements and identify possible capacity constraints.

For future-date simulations, historical demand patterns may be used to construct synthetic input readings.

**Synthetic readings represent simulated electricity demand, not verified live depot measurements.**

The application uses saved forecasting models for prediction without retraining them each time a forecast is generated.

### 4. Machine Learning Model Evaluation

Optivolt supports comparison of forecasting approaches, including:

| Model | Purpose |
|---|---|
| Previous-day baseline | Reference forecasting method |
| Random Forest | Ensemble machine-learning approach |
| XGBoost | Gradient-boosted regression |
| LSTM 1 | Short-term electricity-demand forecasting |
| LSTM 2 | Direct long-term electricity-demand forecasting |

Model evaluation includes:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R² score.
- Peak-demand prediction error.
- Capacity-breach detection.
- Confusion matrices and classification metrics, where applicable.

These metrics help assess forecasting accuracy and the reliability of grid-capacity warnings.

### 5. Interactive Dashboard and Depot Management

The Streamlit dashboard brings together:

- Depot selection and management.
- Historical electricity-demand analysis.
- Forecast horizon configuration.
- Automatic forecasting-method selection.
- Future-demand scenario planning.
- Grid-capacity monitoring.
- Interactive forecast visualisations.
- Model performance evaluation.

New depots require suitable historical observations and compatible model availability before forecasts can be generated.

## Dataset

The project uses historical electricity-demand data associated with the **UK Power Networks (UKPN) Optimise Prime project**.

The dataset includes electricity-demand observations recorded at 15-minute intervals across nine depot locations.

Historical observations are used to analyse electricity consumption patterns, prepare model inputs, train forecasting models and evaluate prediction accuracy.

Additional synthetic demand data has also been explored for experimental model training and scenario analysis.

Experimental synthetic-data-augmented models are maintained separately from the original trained models and are not automatically treated as production-ready.

## Technologies Used

| Technology | Application |
|---|---|
| Python | Application development |
| Streamlit | Interactive web dashboard |
| Pandas | Data preparation and analysis |
| NumPy | Numerical processing |
| TensorFlow / Keras | LSTM neural networks |
| Scikit-learn | Machine-learning models and evaluation |
| XGBoost | Gradient-boosted forecasting |
| Plotly | Interactive visualisations |
| GitHub | Version control and source-code hosting |
| Streamlit Community Cloud | Web application deployment |

## Forecasting Methodology

The Optivolt forecasting workflow consists of:

1. Loading historical depot electricity-demand observations.
2. Validating timestamps and preparing demand data.
3. Analysing historical consumption patterns.
4. Preparing time-series sequences and model inputs.
5. Loading the appropriate saved forecasting model.
6. Selecting the forecasting method according to the prediction horizon.
7. Generating electricity-demand predictions.
8. Comparing predicted demand against configured grid-capacity limits.
9. Displaying forecasts, summary indicators and capacity warnings.

### Short-Term Forecasting

LSTM 1 uses the previous 24 hours of observations to generate recursive electricity-demand predictions for 24, 48 or 72 hours.

### Hybrid Forecasting

For 7-day and 14-day predictions, Optivolt combines recursive LSTM 1 predictions for the first 72 hours with historical demand profiles for the remaining period.

This is a hybrid forecasting method, not a third trained LSTM model.

### Long-Term Forecasting

LSTM 2 is a separately trained long-term forecasting model designed to predict 30 days of electricity demand directly using a longer historical input sequence.

Its use depends on the availability of suitable input data and model artefacts.

### Future-Demand Simulation

Future-demand scenarios may use historical depot patterns to construct synthetic input readings for a selected future period.

Fleet-growth adjustments allow users to explore hypothetical increases in electricity demand.

These scenarios support planning and demonstration but should not be interpreted as live operational forecasts.

## Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/charlyclassy/Optivolt-Project.git
cd Optivolt-Project
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch the application using the configured Streamlit entry-point file:

```bash
streamlit run depot_updates.py
```

The application requires its supporting datasets, saved model files and dependencies to be available in their expected locations.

## Live Application

Optivolt is deployed using Streamlit Community Cloud.

**[Open Optivolt](https://optivolt-project-4htmec97k5xzsa25mvtfeq.streamlit.app/)**

The deployed application uses the version of the code and supporting files available in its configured GitHub repository and branch.

## Limitations

- Forecast accuracy depends on the quality and representativeness of historical data.
- Historical observations may not reflect current depot operating conditions.
- Recursive forecasting errors can accumulate over longer prediction horizons.
- Hybrid forecasts rely partly on historical demand profiles.
- Synthetic data represents simulated demand rather than verified real-time measurements.
- Long-term LSTM predictions require suitable historical input sequences.
- Fleet-growth scenarios depend on assumptions about future electricity consumption.
- Grid-capacity warnings are planning indicators rather than guarantees of operational safety.

## Future Improvements

Potential future development includes:

- Integration with live smart-meter and depot electricity-demand readings.
- Further validation of long-term LSTM forecasting performance.
- Evaluation of synthetic-data-augmented forecasting models.
- Incorporation of weather and EV charging-schedule information.
- Improved peak-demand prediction and uncertainty estimation.
- Expanded fleet-growth and charging-infrastructure scenarios.
- Additional depot datasets and forecasting capabilities.

## Project Summary

Optivolt demonstrates how data analytics, deep learning and interactive application development can support electricity-demand forecasting and grid-capacity planning for EV depots.

By combining two trained LSTM models, three forecasting methods, future-demand simulations and capacity-risk visualisations, the application provides a research and planning tool for exploring EV depot electricity requirements.

The project is intended for forecasting research, demonstration and decision support rather than certified real-time grid operation.

---

**Optivolt — Forecasting electricity demand. Supporting smarter EV depot planning.**
