# ⚡ Optivolt — EV Depot Demand Forecasting & Grid Capacity Planning

## Project Overview

**Optivolt** is an interactive machine-learning application developed using Python and Streamlit to forecast electricity demand at electric vehicle (EV) depots and support grid-capacity planning.

The project combines historical electricity-demand data, predictive modelling, capacity-risk assessment and interactive visualisations to help users understand demand patterns and evaluate potential infrastructure constraints.

🌐 **[Launch the Live Optivolt Application](https://optivolt-project-4htmec97k5xzsa25mvtfeq.streamlit.app/)**

💻 **[View the GitHub Repository](https://github.com/charlyclassy/Optivolt-Project)**

## Project Objectives

- Analyse historical electricity-demand patterns across EV depots.
- Develop and evaluate machine-learning forecasting models.
- Predict future electricity demand using LSTM neural networks.
- Identify periods of potential grid-capacity risk.
- Evaluate the impact of EV fleet expansion.
- Provide accessible, interactive dashboards for operational decision-making.

## Key Features

### 1. Electricity Demand Forecasting

Optivolt uses a Long Short-Term Memory (LSTM) neural network to predict future electricity demand.

The saved short-term model uses 96 historical readings, equivalent to 24 hours of data at 15-minute intervals.

The application supports:

- Short-term forecasting for up to 72 hours.
- Extended forecasting for 7, 14 and 30 days.
- Interactive forecast visualisations.
- Demand trend and peak-demand analysis.

Extended forecasts use a hybrid approach combining short-term LSTM predictions with historical time-of-week demand patterns.

### 2. Grid Capacity Risk Assessment

The application compares forecast electricity demand against configurable grid-capacity thresholds.

Features include:

- Forecast peak demand in kilowatts (kW).
- Grid-capacity utilisation.
- Remaining grid headroom.
- Warning and breach thresholds.
- Capacity-risk visualisations.
- Interpretation of forecast results.

### 3. EV Fleet Growth Scenarios

Users can explore how increases in EV fleet size may affect future electricity demand.

Scenario analysis supports comparisons between different fleet-growth assumptions and helps identify potential capacity constraints.

### 4. Machine Learning Model Evaluation

Optivolt includes comparisons between:

| Model | Purpose |
|---|---|
| Previous-day baseline | Reference forecasting approach |
| Random Forest | Ensemble machine-learning model |
| XGBoost | Gradient-boosted regression model |
| LSTM | Deep-learning time-series forecasting |

Performance evaluation includes:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
- Capacity-breach detection
- Confusion matrices and classification metrics

These metrics help assess forecasting accuracy and the reliability of capacity-risk predictions.

### 5. Interactive Dashboard

The Streamlit application brings together forecasting, historical demand analysis, scenario planning and model evaluation.

Interactive controls allow users to explore different depots, forecasting horizons and capacity-planning assumptions.

## Dataset

The project uses historical electricity-demand data associated with the **UK Power Networks Optimise Prime** project.

Data is analysed at 15-minute intervals across nine depot locations.

Historical demand observations are used to investigate consumption patterns, prepare forecasting inputs and evaluate model performance.

## Technologies Used

| Technology | Application |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and dashboard |
| Pandas | Data preparation and analysis |
| NumPy | Numerical processing |
| TensorFlow / Keras | LSTM neural network |
| Scikit-learn | Machine-learning models and evaluation |
| XGBoost | Gradient-boosted forecasting |
| Plotly | Interactive visualisations |
| GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

## Forecasting Methodology

The forecasting workflow consists of:

1. Loading historical electricity-demand observations.
2. Validating timestamps and preparing the dataset.
3. Analysing historical demand trends.
4. Preparing features and time-series sequences.
5. Training or loading machine-learning models.
6. Generating electricity-demand predictions.
7. Evaluating forecast accuracy.
8. Comparing forecast demand against capacity thresholds.
9. Presenting results through interactive dashboards.

The existing short-term LSTM uses 24 hours of previous readings to generate recursive forecasts for up to 72 hours.

For longer horizons, Optivolt combines those predictions with historical demand patterns rather than claiming to use a separately trained long-term LSTM.

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

Launch the application:

```bash
streamlit run app.py
```

The application also requires its associated datasets and saved model artefacts to be available in the expected locations.

## Limitations

- Forecast accuracy depends on the quality and representativeness of historical data.
- Recursive long-horizon predictions may accumulate errors.
- Extended forecasts rely partly on historical demand patterns.
- Fleet-growth scenarios are estimates based on planning assumptions.
- Grid-capacity assessments depend on the thresholds configured in the application.

## Future Improvements

Potential future development includes:

- Training a dedicated long-term LSTM for multi-week forecasting.
- Incorporating weather and charging-schedule information.
- Improving prediction accuracy during unusual demand peaks.
- Expanding scenario-planning functionality.
- Supporting additional depot locations and datasets.

## Project Summary

Optivolt demonstrates how data analytics, machine learning and interactive application development can be combined to address a practical challenge in EV infrastructure planning.

The project supports the interpretation of historical electricity demand, forecasting of future consumption and identification of potential grid-capacity risks.

---

**Optivolt — Forecasting electricity demand. Supporting smarter EV depot planning.**
