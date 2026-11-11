"""Targeted, idempotent UI patch for Optivolt Fleet Growth & Capacity Planning.
Run from the repository root: python apply_planning_ui_patch.py
Only optivolt_lstm.py is modified; a timestamped backup is created.
"""
from pathlib import Path
from datetime import datetime
import ast
import shutil
import sys

p = Path('optivolt_lstm.py')
if not p.is_file():
    sys.exit('ERROR: Run from the Optivolt repository root (optivolt_lstm.py not found).')
s = p.read_text(encoding='utf-8')
if 'OPTIVOLT_PLANNING_TWO_STAGE_V1' in s:
    sys.exit('Already patched; no changes made.')

def change(old, new, label):
    global s
    count = s.count(old)
    if count != 1:
        sys.exit(f'ERROR: {label} expected once, found {count}. File unchanged.')
    s = s.replace(old, new, 1)

change('''    st.markdown("### Forecast Period")
    _time_choices''', '''    # OPTIVOLT_PLANNING_TWO_STAGE_V1 — UI-only workflow; existing model and calculations retained.
    st.markdown("### Step 1 — Forecast depot demand using the saved LSTM")
    st.caption("First create a baseline electricity-demand forecast for the selected future dates. "
               "The model uses the selected depot's historical 15-minute readings; "
               "planning changes are applied separately in Step 2.")
    _hist_for_info = history_series.dropna().sort_index()
    _model_ready = pipeline.get("model") is not None
    _info1, _info2, _info3 = st.columns(3)
    _info1.metric("Selected depot", str(_planning_depot or "Not selected"))
    _info2.metric("Historical data through", _hist_for_info.index[-1].strftime("%d %b %Y") if len(_hist_for_info) else "Unavailable")
    _info3.metric("Saved LSTM model", "Available" if _model_ready else "Unavailable")
    st.caption(f"Data source: selected depot demand readings · {len(_hist_for_info):,} historical observations "
               f"· {SAMPLING_MINUTES}-minute intervals. Historical coverage: "
               f"{_hist_for_info.index[0]:%d %b %Y} to {_hist_for_info.index[-1]:%d %b %Y}." if len(_hist_for_info)
               else "No historical depot demand readings available.")
    _time_choices''', 'step 1 and data provenance')
change('''    if st.button("Generate baseline with saved LSTM", type="primary", key="planning_generate_baseline"):''', '''    if st.button("Generate LSTM baseline demand forecast", type="primary", key="planning_generate_baseline"):
        # A newly generated baseline invalidates any earlier planning results.
        st.session_state.pop("planning_applied_signature", None)''', 'baseline button')
change('''        st.success(f"Using the saved Demand Prediction forecast for {_planning_depot} and the selected dates.")''', '''        st.success(f"LSTM baseline ready: using the saved Demand Prediction forecast for {_planning_depot} and the selected dates.")''', 'saved baseline status')
change('''        st.warning("No saved LSTM forecast matches these dates. Showing a historical-pattern planning reference, "
            "not an LSTM prediction. Select 'Generate baseline with saved LSTM' above for model-based results.")''', '''        st.info("LSTM baseline not yet generated for these dates. The historical-pattern reference is for "
                "orientation only and is **not** an LSTM prediction. Select "
                "**Generate LSTM baseline demand forecast** before assessing planning changes.")''', 'reference clarification')
change('''    st.markdown("#### Planning Setup")''', '''    st.markdown("### Step 2 — Apply planning conditions")
    st.caption("Choose a preset or adjust the eight operational assumptions below. "
               "These changes modify the baseline demand; they do not retrain the LSTM.")''', 'step 2 title')
change('''    st.markdown("### Planning Assumptions")''', '''    st.markdown("#### Operational changes to assess")''', 'assumptions title')
anchor = '''    # ------------------------------------------------------------
    # Apply scenario transformations
    # ------------------------------------------------------------
    scenario_y = base_forecast.copy()'''
replacement = '''    # ------------------------------------------------------------
    # Explicit second action: only calculate planning outcomes after user request.
    # A changed control, period or baseline invalidates prior results.
    # ------------------------------------------------------------
    _has_lstm_baseline = (_planning_key in _planning_cache and
                          len(_planning_cache[_planning_key]) == _steps) or _saved_valid
    _planning_signature = (
        _planning_key,
        tuple(np.asarray(base_forecast, dtype=float).round(8)),
        int(scenario_growth_pct), float(scenario_fixed_load_kw),
        int(scenario_coincidence_pct), int(scenario_reduction_pct),
        int(flexible_pct), float(shift_hours), bool(stress_enabled),
        float(stress_start_hour), float(stress_duration_hours), int(stress_uplift_pct),
        bool(cap_enabled), float(scenario_load_cap_kw), float(planning_threshold),
        float(warning_threshold),
    )
    st.markdown("#### Generate planning-based demand forecast")
    st.caption("Apply the selected operating conditions to the LSTM baseline, then compare "
               "peak demand, grid utilisation, headroom, warning exposure and capacity breaches.")
    if st.button("Generate planning-based demand forecast", type="primary",
                 key="planning_generate_adjusted", disabled=not _has_lstm_baseline):
        st.session_state["planning_applied_signature"] = _planning_signature
    if not _has_lstm_baseline:
        st.info("Step 1 is required: generate or load a matching LSTM baseline before calculating planning results.")
        st.stop()
    if st.session_state.get("planning_applied_signature") != _planning_signature:
        st.info("Planning conditions are ready. Select **Generate planning-based demand forecast** "
                "to calculate and display the comparison. Re-run after changing any assumption.")
        st.stop()
    st.markdown("### Step 3 — Forecast & Capacity Impact")
    st.caption("LSTM baseline demand forecast vs planning-based demand forecast. "
               "Both use the same selected depot and forecast period.")

    # ------------------------------------------------------------
    # Apply scenario transformations (original calculations unchanged)
    # ------------------------------------------------------------
    scenario_y = base_forecast.copy()'''
change(anchor, replacement, 'planning action gate')
change('''    st.caption(f"**Capacity assessment:** {risk_status}. {risk_description}")''', '''    _alert_bg, _alert_border, _alert_icon = (
        ("#fff1f2", "#dc2626", "🚨") if risk_status == "Grid capacity exceeded"
        else ("#fffbeb", "#d97706", "⚠️") if "Warning" in risk_status
        else ("#ecfdf5", "#059669", "✅")
    )
    import html as _planning_html
    st.markdown(
        f'<div role="alert" style="background:{_alert_bg};border:1px solid {_alert_border};'
        f'border-left:6px solid {_alert_border};border-radius:12px;padding:14px 18px;'
        f'margin:12px 0;color:#172b4d;line-height:1.55">'
        f'<strong style="font-size:1.05rem">{_alert_icon} Capacity assessment: '
        f'{_planning_html.escape(risk_status)}</strong><div style="margin-top:4px">'
        f'{_planning_html.escape(risk_description)}</div></div>',
        unsafe_allow_html=True,
    )''', 'capacity alert')
ast.parse(s, filename=str(p))
backup = p.with_name(p.name + '.backup-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
shutil.copy2(p, backup)
p.write_text(s, encoding='utf-8')
print(f'Updated: {p}; backup: {backup}')
