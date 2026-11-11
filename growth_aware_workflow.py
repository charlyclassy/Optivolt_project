"""Separate train / validate / predict workflow for assumed fleet and charging inputs.

The measured holdout is never augmented. Fleet metadata is hypothetical.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from growth_aware_model import train_and_backtest, predict_scenario

ROOT = Path(__file__).resolve().parent / 'growth_aware_saved'


def _paths(depot):
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in depot)
    return ROOT / f'{safe}.keras', ROOT / f'{safe}.json'


def _persist(depot, result, assumptions):
    """Keep every candidate; promote by training-period validation loss only.

    The chronological test set is for reporting, never candidate selection.
    """
    model, scale, _series, metrics, details = result
    model_path, meta_path = _paths(depot)
    model_path.parent.mkdir(exist_ok=True, parents=True)
    previous = {}
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            previous = {}
    version = int(previous.get('version', 0)) + 1
    candidate_path = model_path.with_name(f'{model_path.stem}_v{version}.keras')
    model.save(str(candidate_path))
    new_loss = float(details.get('training_selection', {}).get('selected_validation_loss', float('inf')))
    old_loss = float(previous.get('selection_validation_loss', float('inf')))
    promote = not model_path.exists() or new_loss < old_loss
    if promote:
        shutil.copy2(candidate_path, model_path)
    history = list(previous.get('history', []))
    history.append({
        'version': version, 'file': candidate_path.name,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection_validation_loss': new_loss,
        'promoted': bool(promote),
        'metrics': metrics.to_dict(orient='records'),
    })
    payload = {
        'scale': float(scale) if promote else previous.get('scale', float(scale)),
        'assumptions': assumptions if promote else previous.get('assumptions', assumptions),
        'metrics': metrics.to_dict(orient='records') if promote else previous.get('metrics', metrics.to_dict(orient='records')),
        'details': details if promote else previous.get('details', details),
        'approved': bool(previous.get('approved', False)) if not promote else False,
        'version': version,
        'active_version': version if promote else previous.get('active_version', version),
        'selection_validation_loss': new_loss if promote else old_loss,
        'history': history,
        'latest_candidate': history[-1],
        'acceptance_min_gain_percent': previous.get('acceptance_min_gain_percent', DEFAULT_MIN_GAIN),
        'acceptance_max_mae_kw': previous.get('acceptance_max_mae_kw', DEFAULT_MAX_MAE),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return model_path, promote


def _load(depot):
    model_path, meta_path = _paths(depot)
    if not (model_path.exists() and meta_path.exists()):
        return None
    from tensorflow import keras
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    return keras.models.load_model(str(model_path)), meta


DEFAULT_MIN_GAIN = 10.0  # Illustrative planning criterion, not an industry standard.
DEFAULT_MAX_MAE = 2.5   # Must be reviewed against the depot's operational requirements.


def _assessment(metrics, min_gain=DEFAULT_MIN_GAIN, max_mae=DEFAULT_MAX_MAE):
    """Assess the 24h holdout result without changing or retraining the model."""
    table = metrics.copy()
    table['Model'] = table['Model'].replace({'Direct model': 'Newly Trained LSTM'})
    p = table.pivot_table(index='Horizon (hours)', columns='Model', values='MAE (kW)', aggfunc='first')
    if 24 not in p.index or not {'Newly Trained LSTM', 'Previous-day baseline'}.issubset(p.columns):
        return False, False, float('nan'), float('nan')
    new_mae = float(p.loc[24, 'Newly Trained LSTM'])
    base_mae = float(p.loc[24, 'Previous-day baseline'])
    gain = (base_mae - new_mae) / base_mae * 100 if base_mae > 0 else float('nan')
    return bool(np.isfinite(gain) and gain >= min_gain), bool(np.isfinite(new_mae) and new_mae <= max_mae), gain, new_mae


def _status(metrics, min_gain=DEFAULT_MIN_GAIN, max_mae=DEFAULT_MAX_MAE):
    gain_ok, mae_ok, _, _ = _assessment(metrics, min_gain, max_mae)
    return gain_ok and mae_ok


def _train_with_progress(raw, vehicles, charger, util, epochs, improve=False):
    bar = st.progress(0, text='Preparing training data…')
    status = st.empty()

    def report(attempt, attempts, epoch, max_epochs, val_loss):
        fraction = ((attempt - 1) * max_epochs + epoch) / (attempts * max_epochs)
        pct = min(99, int(100 * fraction))
        bar.progress(pct, text=f'Training {pct}%')
        status.caption(f'Model {attempt}/{attempts} · Epoch {epoch}/{max_epochs}')

    result = train_and_backtest(raw, vehicles, charger, util, epochs,
                                progress_callback=report, improve=improve)
    bar.progress(100, text='Training and evaluation complete')
    status.empty()
    return result


def render(depot, raw):
    st.markdown('''<div style="background:linear-gradient(110deg,#15366C,#087D8B);padding:17px 22px;
    border-radius:15px;margin:10px 0 12px"><div style="font-size:25px;font-weight:800;color:#FFFFFF">
    Fleet &amp; Charging-Aware Prediction</div><div style="color:#E5F8FF;font-size:15px">
    Train, validate and use a separate direct 24-hour model.</div></div>''', unsafe_allow_html=True)
    st.markdown('''<style>
    div[data-testid="stAppViewContainer"] div[data-testid="stButton"] button[kind="primary"] {
      background:linear-gradient(110deg,#1D4ED8,#087D8B)!important;
      color:#FFFFFF!important;font-weight:750!important;border:0!important;
      box-shadow:0 5px 14px rgba(21,54,108,.18)!important;
    }
    div[data-testid="stAppViewContainer"] div[data-testid="stButton"] button[kind="primary"] * {color:#FFFFFF!important;}
    div[data-testid="stDataFrame"] {border:1px solid #B8CFEA;border-radius:12px;overflow:hidden;}
    </style>''', unsafe_allow_html=True)
    st.caption('Fleet and charging settings are planning assumptions.')
    # Apply navigation requests BEFORE the radio widget is instantiated.
    stage_key = f'ga_stage_{depot}'
    pending_key = f'ga_pending_stage_{depot}'
    if pending_key in st.session_state:
        st.session_state[stage_key] = st.session_state.pop(pending_key)
    stage = st.radio('Workflow', ['1 · Train', '2 · Validate', '3 · Predict'], horizontal=True,
                     key=stage_key)
    key = f'ga_workflow_{depot}'
    saved_path, meta_path = _paths(depot)
    session = st.session_state.get(key)
    if session is not None and not meta_path.exists():
        session = None
    if stage.startswith('1'):
        a,b,c,d=st.columns(4)
        with a: vehicles=st.number_input('Vehicles', min_value=1,max_value=1000,value=12,key=f'gw_v_{depot}')
        with b: charger=st.number_input('Charger rating (kW)',min_value=1.0,max_value=350.0,value=7.0,key=f'gw_c_{depot}')
        with c: util=st.slider('Charging utilisation (%)',0,100,35,key=f'gw_u_{depot}')/100
        with d: epochs=st.selectbox('Training epochs',[3,5,10,20],index=1,key=f'gw_e_{depot}')
        if st.button('Train and evaluate on historical holdout',type='primary',key=f'gw_train_{depot}'):
            try:
                with st.spinner('Training model…'):
                    result=_train_with_progress(raw,vehicles,charger,util,epochs)
                    st.session_state[key]=result
                    _persist(depot,result,{'vehicles':vehicles,'charger_kw':charger,'utilisation':util,'epochs':epochs})
                st.session_state[pending_key] = '2 · Validate'
            except Exception as exc:
                st.error(f'Training failed: {exc}')
            else:
                st.rerun()

    elif stage.startswith('2'):
        if not meta_path.exists():
            st.info('Train a model first.')
            return
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        metrics = pd.DataFrame(meta['metrics'])
        details = meta['details']
        metrics['Model'] = metrics['Model'].replace({'Direct model': 'Newly Trained LSTM'})
        st.markdown("""<div style="background:linear-gradient(110deg,#15366C,#087D8B);padding:14px 18px;border-radius:12px;margin:6px 0 12px"><strong style="color:white;font-size:22px">Validation results</strong></div>""", unsafe_allow_html=True)
        if 'NMAE (%)' not in metrics.columns:
            st.warning('Retrain to calculate normalised accuracy.')
            return
        criteria_key = f'gw_criteria_open_{depot}'
        show_criteria = bool(st.session_state.get(criteria_key, False))
        gain_key = f'gw_gain_{depot}'
        mae_key = f'gw_mae_{depot}'
        if gain_key not in st.session_state:
            st.session_state[gain_key] = float(meta.get('acceptance_min_gain_percent', DEFAULT_MIN_GAIN))
        if mae_key not in st.session_state:
            st.session_state[mae_key] = float(meta.get('acceptance_max_mae_kw', DEFAULT_MAX_MAE))
        if show_criteria:
            st.markdown('<div style="background:#15366C;color:#FFFFFF;padding:12px 16px;border-radius:10px;font-weight:750">Review acceptance criteria</div>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                st.number_input('Minimum 24h MAE improvement (%)', min_value=0.0, max_value=100.0,
                    step=1.0, key=gain_key, help='Relative to previous-day baseline; higher improvement is better.')
            with cb:
                st.number_input('Maximum 24h MAE (kW)', min_value=0.01, max_value=10000.0,
                    step=0.1, key=mae_key, help='Absolute error limit; lower MAE is better.')
            st.caption('Planning thresholds only. Changing them reassesses existing holdout results without retraining.')
        min_gain = float(st.session_state[gain_key])
        max_mae = float(st.session_state[mae_key])
        gain_ok, mae_ok, gain24, mae24 = _assessment(metrics, min_gain, max_mae)
        passed = bool(gain_ok and mae_ok)
        pivot = metrics.pivot(index='Horizon (hours)', columns='Model', values=['MAE (kW)', 'RMSE (kW)', 'NMAE (%)'])
        new = pivot['MAE (kW)']['Newly Trained LSTM']
        base = pivot['MAE (kW)']['Previous-day baseline']
        improvement = (base-new)/base.replace(0,np.nan)*100
        nmae = pivot['NMAE (%)']['Newly Trained LSTM']
        a,b,c = st.columns(3)
        a.metric('24h vs baseline', f'{improvement.loc[24]:+.1f}%' if 24 in improvement.index else 'N/A')
        b.metric('24h normalised error', f'{nmae.loc[24]:.1f}%' if 24 in nmae.index else 'N/A')
        c.metric('Decision', 'PASS' if passed else 'REVIEW')
        st.success('Selected planning criteria met. Continue to Predict.') if passed else st.warning('Selected criteria not met. Review thresholds or improve and retrain; prediction remains experimental.')
        st.caption('MAE: average absolute error (lower is better). RMSE: penalises large misses. NMAE: error relative to mean demand. RMSE and NMAE remain visible but are not acceptance gates.')
        st.caption(f'24h baseline gain ≥ {min_gain:g}%: ' + ('✓' if gain_ok else '✗') + f'  ·  24h MAE ≤ {max_mae:g} kW: ' + ('✓' if mae_ok else '✗') + '  ·  RMSE/NMAE: diagnostic only')
        table = pd.DataFrame({
            'Ahead (h)': new.index.astype(int),
            'LSTM MAE (kW)': new.values.round(2),
            'Baseline MAE (kW)': base.values.round(2),
            'Gain (%)': improvement.values.round(1),
            'LSTM RMSE (kW)': pivot['RMSE (kW)']['Newly Trained LSTM'].values.round(2),
            'NMAE (%)': nmae.values.round(1),
        })
        st.dataframe(table, hide_index=True, use_container_width=True,
            column_config={'Ahead (h)': st.column_config.NumberColumn('Ahead (h)', format='%d')})
        latest = meta.get('latest_candidate')
        if latest and len(meta.get('history', [])) > 1:
            hist = meta['history']
            prev = hist[-2]
            def _mae24(item):
                rows = pd.DataFrame(item.get('metrics', []))
                rows['Model'] = rows['Model'].replace({'Direct model':'Newly Trained LSTM'})
                match = rows[(rows['Model']=='Newly Trained LSTM') & (rows['Horizon (hours)']==24)]
                return float(match.iloc[0]['MAE (kW)']) if len(match) else None
            old, current = _mae24(prev), _mae24(latest)
            if old is not None and current is not None:
                st.info(f'Latest retraining: {"Improved" if current < old else "Worse" if current > old else "Unchanged"} on the reported 24h test MAE ({old:.2f} → {current:.2f} kW). ' +
                    ('Candidate promoted by training-period validation loss.' if latest['promoted'] else 'Previous active model retained based on training-period validation loss.'))
        meta['approved'] = passed
        meta['acceptance_min_gain_percent'] = min_gain
        meta['acceptance_max_mae_kw'] = max_mae
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        st.markdown('''<div style="background:linear-gradient(110deg,#15366C,#087D8B);padding:12px 18px;
        border-radius:12px;margin:18px 0 12px;color:white;font-size:20px;font-weight:800">Next step</div>''',
        unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            if st.button('Improve & Retrain', type='primary', use_container_width=True, key=f'gw_improve_{depot}'):
                try:
                    assumptions = meta['assumptions']
                    with st.spinner('Training candidates…'):
                        result = _train_with_progress(raw, assumptions['vehicles'], assumptions['charger_kw'],
                            assumptions['utilisation'], max(10,int(assumptions['epochs'])), improve=True)
                        _, promoted = _persist(depot, result, assumptions)
                    st.session_state.pop(key, None)
                    st.session_state.pop(f'gw_pred_{depot}', None)
                    st.session_state[pending_key] = '2 · Validate'
                    st.rerun()
                except Exception as exc:
                    st.error(f'Retraining failed: {exc}')
        with col2:
            if st.button('Hide criteria' if show_criteria else 'Review criteria', use_container_width=True, key=f'gw_criteria_{depot}'):
                st.session_state[criteria_key] = not show_criteria
                st.rerun()
        with col3:
            if st.button('Continue to Predict', use_container_width=True, key=f'gw_next_{depot}'):
                st.session_state[pending_key] = '3 · Predict'
                st.rerun()
        with st.expander('Measured vs predicted · held-out example'):
            if details.get('example_observed_kw') and details.get('example_predicted_kw'):
                import plotly.graph_objects as go
                t = pd.date_range(pd.Timestamp(details['example_start']), periods=len(details['example_observed_kw']), freq='15min')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t,y=details['example_observed_kw'],name='Measured',line=dict(color='#183B70',width=3)))
                fig.add_trace(go.Scatter(x=t,y=details['example_predicted_kw'],name='New LSTM',line=dict(color='#078B91',width=3)))
                fig.update_layout(template='plotly_white',height=340,margin=dict(l=20,r=15,t=35,b=20),yaxis_title='Demand (kW)')
                st.plotly_chart(fig,use_container_width=True)
        st.download_button('Download validation CSV', metrics.to_csv(index=False), file_name=f'{depot}_validation.csv',mime='text/csv')
        st.caption(f"Measured test: {details['test_start'][:10]}–{details['test_end'][:10]} · {details['test_windows']} windows.")
        with st.expander('Limitations'):
            st.write('Synthetic growth augments training only. Fleet inputs are assumptions, not measured telemetry. The same held-out test set must not be repeatedly used to choose a model or tune criteria. Repeated experiments require a new final untouched test period.')
    else:
        if not meta_path.exists():
            st.warning('Train and validate a model first.')
            return
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        approved = bool(meta.get('approved',False))
        if not approved:
            st.warning('Experimental forecast only — this model has not passed the selected accuracy criteria. Do not use it for operational decisions.')
            confirm = st.checkbox('I understand this prediction is experimental', key=f'gw_experimental_{depot}')
            if not confirm:
                return
        try:
            model, _ = _load(depot)
            scale = float(meta['scale'])
            from growth_aware_model import _clean
            series = _clean(raw)
        except Exception as exc:
            st.error(f'Could not load the active model: {exc}')
            return
        x,y,z,w = st.columns(4)
        with x: date = st.date_input('Prediction start date',value=pd.Timestamp.today().date()+pd.Timedelta(days=1),key=f'gw_date_{depot}')
        with y: vehicles = st.number_input('Vehicles',1,1000,12,key=f'gw_pv_{depot}')
        with z: charger = st.number_input('Charger power (kW)',1.,350.,7.,key=f'gw_pc_{depot}')
        with w: util = st.slider('Utilisation (%)',0,100,35,key=f'gw_pu_{depot}')/100
        growth = st.slider('Fleet-demand growth assumption (%)',0,150,20,key=f'gw_pg_{depot}')
        capacity = st.number_input('Grid capacity (kW)',min_value=.1,value=70.,key=f'gw_cap_{depot}')
        if st.button('Generate future prediction',type='primary',key=f'gw_predict_{depot}'):
            try:
                with st.spinner('Generating 24-hour forecast…'):
                    prediction = predict_scenario(model,scale,series,date,vehicles,charger,util,growth)
                st.session_state[f'gw_pred_{depot}'] = prediction
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')
        prediction = st.session_state.get(f'gw_pred_{depot}')
        if prediction is not None:
            peak = float(prediction.max())
            energy = float(prediction.sum()*.25)
            a,b,c,d = st.columns(4)
            a.metric('Peak',f'{peak:.2f} kW')
            b.metric('Energy',f'{energy:.1f} kWh')
            c.metric('Headroom',f'{capacity-peak:+.2f} kW')
            d.metric('Capacity breach',f'{(prediction>=capacity).sum()*.25:.2f} h')
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prediction.index,y=prediction.values,name='24h prediction',line=dict(color='#087F8C',width=3),fill='tozeroy'))
            fig.add_hline(y=capacity,line_color='#C22B34',line_dash='dash',annotation_text='Grid capacity')
            fig.update_layout(template='plotly_white',height=370,margin=dict(l=20,r=20,t=25,b=20),yaxis_title='Demand (kW)')
            st.plotly_chart(fig,use_container_width=True)
            st.download_button('Download prediction CSV',prediction.rename('predicted_demand_kw').to_csv(header=True),file_name=f'{depot}_fleet_prediction.csv',mime='text/csv')
            st.caption('Scenario-based forecast using assumed charging inputs, not verified future demand.')
