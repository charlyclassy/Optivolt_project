"""Unified Optivolt demand workspace; delegates modelling to existing modules."""
import html
import streamlit as st
import depot_updates as dep

SOURCES = ('Use existing UKPN depot data', 'Upload new depot demand data', 'Update existing depot demand data')


def render(selected=None):
    st.markdown('''<div style="background:linear-gradient(110deg,#173b77,#087e89);padding:19px 24px;border-radius:17px;margin-bottom:12px;box-shadow:0 7px 20px #12345a22">
      <div style="color:white;font-size:29px;font-weight:800;line-height:1.2">⚡ Demand Prediction Workspace</div>
      <div style="color:#e4f7ff;font-size:15px;margin-top:5px">Select data → Prepare → Choose method → Predict</div></div>''',unsafe_allow_html=True)
    st.markdown('### 1 · Select data source')
    source=st.radio('Data source', SOURCES, key='dw_data_source', label_visibility='collapsed')
    names=dep.visible_depots()
    if source != SOURCES[0]:
        # Reuse the existing, tested upload / update and version-control functions.
        # The upload form owns validation, timestamp merging, and explicit model activation.
        st.info('Upload and save the readings below. Uploading does not automatically retrain or replace a saved model.')
        dep.render_upload(default_mode='Add new depot' if source == SOURCES[1] else 'Update existing depot', embedded=True)
        st.divider()
        st.markdown('### 2 · Select a depot for prediction')
    else:
        st.markdown('### 2 · Choose depot and prediction method')
    default = selected if selected in names else (names[0] if names else None)
    if not names:
        st.warning('No depot data is available. Upload a depot dataset first.')
        return
    depot=st.selectbox('Depot',names,index=names.index(default),key='dw_forecast_depot')
    if source != SOURCES[0]:
        st.caption('After saving a new depot, select it here. Newly uploaded depots need a compatible trained model before forecasting.')
    st.markdown('### 3 · Demand prediction')
    try:
        model, scaler=dep.active_paths(depot)
        if not model.exists() or not scaler.exists():
            st.warning('No compatible saved LSTM is available for this depot. Train and activate a model before forecasting.')
            return
        dep.render_fast_forecast(depot)
    except Exception as exc:
        st.error(f'Prediction unavailable: {exc}')
