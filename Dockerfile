FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEPOTWATT_ENV=production \
    DEPOTWATT_STATE_DIR=/state
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py config.py data_processing.py evaluation.py feature_engineering.py forecasting.py model_loader.py optivolt_lstm.py ./
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data/processed ./data/processed
COPY data/sample_processed_data.csv ./data/sample_processed_data.csv
# Files required by the integrated legacy LSTM evaluation and report pages.
COPY UKPN_OptimisePrime_*.csv ./
COPY model_comparison_metrics.csv heldout_predictions_all_models.csv lstm_24h_recursive_forecast.csv ./
COPY models ./model_seed
COPY .streamlit ./.streamlit
COPY docker-entrypoint.sh /usr/local/bin/optivolt-entrypoint
RUN chmod +x /usr/local/bin/optivolt-entrypoint \
    && useradd --create-home appuser \
    && mkdir -p /state /app/models /app/outputs \
    && chown -R appuser:appuser /state /app/models /app/outputs /app
USER appuser
VOLUME ["/state", "/app/models"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["/usr/local/bin/optivolt-entrypoint"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
