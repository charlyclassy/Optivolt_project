#!/bin/sh
set -eu
mkdir -p /app/models/global_model /state/uploads
if [ ! -f /app/models/global_model/xgboost_model.json ]; then
    echo "Initialising model volume from the bundled model files..."
    cp -a /app/model_seed/. /app/models/
fi
if [ ! -f /app/models/global_model/xgboost_model.json ]; then
    echo "ERROR: Active model package is missing" >&2
    exit 1
fi
exec "$@"
