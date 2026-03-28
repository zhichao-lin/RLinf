export RLINF_LOG_LEVEL=DEBUG

SCRIPT_NAME=${1:-channel_example.py}
SCRIPT_BASENAME=$(basename "$SCRIPT_NAME" .py)

mkdir -p logs
timestamp=$(date "+%Y%m%d_%H%M%S")
python "$SCRIPT_NAME" 2>&1 | tee logs/${SCRIPT_BASENAME}_${timestamp}.log
