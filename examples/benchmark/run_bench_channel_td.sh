#!/usr/bin/env bash

set -euo pipefail

DELAY_HOURS=0
DELAY_SECONDS=$(awk -v h="$DELAY_HOURS" 'BEGIN{print h * 3600}')

echo " Current time: $(data + "%Y-%m-%d %H:%M:%S")"
echo " Will run after ${DELAY_HOURS} hours (${DELAY_SECONDS} seconds)"

sleep ${DELAY_SECONDS}

echo -e "\n Time's up. Run the script"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

DESC_NAME="${1:-}"
shift || true

if [[ -z "${DESC_NAME}" ]]; then
  echo "Usage: $0 <desc> [extra-args...]"
  echo
  echo "Available desc values:"
  echo "  debug   : batch-size=32   seq-len=128    field-num=2"
  echo "  tiny    : batch-size=64   seq-len=1024   field-num=4"
  echo "  small   : batch-size=512  seq-len=12800  field-num=4"
  echo "  medium  : batch-size=1024 seq-len=65536  field-num=4"
  echo "  large   : batch-size=2048 seq-len=128000 field-num=5"
  echo "  xlarge  : batch-size=4096 seq-len=128000 field-num=5"
  echo "  huge    : batch-size=4096 seq-len=128000 field-num=10"
  echo "  all     : run all of the above in sequence"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/bench_channel_td_${DESC_NAME}_${TIMESTAMP}.log"
STATS_JSON="${LOG_DIR}/bench_channel_td_${DESC_NAME}_${TIMESTAMP}.json"
echo "Log file: ${LOG_FILE}"
echo "Stats JSON: ${STATS_JSON}"
echo

exec > >(tee "${LOG_FILE}") 2>&1

declare -a DESC_SAME=(
  "debug"
  "tiny"
  "small"
  "medium"
  "large"
  "xlarge"
)

declare -a DESC_CROSS=(
  "debug"
  "tiny"
  "small"
  "medium"
  "large"
  "xlarge"
  "huge"
)

run_one() {
  local desc="$1"
  shift || true

  local placements=("same" "cross")
  local payload_devices=("cpu" "npu")
  local matched_same=""
  for name in "${DESC_SAME[@]}"; do
    if [[ "${name}" == "${desc}" ]]; then
      matched_same="yes"
      break
    fi
  done

  for placement in "${placements[@]}"; do
    if [[ "${placement}" == "same" && -z "${matched_same}" ]]; then
      echo "[WARN] desc=${desc} is not in placement=same range (debug..xlarge), skip same."
      continue
    fi

    echo "Running bench_channel_td.py with desc='${desc}', placement='${placement}'"
    echo

    set -x
    for dev in "${payload_devices[@]}"; do
      echo "  payload-device=${dev}"
      if [[ "${dev}" == "npu" ]]; then
        # NPU may be unavailable; do not fail the whole script.
        set +e
        "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_channel_td.py" \
          --desc "${desc}" \
          --placement "${placement}" \
          "$@" \
          --stats-json-path "${STATS_JSON}" \
          --payload-device "${dev}"
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          echo "  [WARN] npu run failed with exit code ${rc}, continue."
        fi
      else
        "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_channel_td.py" \
          --desc "${desc}" \
          --placement "${placement}" \
          "$@" \
          --stats-json-path "${STATS_JSON}" \
          --payload-device "${dev}"
      fi
    done
    set +x
  done
}

case "${DESC_NAME}" in
  all)
    for entry in "${DESC_CROSS[@]}"; do
      run_one "${entry}" "$@"
    done
    exit 0
    ;;
  *)
    MATCHED=""
    for name in "${DESC_CROSS[@]}"; do
      if [[ "${name}" == "${DESC_NAME}" ]]; then
        MATCHED="yes"
        break
      fi
    done
    if [[ -z "${MATCHED}" ]]; then
      echo "Unknown desc: ${DESC_NAME}"
      echo "Valid values: debug, tiny, small, medium, large, xlarge, huge, all"
      exit 1
    fi
    run_one "${DESC_NAME}" "$@"
    ;;
esac
