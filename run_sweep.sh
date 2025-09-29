#!/usr/bin/env bash
set -euo pipefail

ENTITY="rich-white285-university-of-nevada-reno"
PROJECT="reproduce_mega_descriptor"
SWEEP_CONFIG="./sweep.yaml"

# Create the sweep and capture BOTH stdout and stderr
# CREATE_OUT="$(wandb sweep --entity "$ENTITY" --project "$PROJECT" "$SWEEP_CONFIG" 2>&1)"
CREATE_OUT="$(wandb sweep --entity "$ENTITY" --project "$PROJECT" "$SWEEP_CONFIG" 2>&1 | tee /dev/stderr)"

echo "$CREATE_OUT"

# Prefer parsing the "Run sweep agent with:" line
SWEEP_PATH="$(sed -n 's/.*wandb agent \([^[:space:]]\+\).*/\1/p' <<< "$CREATE_OUT" | tail -n1)"

# Fallback: parse the ID and construct entity/project/<id>
if [[ -z "${SWEEP_PATH:-}" ]]; then
  SWEEP_ID="$(sed -n 's/.*Creating sweep with ID: \([A-Za-z0-9]\+\).*/\1/p' <<< "$CREATE_OUT" | tail -n1)"
  if [[ -n "${SWEEP_ID:-}" ]]; then
    SWEEP_PATH="$ENTITY/$PROJECT/$SWEEP_ID"
  fi
fi

if [[ -z "${SWEEP_PATH:-}" ]]; then
  echo "Failed to parse sweep path from wandb output:"
  echo "$CREATE_OUT"
  exit 1
fi

echo "Created sweep: $SWEEP_PATH"

# Ensure agent context is right (not strictly necessary when using full path)
export WANDB_ENTITY="$ENTITY"
export WANDB_PROJECT="$PROJECT"

exec wandb agent "$SWEEP_PATH"
