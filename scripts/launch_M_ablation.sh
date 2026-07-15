#!/usr/bin/env bash
# Launch M1/M2 ablation experiments (200ep x 2 seeds)
set -euo pipefail

ROOT=$(dirname "$(realpath "$0")")/..
cd "$ROOT"

RUNS=("single_run/rrbot_arm_pusher_M1_lenonly" "single_run/rrbot_arm_pusher_M2b_gearonly")
SEEDS=(0 1)

for run in "${RUNS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    outdir="${run}_s${seed}"
    echo "=== Launching ${outdir} ==="
    mkdir -p "$outdir"
    cp -r "$run/.hydra" "$outdir/.hydra" || true
    # write seed override
    printf "- seed=%s\n" "$seed" > "$outdir/.hydra/overrides_seed.yaml"
    # Build command (uses existing launcher script if present)
    cmd=("bash" "single_run/L2_launcher.sh" "$outdir" )
    echo "DRY-RUN: to actually launch: ${cmd[*]}"
    echo "Saved overrides in ${outdir}/.hydra/overrides_seed.yaml"
  done
done

echo "Prepared ${#RUNS[@]} runs x ${#SEEDS[@]} seeds. To start training, run the launcher commands above." 
