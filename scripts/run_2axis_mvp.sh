#!/usr/bin/env bash
# run_2axis_mvp.sh: 2軸アーム MVP 学習パイプライン
#
# topology.json → MuJoCo XML / Choreonoid .body 生成 → Stackelberg PPO 学習
#
# 使い方:
#   bash run_2axis_mvp.sh                          # デフォルト設定で実行
#   ENGINE=mujoco bash run_2axis_mvp.sh            # MuJoCo のみ
#   DRY_RUN=1 bash run_2axis_mvp.sh               # ファイル生成のみ（学習なし）
#   TOPOLOGY=path/to/my_arm.json bash run_2axis_mvp.sh  # 別トポロジー
#
# 環境変数:
#   TOPOLOGY        topology.json のパス（デフォルト: rrbot_topology.json）
#   ENGINE          mujoco | choreonoid | both（デフォルト: choreonoid）
#   NUM_THREADS     並列スレッド数（デフォルト: 4）
#   MAX_EPOCHS      最大エポック数（デフォルト: 1000）
#   OUTPUT_BASE     出力先ベースディレクトリ（デフォルト: single_run）
#   ENABLE_WANDB    true | false（デフォルト: false）
#   DRY_RUN         1 = ファイル生成のみ、学習なし（デフォルト: 0）

set -euo pipefail

# ── パラメータ ────────────────────────────────────────────────────────────
TOPOLOGY="${TOPOLOGY:-data/rrbot_description/rrbot_topology.json}"
ENGINE="${ENGINE:-choreonoid}"
NUM_THREADS="${NUM_THREADS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
OUTPUT_BASE="${OUTPUT_BASE:-single_run}"
ENABLE_WANDB="${ENABLE_WANDB:-false}"
DRY_RUN="${DRY_RUN:-0}"

# ── プロジェクトルートに移動 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── トポロジー JSON からロボット名を導出 ─────────────────────────────────
TOPO_BASE="$(basename "${TOPOLOGY}" .json)"
XML_NAME="${TOPO_BASE}"
XML_OUT="assets/mujoco_envs/${XML_NAME}.xml"
BODY_OUT_DIR="assets/choreonoid/bodies/${XML_NAME}"

echo "========================================"
echo " 2-axis arm MVP pipeline"
echo "  topology : ${TOPOLOGY}"
echo "  engine   : ${ENGINE}"
echo "  xml_name : ${XML_NAME}"
echo "========================================"

# ── Step 1: MuJoCo XML 生成 ──────────────────────────────────────────────
echo ""
echo "=== Step 1: topology.json → MuJoCo XML ==="
python3 scripts/topology_to_xml.py \
    --topology "${TOPOLOGY}" \
    --output   "${XML_OUT}" \
    --validate
echo "  → ${XML_OUT}"

# ── Step 2: Choreonoid .body ファイル生成 ─────────────────────────────────
echo ""
echo "=== Step 2: topology.json → Choreonoid .body ==="
mkdir -p "${BODY_OUT_DIR}"
python3 scripts/dynamic_body_updater.py \
    --topology  "${TOPOLOGY}" \
    --output-dir "${BODY_OUT_DIR}"
echo "  → ${BODY_OUT_DIR}/"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "=== DRY RUN: モデルファイル生成完了。学習はスキップ。==="
    exit 0
fi

# ── Step 3: 学習 ─────────────────────────────────────────────────────────
run_mujoco() {
    local out_dir="${OUTPUT_BASE}/mvp_${TOPO_BASE}_mujoco"
    echo ""
    echo "=== Step 3a: MuJoCo training → ${out_dir} ==="
    python3 design_opt/train.py \
        cfg=pusher \
        xml_name="${XML_NAME}" \
        num_threads="${NUM_THREADS}" \
        max_epoch_num="${MAX_EPOCHS}" \
        enable_wandb="${ENABLE_WANDB}" \
        fix_skeleton=true \
        "hydra.run.dir=${out_dir}"
}

run_choreonoid() {
    local out_dir="${OUTPUT_BASE}/mvp_${TOPO_BASE}_cnoid"
    echo ""
    echo "=== Step 3b: Choreonoid training → ${out_dir} ==="
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window \
        --python scripts/choreonoid_train.py \
        cfg=pusher \
        xml_name="${XML_NAME}" \
        num_threads="${NUM_THREADS}" \
        max_epoch_num="${MAX_EPOCHS}" \
        enable_wandb="${ENABLE_WANDB}" \
        fix_skeleton=true \
        "hydra.run.dir=${out_dir}"
}

case "${ENGINE}" in
    mujoco)     run_mujoco ;;
    choreonoid) run_choreonoid ;;
    both)       run_mujoco; run_choreonoid ;;
    *)
        echo "ERROR: ENGINE は mujoco | choreonoid | both のいずれかを指定してください" >&2
        exit 1
        ;;
esac

echo ""
echo "=== Done: ${ENGINE} training completed ==="
