#!/usr/bin/env bash
# run_tripo_pipeline.sh: Tripo3D GLB → MuJoCo XML まで一発で通す
#
# 使い方:
#   bash scripts/run_tripo_pipeline.sh <GLB> <OUT_DIR> <XML_NAME>
#
# 例:
#   bash scripts/run_tripo_pipeline.sh \
#     data/my_arm/arm.glb \
#     data/my_arm \
#     my_arm
#
# 出力:
#   <OUT_DIR>/meshes/link_0.stl, link_1.stl, ...  (link-local STL)
#   <OUT_DIR>/<XML_NAME>.urdf                       (Choreonoid 可視化用)
#   <OUT_DIR>/topology.json                         (Stackelberg 形式)
#   assets/mujoco_envs/<XML_NAME>.xml               (RL 学習用)
#
# オプション環境変数:
#   JOINT_COLOR="255 0 255"   マーカー色 RGB（デフォルト: magenta）
#   JOINT_TOL=40              色検出 tolerance
#   JOINTS=""                 手動関節 Z 位置（例: "-0.07 0.277"）
#   LINK_NAMES=""             リンク名（例: "upper_arm forearm hand"）
#   RANGES=""                 関節可動域（例: "-60 60 -90 90 -45 45"）
#   GEARS=""                  アクチュエータゲイン（例: "150 100 80"）

set -euo pipefail

GLB="${1:?第1引数に GLB ファイルパスを指定してください}"
OUT_DIR="${2:?第2引数に出力ディレクトリを指定してください}"
XML_NAME="${3:-tripo_arm}"

JOINT_COLOR="${JOINT_COLOR:-255 0 255}"
JOINT_TOL="${JOINT_TOL:-40}"
JOINTS="${JOINTS:-}"
LINK_NAMES="${LINK_NAMES:-}"
RANGES="${RANGES:-}"
GEARS="${GEARS:-}"

echo "=========================================="
echo "  run_tripo_pipeline.sh"
echo "  GLB     : $GLB"
echo "  OUT_DIR : $OUT_DIR"
echo "  XML_NAME: $XML_NAME"
echo "=========================================="

# ----------------------------------------------------------------
# Step 1: GLB → per-link STL (link-local) + Choreonoid URDF
# ----------------------------------------------------------------
echo ""
echo "[Step 1] GLB → STL + URDF"

STEP1_ARGS=(
  --glb        "$GLB"
  --out-dir    "$OUT_DIR/meshes"
  --urdf       "$OUT_DIR/${XML_NAME}.urdf"
  --joint-color $JOINT_COLOR
  --joint-tol  "$JOINT_TOL"
)
[ -n "$JOINTS"     ] && STEP1_ARGS+=(--joints     $JOINTS)
[ -n "$LINK_NAMES" ] && STEP1_ARGS+=(--names       $LINK_NAMES)
# LINK_ROT="name1 deg1 [name2 deg2 ...]" 形式（例: LINK_ROT="hand 90"）
if [ -n "${LINK_ROT:-}" ]; then
  read -ra LR_VALS <<< "$LINK_ROT"
  for ((i=0; i<${#LR_VALS[@]}; i+=2)); do
    STEP1_ARGS+=(--link-rot "${LR_VALS[i]}" "${LR_VALS[i+1]}")
  done
fi

python3 scripts/glb_to_links.py "${STEP1_ARGS[@]}"

# ----------------------------------------------------------------
# Step 2: STL → topology.json (Stackelberg 形式)
# ----------------------------------------------------------------
echo ""
echo "[Step 2] STL → topology.json"

# mesh_to_params の parts は「可動リンクのみ」（各 part に関節が1つ対応する）。
# FIXED_BASE=1 のとき、最下段セグメント（固定台座）を Step 2 から除外する。
# LINK_NAMES 指定時は Step 1 の STL が <name>.stl で出力されるため、その順序を使う。
MP_NAMES=""
if [ -n "$LINK_NAMES" ]; then
  read -ra NAME_ARR <<< "$LINK_NAMES"
  START=0
  [ "${FIXED_BASE:-0}" = "1" ] && START=1
  PARTS=""
  for ((i=START; i<${#NAME_ARR[@]}; i++)); do
    PARTS+=" $OUT_DIR/meshes/${NAME_ARR[i]}.stl"
    MP_NAMES+=" ${NAME_ARR[i]}"
  done
else
  PARTS=$(ls "$OUT_DIR/meshes"/link_*.stl 2>/dev/null | sort) || true
  if [ "${FIXED_BASE:-0}" = "1" ]; then
    PARTS=$(echo "$PARTS" | tail -n +2)
  fi
fi
if [ -z "${PARTS// /}" ]; then
  echo "ERROR: $OUT_DIR/meshes に STL が見つかりません" >&2
  exit 1
fi

STEP2_ARGS=(
  --parts    $PARTS
  --output   "$OUT_DIR/topology.json"
  --validate
)
[ -n "$MP_NAMES" ] && STEP2_ARGS+=(--names $MP_NAMES)
[ -n "$GEARS"      ] && STEP2_ARGS+=(--gears $GEARS)

# --ranges は "lo hi lo hi ..." を "lo hi" "lo hi" ... に変換
if [ -n "$RANGES" ]; then
  R_ARGS=()
  read -ra R_VALS <<< "$RANGES"
  for ((i=0; i<${#R_VALS[@]}; i+=2)); do
    R_ARGS+=("${R_VALS[i]} ${R_VALS[i+1]}")
  done
  STEP2_ARGS+=(--ranges "${R_ARGS[@]}")
fi

python3 scripts/mesh_to_params.py "${STEP2_ARGS[@]}"

# ----------------------------------------------------------------
# Step 3: topology.json → MuJoCo XML
# ----------------------------------------------------------------
echo ""
echo "[Step 3] topology.json → MuJoCo XML"

python3 scripts/topology_to_xml.py \
  --topology "$OUT_DIR/topology.json" \
  --output   "assets/mujoco_envs/${XML_NAME}.xml" \
  --validate

# ----------------------------------------------------------------
# 完了メッセージ
# ----------------------------------------------------------------
echo ""
echo "=========================================="
echo "  完了！"
echo ""
echo "  Choreonoid 可視化:"
echo "    choreonoid --urdf $OUT_DIR/${XML_NAME}.urdf"
echo ""
echo "  RL 学習:"
echo "    choreonoid --no-window --python scripts/choreonoid_train.py \\"
echo "      cfg=pusher xml_name=${XML_NAME} \\"
echo "      max_epoch_num=200 num_threads=4 enable_wandb=false \\"
echo "      fix_skeleton=true +robot_param_scale=1 \\"
echo "      hydra.run.dir=single_run/${XML_NAME}_run"
echo "=========================================="
