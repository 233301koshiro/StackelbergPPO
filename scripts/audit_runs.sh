#!/usr/bin/env bash
# audit_runs.sh: 全 run の一次データ（.hydra/overrides.yaml + log_train.txt）から
# ground truth 表を生成する。md の実験記録と定期的に突き合わせるための監査ツール。
#
# 使い方: bash scripts/audit_runs.sh
# 出力: run名 | 最終ep | 最終exec_R_eps | best報酬 | 設定（overrides）
#
# 背景: 2026-07-10 の全数監査で「F7 のリンク数・scale」「scale=0.1 の形態可動性」
# 「B の fix_skeleton」等、md 側の設定記述の転記ミスが複数見つかった。
# ログと overrides は嘘をつかない — 実験記録を書く前にこの表を見ること。
set -u
cd "$(dirname "$0")/.."

for d in single_run/*/ single_run/archive/*/; do
  [ -f "$d/.hydra/overrides.yaml" ] || continue
  name=$(basename "$d")
  ov=$(tr '\n' ' ' < "$d/.hydra/overrides.yaml" | sed 's/- //g')
  log="$d/log/log_train.txt"
  if [ -f "$log" ]; then
    last=$(grep -E 'T_sample' "$log" | tail -1)
    lastep=$(echo "$last" | grep -oE '^\[[^]]+\] [0-9]+' | grep -oE '[0-9]+$')
    laster=$(echo "$last" | grep -oE 'exec_R_eps [-0-9.]+' | awk '{print $2}')
    best=$(grep -oE 'rewards [-0-9.]+' "$log" | tail -1 | awk '{print $2}')
  else
    lastep="-"; laster="-"; best="-"
  fi
  echo "$name | ep=$lastep | last_exec=$laster | best=$best | $ov"
done 2>/dev/null | sed 's/cfg=pusher //; s/xml_name=rrbot_arm //; s/num_threads=4 //; s/enable_wandb=false //; s|hydra.run.dir=\S*||'
