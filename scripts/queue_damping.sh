#!/usr/bin/env bash
# 2026-09-26: **関節ダンピングの感度分析**（助教の指摘3 の残り／実験系譜 9-171 予定）。
#
# 修論 6.4.2 (4) がこう書いている:
#   「打撃的方策の高い先端速度は現在の関節ダンピング係数の設定に依存している可能性があり、
#     この一点については実機相当の設定で同じ方策が現れるかを確認していない」
# 密度・ギア・リンク長は 9-146 等で実測して答えた。⛔ **残っているのはダンピングだけ。**
#
# ⭐ **投入前に計算した（9-171 の冒頭）**: `damping=1` は無視できる値ではない。
#   打撃のピーク（関節の絶対角速度 95〜108 rad/s）でダンピングトルクは **駆動上限 150 N·m の 61〜72 %** に達する。
#   ⚠️ 絶対角速度なので関節速度の上限側の見積もりだが、桁として一次のパラメータである。
#   ⭐ **だから 0.1 倍と 10 倍で挟む**: 0.1 なら明確に無視でき、10 なら明確に支配する。
#
# 条件（変数は damping だけ。XML の差分は 2 行＝default joint の damping のみ。検算済み）:
#   damping=0.1  e2e_a1v_damp01_pusher / _s2
#   damping=1    **既存**（`e2e_a1v_real_pusher` / `_s2`。9-146）→ 回さない
#   damping=10   e2e_a1v_damp10_pusher / _s2
#
# ⚠️ **タスクは Pusher のみ。**6.4.2 (4) の主張は打撃的方策＝ Pusher の現象なので、
#   そこに絞る。Reach まで広げると 8 本になり、問いに対して過剰である。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① **3 点とも打撃的方策が出る**（先端のピーク速度が同じ桁）
#        → ⭐⭐ **打撃はダンピング設定の産物ではない。**6.4.2 (4) の但し書きを弱められる
#   ② **b=10 で打撃が消える**（ピーク速度が 1 桁落ちる／cube の移動が激減）
#        → ⭐ **但し書きは正しかった。**⭐ **「どの程度のダンピングまで成立するか」を数字で書ける**
#   ③ **b=0.1 と b=1 で差が無く、b=10 だけ落ちる**
#        → ⭐ ②の一種。**現在値 1 は「効いていないが、あと一桁で効き始める」境界の手前**という読み
#   ⚠️ **報酬の帯だけで判定しない。**打撃かどうかは **先端のピーク速度と cube の移動量**で見る
#     （`fwd_cube` は平均なので 1 発当てる方策ではほぼ 0 になる。9-135・9-43 で 2 回誤読した）。
#
# ⚠️ **結論前に `check_before_conclusion.py`（6 項目）と `plot_run.py` を回し、目視記録へ 1 行書く。**
#
# 起動: nohup bash scripts/queue_damping.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_damping.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 XML=$2 S=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_real xml_name=$XML num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false \
    +reward_specs.init_contact_penalty=1900 +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, xml=$XML, seed=$S)"; sleep 60; }

log "=== ダンピング感度（Pusher・各 2 seed）開始"
if ! python3 scripts/audit_xml_reach.py \
      assets/mujoco_envs/e2e_a1v_real_damp01.xml \
      assets/mujoco_envs/e2e_a1v_real_damp10.xml >> "$LOG" 2>&1; then
  log "⛔ XML の公称/実効リーチが食い違う。中止（Bug 23 と同型）"; exit 1; fi
log "✅ XML 検算を通過"

launch e2e_a1v_damp01_pusher    e2e_a1v_real_damp01 0
launch e2e_a1v_damp10_pusher    e2e_a1v_real_damp10 0
launch e2e_a1v_damp01_pusher_s2 e2e_a1v_real_damp01 1
launch e2e_a1v_damp10_pusher_s2 e2e_a1v_real_damp10 1
log "=== キュー終了"
