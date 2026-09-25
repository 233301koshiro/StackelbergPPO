#!/usr/bin/env bash
# 2026-09-25: **山場2（助言の閉ループ）を実在条件で検証する**（実験系譜 9-169 予定）。
#
# 9-164 で「実寸条件では第3層が助言を出さない」と分かった。完走 4 本すべてで
# 関節使用率の最小が 14 % で、閾値 10 % を下回る関節が一つも出なかった。
# ⛔ **したがって閉ループを検証する対象そのものが存在しない状態だった。**
#
# ⭐ **だが 9-119 は「使用率 12 % の関節を固定したら 2 シードとも良化した」と示している。**
#   ＝ **閾値 10 % が良い助言を取り逃している。**修論 6.4.2 (8) にも見落としの実例として書いてある。
#   したがって正確には「実寸では遊ぶ関節が無い」ではなく「**現在の閾値では助言が出ない**」。
#
# そこで閾値を下げずに、**実測で最小だった関節を直接固定して**閉ループが閉じるかを見る。
#   対照: e2e_a1v_real_reach / _s2（既にある。ギア 20〜150・密度 294.2 の実寸条件）
#   提案: e2e_a1v_real_fix4_reach / _s2（**関節4 を固定。リンク長・太さ・質量は完全に同一**）
#
# ⚠️ **変数は 1 つだけ。**`xml_name` の `_fix4` 接尾のみ。他のフラグは対照と文字どおり同一
#   （.hydra/overrides.yaml で照合済み）。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ① **固定側の帯が対照と重ならず、固定側が上**
#        → ⭐⭐ **実在条件でも閉ループが閉じる。しかも閾値 10 % が取り逃していた助言である。**
#          9-119（旧条件・使用率 12 %）と合わせ、**閾値の較正不足が 2 条件で再現**したことになる
#   ② **帯が重なる**
#        → **「遜色ない」までしか言えない。**⭐ 助言の採否手続き（3.12.5）は
#          「帯が重なるなら採用しない」なので、**手続きとしては正しく動いている**と読む
#   ③ **固定側が明確に悪い（帯が分離して下）**
#        → ⛔ **使用率 14 % の関節は実際に要る。**閾値 10 % はむしろ妥当だったことになり、
#          9-119 との食い違いを形態の違いで説明する必要が出る
#
# ⚠️ **①②③ のどれでも修論に書く価値がある。**②は「助言が出ない」ことの裏づけになる。
# ⚠️ **結論前に `check_before_conclusion.py`（6 項目）と `plot_run.py` を回し、目視記録へ 1 行書く。**
#
# 起動: nohup bash scripts/queue_real_fix4.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_real_fix4.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_real_fix4.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 S=$2
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3_real xml_name=e2e_a1v_real_fix4 num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 \
    +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 \
    +env_specs.check_init_contact=false +reward_specs.init_contact_penalty=1900 \
    +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!)"; sleep 60; }

log "=== キュー開始（実寸条件で関節4 を固定・2 seed）"
if ! python3 scripts/audit_xml_reach.py assets/mujoco_envs/e2e_a1v_real_fix4.xml >> "$LOG" 2>&1; then
  log "XML の公称/実効リーチが食い違う。投入を中止する（Bug 23 と同型）"; exit 1
fi
log "XML 検算を通過"
launch e2e_a1v_real_fix4_reach 0
launch e2e_a1v_real_fix4_reach_s2 1
log "=== キュー終了"
