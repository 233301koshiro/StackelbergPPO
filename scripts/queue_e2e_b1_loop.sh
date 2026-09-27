#!/usr/bin/env bash
# 2026-09-27: ⭐⭐⭐ **結合テスト — パイプライン出力で閉ループを閉じる**（実験系譜 9-186 予定）。
#
# ⭐⭐ **設計書が挙げた空白そのもの**（[結合テスト_見込み違いの形態.md](../docs/研究応用/設計/結合テスト_見込み違いの形態.md)）:
#   9-19: B1（手描き → M1 → M2 → M3 由来）が第1層で棄却され「N 倍に伸ばせ」が出た。
#         ⛔ **その助言に従った先が無い**（B1 は意図的に学習させていない）。
#   9-11: 不適合形態に 1.99 倍を適用して −396.92 → −0.33。
#         ⛔ **だが形態は手書きの XML。パイプラインの出力ではない。**
#   ⭐⭐⭐ **つまり「パイプラインが出した不適合形態を、パイプラインの助言で直して、
#        達成まで持っていった」例が 1 つも無い。**本キューがそれを埋める。
#
# 形態（`close_loop.py` が生成。人手ゼロ）:
#   `e2e_b1`      総リーチ **0.7891 m** … ❌ 棄却（対象は 0.850 m 先）
#   `e2e_b1_p108` 総リーチ **0.8522 m** … ⭐ 通過（助言どおり **1.08 倍**。余裕の上乗せなし）
#
# ⚠️⚠️ **cfg は `pusher_gearonly`（`body_params: {}` ＝ リンク長を凍結）。**
#   ⛔ **9-152 の教訓**: 「棄却側だけが確実」は**判定と学習が同じ設計空間を見ているときにしか成り立たない**。
#   ⛔ `pusher_tripo_v3` は長さを最適化するので、**9-151 では棄却した配置が 2.1 mm まで到達した**。
#   ⭐ **判定はリンク長を凍結して行っているので、学習も凍結する。**
#
# ⚠️ **変数は形態だけ。**他のフラグは 2 条件で同一。
#
# 読み方（**結果を見る前に決める**。§5-2 ⑤-2）:
#   ⭐ ① **B1 が未到達・B1_p108 が到達**
#        → ⭐⭐⭐ **パイプライン出力で閉ループが閉じた。**
#          **第1層の「棄却側だけが確実」（9-78）も同時に実証される**
#   ⛔ ② **B1 も到達してしまう**
#        → ⛔⛔ **第1層の保証が反証される。**⚠️ **重大。**修論 3.12・5.5.2 の書き換えが要る。
#          ⭐ **それでも記録する**（§4-2: 撤回した実験は開示義務）
#   ⚠️ ③ **どちらも未到達**
#        → ⚠️ **助言が足りない。**1.08 倍では不十分だったことになる。
#          ⭐ Bug 22（倍率の切り捨て）の再発を疑い、必要倍率を測り直す
#
# ⚠️ **報酬だけで判定しない。**再生して ①対象に触れたか ②どこまで押したか を軌跡で見る。
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
#
# 起動: nohup bash scripts/queue_e2e_b1_loop.sh > /dev/null 2>&1 & disown
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_e2e_b1_loop.log
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
launch(){ local RUN=$1 XML=$2 S=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  # ⚠️ 環境変数は env の引数として渡す（Bug 49）
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, xml=$XML, seed=$S)"; sleep 60; }

log "=== 結合テスト（パイプライン出力の閉ループ・各 2 seed）開始"
if ! python3 scripts/audit_xml_reach.py \
      assets/mujoco_envs/e2e_b1.xml assets/mujoco_envs/e2e_b1_p108.xml >> "$LOG" 2>&1; then
  log "⛔ XML の公称/実効リーチが食い違う。中止（Bug 23）"; exit 1; fi
log "✅ XML 検算を通過"

launch e2e_b1_pusher       e2e_b1      0
launch e2e_b1_p108_pusher  e2e_b1_p108 0
launch e2e_b1_pusher_s2    e2e_b1      1
launch e2e_b1_p108_pusher_s2 e2e_b1_p108 1
log "=== キュー終了"
