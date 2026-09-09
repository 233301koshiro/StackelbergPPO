#!/usr/bin/env bash
# 2026-09-09: 壁つきホッケー（実験系譜 9-55 の設計、パック位置は 9-59 で 0.55 へ）。
#
# **問い**: 板で直線経路を塞いだとき、**側壁で反射させてゴールへ入れることを学習できるか。**
#   9-59 の測定で、パックの初期 |y| ≤ 0.20 では**直線でゴールに入る角度が存在しない**。
#   `cube_y_noise=0.4`（一様）なら**半分強のエピソードが反射を要求する**。
#
# ⚠️ **前段は 9-52（発表の主結論の seed=1）。そちらを優先する。**
#   9-52 は 09-11 の発表に直接効くので、完走を待ってから投入する。
#
# ⚠️ **1 本で足りる**（9-55）。切り分けが run 内で完結するため:
#   学習後に `probe_shot_aim.py` でパックの y を −0.4〜+0.4 に固定して到達最大 x を測れば、
#   「腕が悪いのか、戦略を学べていないのか」が読める（9-56 の方法論）。
#
# ⚠️ **旧 `queue_hockey.sh` とは別物。** あちらは 9-50 で畳んだ C（`e2e_hockey_court`、壁なし）で、
#   成立しない条件を待って 09-11 12:00 に自動終了する。**こちらは壁つき `e2e_hockey_wall`。**
#
# 起動: nohup bash scripts/queue_hockey_wall.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_hockey_wall.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_hockey_wall.log
XML=e2e_hockey_wall
DEADLINE=$(date -d '2026-09-12 12:00' +%s)

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_(){ grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
alive_(){ ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"; }
ckpt_(){ local n; n=$(ls "single_run/$1"/models/epoch_*.p 2>/dev/null \
        | sed 's|.*/epoch_||; s|\.p$||' | sort -n | tail -1); [ -n "$n" ] && echo $((10#$n)); }

# Bug 35: 「やったか / やっていないか」の2値で判定しない。4状態に分ける。
launch(){  # $1=run名 $2=seed
  local R=""
  if done_ "$1"; then log "$1 は完走済み。スキップ"; return; fi
  if alive_ "$1"; then log "$1 は稼働中。スキップ"; return; fi
  if [ -s "single_run/$1/log/log_train.txt" ]; then
    local C; C=$(ckpt_ "$1")
    if [ -n "$C" ]; then R="+restore_dir=single_run/$1 epoch=$C"
      log "$1 は epoch $C まで進んで止まっている → そこから再開する"
    else local D="single_run/$1.dead_$(date +%m%d%H%M)"; mv "single_run/$1" "$D"
      log "$1 は checkpoint 無し → $D へ退避し最初から"
    fi
  fi
  mkdir -p "single_run/$1"
  # 設定の出所: 9-55（ctrl_cost 0.001 の再導出まで済み）。⚠️ 勝手に変えない。
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_tripo_v3 xml_name="$XML" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$2" +robot_param_scale=1 \
    +env_specs.cube_y_noise=0.4 +reward_specs.ctrl_cost_coeff=0.001 \
    +env_specs.arm_safe_init=true +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 ${R} \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, xml=$XML${R:+, $R})"
}

log "キュー開始（壁つきホッケー、XML=$XML）。前段 = 9-52 の 4 本"
while ! { done_ e2e_a1_reach_s2 && done_ e2e_a1_fix3_reach_s2 \
       && done_ e2e_a1_pusher_s2 && done_ e2e_a1_fix3_pusher_s2; }; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
  sleep 180
done
log "前段（9-52 の 4 本）が完走"
launch hockey_wall 0
log "投入完了"
