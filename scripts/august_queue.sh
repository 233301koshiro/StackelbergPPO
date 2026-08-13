#!/usr/bin/env bash
# 2026-08-10 〜 08-15 の無人運転キュー（常駐）
#
# 【背景】8/10 に Bug 23（スケール XML 生成器が body pos を伸ばさず、実効リーチが
# 公称の 6 割しかなかった）が発覚し、軸3 の主結果 `tripo_pjr_min` 系が無効になった。
# 修正版の生成器で作り直した XML で**下限検証をやり直す**のが最優先。
#
# 【設計】合計スループットは並走本数によらずほぼ一定（実測 1 epoch ≒ 80 秒、
# T_update が本数に比例して伸びる = 更新フェーズが 20 コアを食い切る）。
# したがって「何本並べるか」ではなく「何エポック投入するか」で予算を考える。
# 8/10 22:50 → 8/15 朝 = 約 106 時間 ≒ 4,700 epoch が上限。
#
#   確定消費: pjp_long_1000 残 159 + pjp_mid_1000 1000（experiment_queue.sh が自動投入）
#   本キュー: 1000（v3 Reach） + 600（下限やり直し） + 800（seed 穴埋め） + 400（余裕5%）
#
# 長い run（v3 Reach 1000ep）は**壁時計で最も長くかかる**ので最優先で投入し、
# 短い run をその周りで回す。優先度の低い末尾2本は締切ゲートで自然に落ちる。
#
# 状態: single_run/august_queue.log
set -u
cd /userdir/StackelbergPPO

LOG=single_run/august_queue.log
MAXJOBS=4                          # 同時学習数。1 run 約 9 GB・メモリ 62 GB
MEM_MIN=10                         # 空きメモリ(GB)がこれ未満なら投入しない
# 200ep はこれ以降投入しない。2026-08-13 に 12:00 → 8/14 02:00 へ延長:
# v3_reach 停止で 1 枠空き、現行レート（約 8 ep/h）なら 8/14 02:00 起動でも 8/15 未明に完走する。
SHORT_STOP="2026-08-14 02:00:00"
HARD_STOP="2026-08-14 06:00:00"    # これ以降は一切投入しない

# ⚠️ Bug 19 対策: pgrep -f は AI や人間が打ったシェルコマンド自身にも当たる。
# run 名の末尾を $ で固定し、hydra.run.dir= を含む学習プロセスだけを見る。
running() { pgrep -f "hydra\.run\.dir=single_run/$1\$" >/dev/null 2>&1; }
finished() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }
njobs()   { pgrep -f "[c]horeonoid_train\.py cfg=" | wc -l; }
freemem() { free -g | awk '/^Mem:/{print $7}'; }
now()     { date '+%F %T'; }
past()    { [ "$(date +%s)" -ge "$(date -d "$1" +%s)" ]; }

# Reach（到達）系の共通フラグ。tripo_pj_* / tripo_pjd_* と完全に揃えてある
REACH_FLAGS="+reward_specs.use_reach=true +reward_specs.target_x=0.8 \
+reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
+reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"

# Pusher（押し）系の共通フラグ。tripo_pjp_* と完全に揃えてある
PUSH_FLAGS="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
+reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"

launch() {  # launch <run名> <xml> <ep数> <seed> <cfg> <フラグ群>
  local run=$1 xml=$2 ep=$3 seed=$4 cfg=$5; shift 5
  mkdir -p "single_run/$run"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window \
    --python scripts/choreonoid_train.py \
    cfg="$cfg" xml_name="$xml" num_threads=4 max_epoch_num="$ep" \
    enable_wandb=false fix_skeleton=true seed="$seed" +robot_param_scale=1 \
    "$@" hydra.run.dir="single_run/$run" \
    > "single_run/$run/stdout.log" 2>&1 &
  echo "[$(now)] $run launched (PID $!, ${ep}ep, xml=$xml, seed=$seed)"
}

# 優先度順。上から順に、空きが出るたびに投入する。
# 形式: 名前|エポック数|種別
QUEUE=(
  # ⛔ 2026-08-13 除外: tripo_v3_reach は 153 epoch（初回 ep0-108 + 再投入後 ep0-45）を通じて
  #    exec_R_eps が −50.00（床貫通ペナルティ）から一度も動かず、学習が成立していなかった。
  #    ETA 15 日で 8/15 までに終わる見込みも無く、CPU の 25% を占めるだけだったため停止。
  #    原因（目標高さ z=0.15 が縦型アームを床方向へ引く疑い）は未確定。再挑戦は要スモーク。
  "tripo_pjr199|200|reach199"          # 【Bug 23 やり直し】助言下限 1.99倍（実効 0.8022 m）seed=0
  "tripo_pjr199_s2|200|reach199s2"     # 同 seed=1
  "tripo_pjr211p|200|push211"          # 【Bug 23 やり直し】Pusher 助言下限 2.11倍（実効 0.8505 m）
  "tripo_pjd_prox_s2|200|proxs2"       # 配分判別の seed=1（実験系譜 9-9 の順位断定に必要）
  "tripo_pjd_dist_s2|200|dists2"       # 同上
  "tripo_pjp_mid_s2|200|pmids2"        # 反転（タスク識別）Pusher 側の seed=1
  "tripo_pjp_long_s2|200|plongs2"      # 同上
  "tripo_pjr209|200|reach209"          # 余裕 5.3%（必要余裕の中間点、4.4.3 の限界）
  "tripo_pjr209_s2|200|reach209s2"     # 同 seed=1
)

start_one() {  # start_one <種別> <run名> <ep数>
  case $1 in
    # 非平面 v3 の Reach。⚠️ 目標は 0.72 m（余裕約 10.7%）。
    # v3 の水平到達限界は 0.797 m しかないので、標準の 0.8 m だと余裕 0% になり
    # Bug 23 で無効化された下限条件と同じ「届かない実験」になってしまう。
    v3reach)   launch "$2" tripo_arm_v3 "$3" 0 pusher_tripo_v3 \
                 +reward_specs.use_reach=true +reward_specs.target_x=0.72 \
                 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
                 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false ;;
    reach199)  launch "$2" tripo_arm_v2c_pj_rec199 "$3" 0 pusher_gearonly $REACH_FLAGS ;;
    reach199s2) launch "$2" tripo_arm_v2c_pj_rec199 "$3" 1 pusher_gearonly $REACH_FLAGS ;;
    push211)   launch "$2" tripo_arm_v2c_pj_rec211 "$3" 0 pusher_gearonly $PUSH_FLAGS ;;
    proxs2)    launch "$2" tripo_arm_v2c_pj_prox "$3" 1 pusher_gearonly $REACH_FLAGS ;;
    dists2)    launch "$2" tripo_arm_v2c_pj_dist "$3" 1 pusher_gearonly $REACH_FLAGS ;;
    pmids2)    launch "$2" tripo_arm_v2c_pj_mid "$3" 1 pusher_gearonly $PUSH_FLAGS ;;
    plongs2)   launch "$2" tripo_arm_v2c_pj_long "$3" 1 pusher_gearonly $PUSH_FLAGS ;;
    reach209)  launch "$2" tripo_arm_v2c_pj_rec209 "$3" 0 pusher_gearonly $REACH_FLAGS ;;
    reach209s2) launch "$2" tripo_arm_v2c_pj_rec209 "$3" 1 pusher_gearonly $REACH_FLAGS ;;
    *) echo "[$(now)] 未知の種別: $1" ;;
  esac
}

{
  echo "[$(now)] august_queue 開始 (pid $$)。予定 ${#QUEUE[@]} 本・同時 $MAXJOBS 本まで"
  while :; do
    if past "$HARD_STOP"; then
      echo "[$(now)] HARD_STOP 到達。新規投入を終了する"; break
    fi

    remaining=0
    for item in "${QUEUE[@]}"; do
      IFS='|' read -r run ep kind <<< "$item"
      finished "$run" && continue
      running "$run"  && { remaining=$((remaining+1)); continue; }
      remaining=$((remaining+1))

      # 200ep は締切を過ぎたら投入しない（中途半端な run は主張に使えない）
      if [ "$ep" -le 200 ] && past "$SHORT_STOP"; then
        echo "[$(now)] $run: SHORT_STOP 超過のため見送り"; continue
      fi

      # 2026-08-13: 長尺保護ゲート（tripo_v3_reach が ep700 を越えるまで pjr209 を待たせる）は撤去。
      # 守るべき長尺そのものが学習不成立で停止したため、ゲートは pjr209 を永久に塞ぐだけになった。
      # **「保護対象が消えたら保護も外す」** — 条件付きゲートは条件の前提が消えたときに害だけ残る。

      jobs_now=$(njobs); mem_now=$(freemem)
      if [ "$jobs_now" -ge "$MAXJOBS" ]; then break; fi
      if [ "$mem_now" -lt "$MEM_MIN" ]; then
        echo "[$(now)] 空きメモリ ${mem_now} GB < ${MEM_MIN} GB → 待機"; break
      fi

      echo "[$(now)] 稼働 ${jobs_now} 本・空き ${mem_now} GB → $run を投入"
      start_one "$kind" "$run" "$ep"
      sleep 120        # 起動直後の負荷が落ち着くまで待ってから次を判断する
      break
    done

    [ "$remaining" -eq 0 ] && { echo "[$(now)] 全 run 完走。キュー終了"; break; }
    sleep 300
  done
} >> "$LOG" 2>&1
