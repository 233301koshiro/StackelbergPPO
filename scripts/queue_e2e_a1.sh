#!/usr/bin/env bash
# 2026-09-02: A1 の学習キュー。GPU が律速なので **2 本ずつ直列**に回す。
#
# 4 本同時に投入したとき GPU 使用率が 92 % で頭打ちになり、T_update が
# 130 秒 → 290〜345 秒（2.4 倍）に伸びた。CPU は load average 4.9/20 コアで
# 空いており律速は GPU。同時実行しても総スループットは変わらず結果が遅く出るだけ。
#
# 段取り（優先順）:
#   1. 平面版 seed=0（稼働中） … pj_short/mid/long との比較用。既存実験に接続する
#   2. 縦型版 seed=0            … 発表用の「スケッチが動く」映像 + 6.4.2(1) の限界解消
#   3. 縦型版 seed=1            … 2 シード化
#   4. 関節固定 seed=0（両タスク）… 第3層の助言を閉ループで検証する（9-20）
#   5. A2 seed=0（両タスク）      … 余裕の梯子の追加点 + 18 % のずれが学習まで残るか（9-22）
#
# なぜ平面版を捨てないか: 既に走っており、pj_* との比較は縦型では成立しない
#   （形態クラスが違う）。両方あれば 9-18b の比較と非平面の証拠が両立する。
#
# 完走判定は `training done!` の有無で行う。`pgrep -f` / `pkill -f` は
# **自分のコマンドラインにマッチして誤爆する**（Bug 19。2026-09-02 に再度踏んだ）。
#
# 起動: nohup bash scripts/queue_e2e_a1.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_e2e_a1.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_e2e_a1.log
MIN_FREE_GB=20
DEADLINE=$(date -d '2026-09-05 12:00' +%s)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_p() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }

launch() {  # $1=run名 $2=seed $3=task $4=xml名
  mkdir -p "single_run/$1"
  if [ "$3" = "reach" ]; then
    EXTRA="+reward_specs.use_reach=true +reward_specs.target_x=0.8 +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false"
  else
    EXTRA="+reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true"
  fi
  # 縦型 XML は関節が1本多く depth 4 に達する。pusher_gearonly は max_body_depth=4
  # （添字 0〜3）なので get_attr_fixed() が IndexError で即死する（9/2 に 2 本これで落ちた）。
  # 縦型用の cfg は既にある（xz オフセット・床貫通チェック・depth 6）のでそれを使う。
  case "$4" in
    *v) CFG=pusher_tripo_v3
        # 床貫通ペナルティは正直な最悪リターンより悪くしないと「貫通した方が得」になる（9-14）。
        # 幾何から逆算: 肩 0.239 m・水平伸展 1.010 m・目標 (0.8,0,0.15)
        #   最遠距離 = |S-T| + 1.010 = 1.815 m → 正直な最悪 = -1815 → 1900 を採用
        # ⚠️ Pusher は正直な試行が正（0〜+270）なので既定 50 のままでよい。付けると比較が壊れる。
        [ "$3" = "reach" ] && EXTRA="$EXTRA +reward_specs.init_contact_penalty=1900" ;;
    *)  CFG=pusher_gearonly ;;
  esac
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg="$CFG" xml_name="$4" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$2" +robot_param_scale=1 $EXTRA \
    hydra.run.dir="single_run/$1" > "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, $3, xml=$4, cfg=$CFG)"
}

wait_free() {
  while true; do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    f=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -ge "$MIN_FREE_GB" ] && return 0
    log "空きメモリ ${f} GB < ${MIN_FREE_GB} GB。待機"
    sleep 300
  done
}

stage() {  # $1 $2 = 前段として完走を待つ run 名、以降 "run seed task xml"
  local a="$1" b="$2"; shift 2
  log "前段の完走を待つ: $a / $b"
  while ! (done_p "$a" && done_p "$b"); do
    [ "$(date +%s)" -ge "$DEADLINE" ] && { log "期限超過。終了"; exit 0; }
    sleep 300
  done
  log "前段が完走: $a / $b"
  for spec in "$@"; do
    set -- $spec
    # ⚠️ ディレクトリの有無だけで判定すると、起動直後にクラッシュした run が
    # 空の log を残して**永久に再試行を塞ぐ**（9/2 に縦型2本がこれで17時間止まった）。
    # log_train.txt が空でないことを「実施済み」の条件にする。
    if [ -s "single_run/$1/log/log_train.txt" ]; then log "$1 は既存。スキップ"; continue; fi
    wait_free
    launch "$1" "$2" "$3" "$4"
    sleep 60
  done
}

log "キュー開始"
stage e2e_a1_reach e2e_a1_pusher \
  "e2e_a1v_reach 0 reach e2e_a1v" \
  "e2e_a1v_pusher 0 pusher e2e_a1v"
stage e2e_a1v_reach e2e_a1v_pusher \
  "e2e_a1v_reach_s2 1 reach e2e_a1v" \
  "e2e_a1v_pusher_s2 1 pusher e2e_a1v"

# 第3段: 関節固定の助言を閉ループで検証する（実験系譜 9-20）。
#   第3層は e2e_a1 の Reach で「関節3 は可動域の 7 % しか使っていない。固定してよい」と
#   提案した。一方 Pusher では 41 % 使っており提案は出ていない。
#   **両タスクを回すのが要点**である:
#     Reach  … 固定しても性能が落ちなければ助言は有効
#     Pusher … 固定して性能が落ちれば、診断が「使う関節／使わない関節」を
#              **タスクごとに使い分けている**ことまで言える
#   片方だけでは「固定しても平気」しか言えない。
#   比較対象は e2e_a1（同 seed=0）。xml_name 以外すべて同一。
stage e2e_a1v_reach_s2 e2e_a1v_pusher_s2 \
  "e2e_a1_fix3_reach 0 reach e2e_a1_fix3" \
  "e2e_a1_fix3_pusher 0 pusher e2e_a1_fix3"

# 第4段: A2 の学習（2026-09-03 追加）。
#   A2 は余裕 33 % で A1 の 21 % より大きい。9-11 の「余裕は単調に害」が成り立つなら
#   **Reach は A1 の −6.30 より悪くなる**はず。A 群の 2 枚が偶然「余裕の梯子」に
#   なっているので、単調性の追加検証になる（9-22）。
#   もう一つの狙い: A1 と A2 は手描きが 1.01 対 1.01 でほぼ一致していたのに
#   総リーチが 18 % ずれた。**第1層の判定は両方 ✅ で吸収されたが、
#   co-design は連続量を比較するので同じ吸収が働く保証はない**。
#   学習まで揃うかを見る（4.5.3 に書いた非対称の検証）。
stage e2e_a1_fix3_reach e2e_a1_fix3_pusher \
  "e2e_a2_reach 0 reach e2e_a2" \
  "e2e_a2_pusher 0 pusher e2e_a2"

log "キュー完了: 予定していた投入をすべて終えた"
