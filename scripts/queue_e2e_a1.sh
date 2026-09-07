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
#   5. A2v seed=0（両タスク）     … 18 % のずれが学習まで残るか（9-22）。**縦型**
#   6. B2v seed=0（両タスク）     … 縦型の「余裕の梯子」の 3 点目
#
# ⚠️ **2026-09-04: 平面での新規投入をやめた。** スケッチは 4 枚とも縦に立った腕なので
#   **縦型が本来の形**である（9-25）。平面を残すのは「平面でなければ成立しない検証」だけ:
#     - `e2e_a1`（完走）      … pj_short/mid/long との比較。pj_* が平面なので縦型では不可
#     - `e2e_a1_fix3`（稼働中）… 第3層の助言は平面 A1 の診断から出た。比較対象も平面 A1
#   それ以外（旧・第5段の平面 `e2e_a2`）は**取り下げて縦型 `e2e_a2v` に置き換えた**。理由は下記。
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
DEADLINE=$(date -d '2026-09-09 12:00' +%s)

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
done_p() { grep -q "training done!" "single_run/$1/log/log_train.txt" 2>/dev/null; }

# その run の学習プロセスが今も生きているか。
# ⚠️ パターンではなく hydra.run.dir で照合する。`[c]` で grep 自身にマッチさせない（Bug 19）。
# 末尾を ( |$) で閉じないと e2e_a2v_reach が e2e_a2v_reach_s2 にも当たる。
alive_p() {
  ps -eo args 2>/dev/null | grep -qE "[c]horeonoid_train\.py.*hydra\.run\.dir=single_run/$1( |\$)"
}

# models/epoch_NNNN.p の最大値。途中再開の epoch 指定に使う
latest_ckpt() {
  local n
  n=$(ls "single_run/$1"/models/epoch_*.p 2>/dev/null \
      | sed 's|.*/epoch_||; s|\.p$||' | sort -n | tail -1)
  [ -n "$n" ] && echo $((10#$n))
}

# 縦型 Reach の床貫通ペナルティ。**XML ごとに違う**（9-14 の原則）。
# 正直な最悪リターンより悪くしないと「貫通した方が得」になる。
#   最遠距離 = |肩 − 目標| + 肩から先の腕の長さ、正直な最悪 = −1000 × 最遠距離
# 目標は (0.8, 0, 0.15) 固定。肩の高さと腕の長さは diagnose_morphology.py が出す。
#   e2e_a1v: 肩 0.239 / 腕 1.010 → |S−T| 0.805 + 1.010 = 1.815 → 1900
#   e2e_a2v: 肩 0.336 / 腕 1.196 → |S−T| 0.821 + 1.196 = 2.017 → 2100
#   e2e_b2v: 肩 0.102 / 腕 1.227 → |S−T| 0.801 + 1.227 = 2.028 → 2100
# ⚠️ **1900 を他の縦型へ流用しないこと。** 腕が長いほど正直な最悪が下がるので足りなくなる。
# ⚠️ Pusher は正直な試行が正（0〜+270）なので既定 50 のまま。付けると比較が壊れる。
floor_penalty() {
  case "$1" in
    e2e_a1v) echo 1900 ;;
    e2e_a2v) echo 2100 ;;
    e2e_b2v) echo 2100 ;;
    *) echo "" ;;   # 未登録の縦型は下で落とす
  esac
}

launch() {  # $1=run名 $2=seed $3=task $4=xml名 $5=追加引数（途中再開用、省略可）
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
        if [ "$3" = "reach" ]; then
          PEN=$(floor_penalty "$4")
          if [ -z "$PEN" ]; then
            log "$1 中止: $4 の床貫通ペナルティが未登録。floor_penalty() に追加すること"
            return 1
          fi
          EXTRA="$EXTRA +reward_specs.init_contact_penalty=$PEN"
        fi ;;
    *)  CFG=pusher_gearonly ;;
  esac
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid \
    --no-window --python scripts/choreonoid_train.py \
    cfg="$CFG" xml_name="$4" num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed="$2" +robot_param_scale=1 $EXTRA ${5:-} \
    hydra.run.dir="single_run/$1" >> "single_run/$1/stdout.log" 2>&1 &
  log "$1 launched (PID $!, seed=$2, $3, xml=$4, cfg=$CFG${5:+, $5})"
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
    # ⚠️ **「ログがあるか」の2値で判定すると必ずどこかで固まる。** 4 状態に分ける。
    #   ① 完走済み            → スキップ
    #   ② 稼働中              → スキップ（キューを二重起動しても二重投入しない）
    #   ③ 途中で死んだ+ckptあり → **そこから再開**（コンテナ停止がこれ。Bug 35）
    #   ④ 途中で死んだ+ckptなし → ログを退避して最初から
    # ②③④を「既存」で一括スキップしていたため、③でキューが期限まで無言で待ち続けた。
    RESUME=""
    if done_p "$1"; then
      log "$1 は完走済み。スキップ"; continue
    elif alive_p "$1"; then
      log "$1 は稼働中。スキップ"; continue
    elif [ -s "single_run/$1/log/log_train.txt" ]; then
      CK=$(latest_ckpt "$1")
      if [ -n "$CK" ]; then
        # `reset_epoch` は付けない（epoch カウンタを継続させる。引き継ぎ書 §2 の再開手順）
        RESUME="+restore_dir=single_run/$1 epoch=$CK"
        log "$1 は epoch $CK まで進んで止まっている → そこから再開する"
      else
        D="single_run/$1.dead_$(date +%m%d%H%M)"
        mv "single_run/$1" "$D"
        log "$1 は途中で死んで checkpoint も無い → $D へ退避し最初から投入する"
      fi
    fi
    wait_free
    launch "$1" "$2" "$3" "$4" "$RESUME"
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

# 第5段: A2 を **縦型**で学習する（2026-09-04 に平面から差し替え）。
#   狙いは 9-22 の②「手描きが 1.01 対 1.01 で一致していたのに XML は 18 % ずれた。
#   第1層は両方 ✅ で吸収したが、co-design は連続量なので同じ吸収が働く保証はない」の検証。
#
#   ⚠️ **平面 A2 ではなく縦型 A2v にした理由が 2 つある。**
#   ① **平面 A1 は seed が 1 本しかなく、18 % の差を seed のばらつきと区別できない。**
#      縦型 A1v は 2 seed 揃っている（Reach −14.96 / −16.46、Pusher 256.19 / 258.02）ので、
#      **A2v の差がそのばらつきを超えるかで判定できる**。比較として厳密に強い。
#   ② 当初のもう一つの狙い「余裕の梯子の 4 点目」は**そもそも成立しない**。
#      A1 を梯子に数えてはいけない理由（手描き由来で配分も総リーチも違う）は
#      A2 にもそのまま当てはまる。梯子は pj_short/mid/long で閉じている。
stage e2e_a1_fix3_reach e2e_a1_fix3_pusher \
  "e2e_a2v_reach 0 reach e2e_a2v" \
  "e2e_a2v_pusher 0 pusher e2e_a2v"

# 第6段: B2 を縦型で学習する。**縦型だけで「余裕の梯子」を作る**ための 3 点目。
#   A1v 20 % → A2v 32 % → B2v 35 %。平面の梯子（pj_*、10/45 %）とは別系統だが、
#   **こちらは 3 点とも手描き由来**なので「描いた絵の余裕が co-design に効くか」を
#   同じ土俵で見られる。⚠️ 32 % と 35 % は近いので、差が出なくても否定にはならない。
#
#   ⚠️ **B1v は学習しない。** 第1層で棄却（0.680 m、1.30 倍必要）されており、
#   **「学習せずに落とせる」ことがシステムの主張**である。走らせると主張が崩れる。
stage e2e_a2v_reach e2e_a2v_pusher \
  "e2e_b2v_reach 0 reach e2e_b2v" \
  "e2e_b2v_pusher 0 pusher e2e_b2v"

# 第7段: A2v・B2v の seed 1（2026-09-07 追加）。
#   9-29 で Pusher が 348（A1v）→ 151（A2v）→ 184（B2v）と**逆転**したが、
#   **A2v・B2v はどちらも 1 seed** なので、逆転が形態由来か seed 由来か分けられない。
#   A1v の seed 幅は 8.17 しかないので、逆転（33 の差）が本物なら seed では説明できない。
#   ⚠️ **Pusher を先に回す。** Reach は 3 点が単調（-13.97 → -14.64 → -19.67）で
#   差も小さく、seed を足しても読みは変わりにくい。差が大きく逆転しているのは Pusher。
stage e2e_b2v_reach e2e_b2v_pusher \
  "e2e_a2v_pusher_s2 1 pusher e2e_a2v" \
  "e2e_b2v_pusher_s2 1 pusher e2e_b2v"

# 第8段: Reach 側の seed 1。単調性の確認（優先度は下）。
stage e2e_a2v_pusher_s2 e2e_b2v_pusher_s2 \
  "e2e_a2v_reach_s2 1 reach e2e_a2v" \
  "e2e_b2v_reach_s2 1 reach e2e_b2v"

log "キュー完了: 予定していた投入をすべて終えた"
