#!/bin/bash
# 四脚 quad_walk_ra235_rab2 を広い床（200 m 四方）で学習し直す 1 本と、その見張り（系譜 9-342 の追記「広い床での取り直し」、2026-11-09。四脚の遊び、修論に入れない）。
#   ユーザー判断 2026-11-09「投入して」。queue_quad_walk.sh（quad_walk_ra_s0）との違いは床だけ:
#   CNOID_FLOOR_BODY=assets/choreonoid/bodies/floor_wide/floor_wide.body（Bug 100。内蔵の床は 10 m 四方で、前の run は x = 5 m の床の端で話が終わっていた）。
#   cfg・XML・seed・エポック・他の物理の環境変数は quad_walk_ra_s0 と同じ。起動は前と同じく main から。
#
# 見張りの変更（queue_quad_walk.sh から 1 つ）: ⛔ のうち「足先のめり込み」「体の原点が床の下」は 9-342 で既に見つかっていて原因が分かっていない
#   （広い床でも足先の中央値 −8〜−20 mm）。前の run はこれで ep20 に見張りが終わり、完走後の best の再生が回らなかった。
#   ⇒ この 2 つだけはログに書いて見張りを続ける。それ以外の ⛔（nan・跳ねた・ギア比・再生の失敗・異常終了）は前と同じく見張りを終える。
#   ⚠️ どちらの場合も学習は止めない（止めるのは Claude かユーザーが ⛔ を読んで、PID と hydra.run.dir を確かめてから。CLAUDE.md §6）。
# 始める時期: 走っている学習が 2 本未満になったら（9-344 の 2 本のどちらかが終わったら）。
#   本数は実行ファイル名で数える（Bug 101）。
#
# 事前登録の読み（queue_quad_walk.sh と同じ文。結果を見る前に書いた）:
#   ① 歩けた: 完走後の best を 5 話再生し、5 話とも高さ・傾きで終わらず（打ち切りの 1000 step まで立つ）、
#      250 step（10 s）での胴の x の進みの中央値が +0.37 m を超える（台本の速歩の最大 |Δx| 0.369 m を超える）
#   ② 立つが進まない: 5 話とも倒れないが、①の進みに届かない
#   ③ 倒れる: 高さ・傾きで終わる話がある
#   ⚠️ ⑤-2: 途中の epoch の値で「歩けない」とは言わない。途中の再生は異常（バグ）を見つけるためだけに使う
#   ⚠️ 足先のめり込み（9-342）が残っていれば、①でも「物理の上で足が床に入った歩き」と書き添える
#
# 出力: single_run/queue_quad_walk_wide.log、single_run/quad_walk_ra_wide_s0/trace_mid/ep<N>.{npz,txt,png}
#   nohup setsid bash scripts/queue_quad_walk_wide.sh > /dev/null 2>&1 < /dev/null &
#   tail single_run/queue_quad_walk_wide.log
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_quad_walk_wide.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
RUN=quad_walk_ra_wide_s0
D=single_run/$RUN
FLOOR=assets/choreonoid/bodies/floor_wide/floor_wide.body
# 物理の環境変数は quad_walk_ra_s0 と同じ（自己衝突は切る: 4 本の脚の付け根が胴の中心から出ていて重なる）＋ 広い床
PHYS="CNOID_CULLING_DEPTH=1.0 CNOID_CTRL_CLIP=1 CNOID_HINGE_DAMP_TORQUE=1 CNOID_JOINT_MAX_VEL=5.97 CNOID_JOINT_MAX_VEL_STRICT=1 CNOID_TICK_SYNC=1 CNOID_SUBSTEPS=4 CNOID_JOINT_LIMIT_TORQUE=1 CNOID_FLOOR_BODY=$FLOOR"

if [ -d "$D/log" ]; then
  log "$RUN は既にある。起動しない（見張りだけ続ける）"
else
  [ -f "$FLOOR" ] || { log "⛔⛔ 床の .body が無い: $FLOOR。起動しない"; exit 3; }
  log "=== 四脚の学習 1 本（広い床）を待つ（git $(git rev-parse --short HEAD)、走っている学習 $(njobs) 本）"
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。起動せずに止まる"; exit 3; fi
  mkdir -p $D
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $PHYS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=quad_walk_ra xml_name=quad_walk_ra235_rab2 seed=0 max_epoch_num=200 num_threads=4 enable_wandb=false fix_skeleton=true \
    hydra.run.dir=$D > $D/stdout.log 2>&1 &
  log "$RUN launched (PID $!、git $(git rev-parse --short HEAD))"
  sleep 60
  n=$(njobs); [ "$n" -gt 2 ] && log "⚠️ 学習が $n 本になった（他のキューと同時に枠を取った）"
  grep -q "CNOID_FLOOR_BODY" $D/launch_env.json 2>/dev/null && log "✅ launch_env.json に CNOID_FLOOR_BODY" || log "⛔⛔ launch_env.json に CNOID_FLOOR_BODY が無い（再生が内蔵の床で走る）"
  grep -q "可動域をトルクで掛ける: 8 関節" $D/stdout.log && log "✅ 可動域のトルク 8 関節" || log "⛔⛔ 可動域のトルクの行が無い"
  grep -q "4 分割" $D/stdout.log && log "✅ 刻み 4 分割" || log "⛔⛔ 刻みの分割の行が無い"
fi

# 見張り: 新しい checkpoint ごとに再生 → 図と判定。学習のプロセスが消えたら（完走でも異常終了でも）best を再生して終わる
KNOWN='足先が床より 5 mm 以上下|体の原点が床より下'
while true; do
  alive=$(ps -eo args | grep "[c]horeonoid_train.py" | grep -cE "hydra.run.dir=$D( |$)")   # 起動行の最後なので行末も
  last=$(tail -1 $D/log/log_train.txt 2>/dev/null | python3 -c 'import sys; print(sys.stdin.read().rstrip()[:160])')
  if grep -aqiE "\bnan\b|\binf\b" <(grep -a "T_sample" $D/log/log_train.txt 2>/dev/null); then
    log "⛔ log_train.txt に nan / inf: $last"; log "⛔ 止めて調べる（見張りは終わる。学習は止めていない）"; exit 3
  fi
  for E in $(ls $D/models/epoch_*.p 2>/dev/null | sed 's/.*epoch_0*//; s/\.p$//' | sort -n); do
    T=$D/trace_mid/ep$(printf %04d $E)
    [ -f $T.txt ] && continue
    mkdir -p $D/trace_mid
    EVAL_RESTORE_DIR=$D EVAL_CHECKPOINT=$E TRACE_OUT=$T.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
      timeout -k 30 1800 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py > $T.log 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f $T.npz ]; then
      echo "再生に失敗（exit $rc）" > $T.txt
      log "⛔ ep$E 再生に失敗（exit $rc、$T.log の末尾: $(tail -2 $T.log | tr '\n' ' ' | cut -c1-200)）"; exit 3
    fi
    python3 scripts/plot_quad_trace.py $D $T.npz > $T.txt 2>&1
    v=$(grep -a "判定:" $T.txt | sed 's/\[quad\] //')
    log "ep$E $(grep -a 'x の進み: 中央値' $T.txt | sed 's/\[quad\] //') ｜ $(grep -a '250 step（10 s）' $T.txt | sed 's/\[quad\] //') ｜ $v"
    if echo "$v" | grep -q "⛔"; then
      # ⛔ の項目（" / " 区切り）から既知の 2 つを除いて、残りがあれば見張りを終える
      rest=$(echo "$v" | sed 's/^判定: ⛔ //; s/  ⚠️.*//' | tr '/' '\n' | grep -vE "$KNOWN" | grep -v '^ *$')
      if [ -n "$rest" ]; then
        log "⛔ ep$E で異常。止めて調べる（見張りは終わる。学習は止めていない）。図: $T.png"; exit 3
      fi
      log "⚠️ ep$E の ⛔ は既知のめり込み（9-342）だけ。見張りを続ける"
    fi
  done
  if [ "$alive" -eq 0 ] && [ -d "$D/log" ]; then
    if grep -aq "training done" $D/log/log_train.txt; then
      mkdir -p $D/trace
      EVAL_RESTORE_DIR=$D EVAL_CHECKPOINT=best USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
        timeout -k 30 1800 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py > $D/trace/trace.log 2>&1
      python3 scripts/plot_quad_trace.py $D > $D/trace/quad_view.txt 2>&1
      log "=== 完走。best: $(grep -a '判定:\|250 step（10 s）' $D/trace/quad_view.txt | sed 's/\[quad\] //' | tr '\n' ' ')。図: $D/trace/quad_view.png"
      exit 0
    fi
    log "⛔ 学習のプロセスが無いのに training done が無い（異常終了）。stdout の末尾: $(tail -3 $D/stdout.log | tr '\n' ' ' | cut -c1-300)"
    exit 3
  fi
  sleep 60
done
