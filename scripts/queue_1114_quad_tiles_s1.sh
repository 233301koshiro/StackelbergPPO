#!/bin/bash
# 四脚 quad_walk_ra235_rab2 を 0.5 m 角の箱の床で、seed 1 で学習する 1 本と、その見張り（系譜 9-348・読み 55。四脚の遊び、修論に入れない）。
#   ユーザー判断 2026-11-13「暇なら四脚」→ 案の 3 つから「seed 1」。quad_walk_ra_tiles_s0（seed 0）との違いは seed だけ。
# ⭐ なぜ: seed 0 は「10 s に +4.74 m」歩いた（系譜 9-342 の 11-11）。それが seed 0 だけのものかを、変数を足さずに確かめる。
#   床の沈み込み（Bug 103、seed 0 の best で 20 話のうち 6 話）は、この run では解かない（seed を変えても解けない）。
# ⭐ 起動行は seed 0 の run の `.hydra/overrides.yaml` と `launch_env.json` をそのまま読む（queue_1113_push_alloc_s45.sh と同じ。手で書き写さない）。
#   床の .body（git の外、make_floor_tiles.py の既定の出力）は 11-10 15:39 に書き直されているが、既定の引数で書いた物とバイト単位で同じで、
#   make_floor_tiles.py は seed 0 の起動（11-10 08:09）の前の 07:55 から変わっていない（11-13 に確かめた）。
#   11-10 08:09 以降に学習・env のコードの変更は無い（git log で確かめた）。
# ⭐ 始める時期: いま走っている押しの配分のキュー（queue_1113_push_alloc_s45.sh、系譜 9-347、修論の核）の bash が終わってから、さらに学習が 2 本未満になったら。
#   本数だけで待つと、s4 の 2 本が終わった瞬間に _s5 の枠を取り合う（11-06 の枠の取り合い）。だから先にキューの bash の終わりを待つ。
#   見込み: 11-14 の朝に起動、約 14 時間。
# ⭐ 学習の本数は実行ファイル名が choreonoid のものだけを数える（Bug 101）。ディスクの空きが 3 GB を切ったら起動しない。
#
# ⭐ 事前登録した読み 55（結果を見てから変えない。系譜 9-348 に同じ文）:
#   前提: 完走（200 epoch、NaN 0）・ワーカーの種 [1001, 1002, 1003, 1004]・launch_env.json に床・可動域のトルク 8 関節・刻み 4 分割・check_before_conclusion.py・目視（目視記録に 1 行）。
#   判定は seed 0 と同じ ①〜③（best の 5 話、学習と同じ箱の床）:
#     ① 歩けた: 5 話とも高さ・傾きで終わらず、250 step（10 s）での胴の x の進みの中央値が +0.37 m を超える
#     ② 立つが進まない: 5 話とも倒れないが、①の進みに届かない
#     ③ 倒れる: 高さ・傾きで終わる話がある
#   55-a: ① → 「箱の床の四脚は 2 seed とも歩いた」と書く（床の接触がときどき抜ける物理での歩き、の但し書きは残す）。
#   55-b: ② か ③ → 「seed 0 は歩き、seed 1 は歩かなかった」と書く。seed の読み方（研究方針 原則 4）に従い、
#         「歩ける」は seed 0 で言えたまま、seed 1 の失敗は「歩けない」の根拠にしない（学習の不足の例として扱う）。原因はこの run では書かない。
#   ⚠️ ⑤-2: 途中の epoch の値で「歩けない」とは言わない。
#   添え書き（判定に使わない。seed 0 と同じ数え方: best を同じ床で 20 話、胴の x が 21 m 未満の step）:
#     「2 mm 以上下の (step, 脛) が 200 件を超える話」（seed 0: 6 / 20）・「2 m 未満しか進まない話」（同 0 / 20）・「1000 step 前に傾き・高さで終わる話」（同 1 / 20）。
#   予想（判定に使わない）: 55-a（seed 0 は ep16 で 579.7、ep50 で 1447.8 と早く伸びた）。
#
# 出力: single_run/queue_1114_quad_tiles_s1.log、single_run/quad_walk_ra_tiles_s1/trace_mid/ep<N>.{npz,txt,png}、
#       完走後 trace/quad_view.{png,txt}（best 5 話）と trace/arm_trace_20.npz（best 20 話、添え書き用）
#   nohup setsid bash scripts/queue_1114_quad_tiles_s1.sh > /dev/null 2>&1 < /dev/null &
set -u
cd /userdir/StackelbergPPO
Q=single_run/queue_1114_quad_tiles_s1.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $Q; }
njobs(){ ps -eo comm,args 2>/dev/null | awk '$1=="choreonoid" && /choreonoid_train\.py/' | wc -l; }
RUN=quad_walk_ra_tiles_s1
SRC=quad_walk_ra_tiles_s0
SEED=1
D=single_run/$RUN
WAIT_PID=981140   # queue_1113_push_alloc_s45.sh の bash（11-13 14:39 起動）

if [ -d "$D/log" ]; then
  log "$RUN は既にある。起動しない（見張りだけ続ける）"
else
  for f in single_run/$SRC/.hydra/overrides.yaml single_run/$SRC/launch_env.json; do
    [ -f "$f" ] || { log "⛔⛔ $f が無い。起動しない"; exit 3; }
  done
  ARGS=$(sed -n 's/^- //p' single_run/$SRC/.hydra/overrides.yaml | sed "s/^seed=.*/seed=$SEED/" | tr '\n' ' ')
  ENVS=$(python3 -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('single_run/$SRC/launch_env.json')).items()))")
  FLOOR=$(python3 -c "import json;print(json.load(open('single_run/$SRC/launch_env.json')).get('CNOID_FLOOR_BODY',''))")
  [ -n "$FLOOR" ] && [ -f "$FLOOR" ] || { log "⛔⛔ 床の .body が無い: '$FLOOR'。起動しない"; exit 3; }
  log "=== 四脚 seed 1（箱の床）を待つ（系譜 9-348・読み 55、git $(git rev-parse --short HEAD)、走っている学習 $(njobs) 本）"
  if grep -aq "queue_1113_push_alloc_s45" /proc/$WAIT_PID/cmdline 2>/dev/null; then
    log "押しの配分のキュー（PID $WAIT_PID）の bash が終わるのを待つ"
    while grep -aq "queue_1113_push_alloc_s45" /proc/$WAIT_PID/cmdline 2>/dev/null; do sleep 60; done
    log "押しの配分のキューの bash が終わった（走っている学習 $(njobs) 本）"
  fi
  while [ "$(njobs)" -ge 2 ]; do sleep 60; done
  AV=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$AV" -lt 3 ]; then log "⛔⛔ ディスクの空き ${AV} GB（3 GB 未満）。起動せずに止まる"; exit 3; fi
  mkdir -p $D
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 $ENVS \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    $ARGS hydra.run.dir=$D > $D/stdout.log 2>&1 &
  log "$RUN launched (PID $!、git $(git rev-parse --short HEAD))  元 $SRC  env: $ENVS  args: $ARGS"
  sleep 60
  n=$(njobs); [ "$n" -gt 2 ] && log "⚠️ 学習が $n 本になった（他のキューと同時に枠を取った）"
  want="[$((1000*SEED+1)), $((1000*SEED+2)), $((1000*SEED+3)), $((1000*SEED+4))]"
  s=$(grep -ao 'ワーカーの乱数の種 \[[0-9, ]*\]' $D/stdout.log | head -1)
  case "$s" in *"$want"*) log "✅ 種: $s";; *) log "⛔⛔ 種の行が期待（$want）と違う／無い: '$s'";; esac
  grep -q "CNOID_FLOOR_BODY" $D/launch_env.json 2>/dev/null && log "✅ launch_env.json に CNOID_FLOOR_BODY" || log "⛔⛔ launch_env.json に CNOID_FLOOR_BODY が無い（再生が内蔵の床で走る）"
  grep -q "可動域をトルクで掛ける: 8 関節" $D/stdout.log && log "✅ 可動域のトルク 8 関節" || log "⛔⛔ 可動域のトルクの行が無い"
  grep -q "4 分割" $D/stdout.log && log "✅ 刻み 4 分割" || log "⛔⛔ 刻みの分割の行が無い"
fi

# 見張り（queue_quad_walk_tiles.sh と同じ）: 新しい checkpoint ごとに再生 → 図と判定。学習のプロセスが消えたら（完走でも異常終了でも）best を再生して終わる
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
      # 添え書き用の 20 話（判定には使わない）
      EVAL_RESTORE_DIR=$D EVAL_CHECKPOINT=best TRACE_EPISODES=20 TRACE_OUT=$D/trace/arm_trace_20.npz USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
        timeout -k 30 3600 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/record_arm_trace.py > $D/trace/trace_20.log 2>&1
      log "best の 20 話: exit $?（$D/trace/arm_trace_20.npz）"
      exit 0
    fi
    log "⛔ 学習のプロセスが無いのに training done が無い（異常終了）。stdout の末尾: $(tail -3 $D/stdout.log | tr '\n' ' ' | cut -c1-300)"
    exit 3
  fi
  sleep 60
done
