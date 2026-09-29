#!/usr/bin/env bash
# ⭐⭐⭐ 結合テスト（第2版）— **Bug 50 修正後の判定器で棄却される形態**で閉ループを試す。
#
# ⚠️ **第1版（系譜 9-186）は成立しなかった。**Bug 50 を直した結果 `e2e_b1` が
# 棄却されなくなり、「棄却した形態 対 助言に従った形態」の比較ではなくなった。
#
# ⭐ 形態（`close_loop.py` が人手ゼロで生成）
#   e2e_cl_short      = e2e_b1 の 0.6 倍  リーチ 0.4735 m → ❌ 棄却
#   e2e_cl_short_fix  = 助言どおり 1.8 倍  リーチ 0.8522 m → ⭐ 通過
#
# ⭐⭐⭐ **助言後の形態は `e2e_b1_p108` と物理が完全に同一**（0.6 × 1.8 = 1.08。
#   XML の差はモデル名の 1 行だけ）。⭐ **9-186 で 2 seed 走っているのでそれを使う。**
#   ⭐ **投入するのは棄却側の 2 本だけ**（約 8.5 GPU 時間。17 → 8.5 に半減）。
#
# ⭐⭐ **Bug 51 を踏まえて助言が 1.59 → 1.80 倍になった。**
#   対象は毎話 ±0.1 m 揺れるので、**棄却は最も有利な位置・助言は最も不利な位置**で計算する。
#   ⭐ 1.80 倍の実効到達 0.9522 m は、対象が最も奥（1.100）でも打点 0.850 m に届く。
#
# ⭐ cfg は `pusher_gearonly`（長さ凍結）。⭐⭐ **Bug 50 修正後の判定器は太さの探索上限
#   0.10 を勘定するので、判定と学習が同じ設計空間を見ている**（9-152 の条件を満たす）。
#
# ⛔⛔ **判定は二値で行う。**9-186 が示したとおり**対象の移動量は話ごとに 250 % 揺れる**ので、
#   移動量の大小では順位を付けられない。⭐ **指標は「cube が 0.01 m 以上動いた話の数 / 5 話」。**
#   cube は腕以外に動く原因が無いので、動いた＝触れた、が物理的に確実である
#   （⚠️ 接触そのものは frame_skip の分解能で判定できない。Bug 48）。
#
# ⭐ 事前登録した読み（**結果を見てから変えない**）
#   ① short 0/5・fix が過半      → ⭐⭐⭐ **閉ループが閉じた**。第1層の棄却も実証
#   ② short が 1 話でも動かす     → ⛔⛔ **棄却がまた外れた。**Bug 50 の修正が不十分
#   ③ short 0/5 だが fix も不安定 → ⭐⭐ **Bug 51 の補正でも足りない。**
#                                   ⭐ 対象が奥に来た話で何が起きているかを軌跡で見る
#   ④ 両方が安定して動かす        → ⚠️ 判定の解像度が足りない。打点を遠ざけて測り直す
#
# ⚠️ **fix 側（= e2e_b1_p108）の実測は既にある**: seed0 [3.88, 4.54, 6.59]、
#   seed1 [0.21, 0.44, 6.31]。⛔ **seed1 は 2/3 話で 0.5 m 未満**なので、
#   ⭐ **読み ① が出るとすれば「short が 0/3、fix が 4/6 話」といった形になる。**
#   ⚠️ **これは事前に書いておく。結果を見てから基準を動かさない。**
#
# ⚠️ **完走後に `check_before_conclusion.py`（6 項目）と `plot_run.py`。目視記録へ 1 行。**
# ⚠️ **環境変数は env の引数として渡す**（Bug 49）。
#
# 起動: nohup bash scripts/queue_closeloop2.sh > /dev/null 2>&1 & disown
# 進捗: single_run/queue_closeloop2.log
set -u
cd /userdir/StackelbergPPO
LOG=single_run/queue_closeloop2.log
: > "$LOG"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
njobs(){ ps -eo args | grep -c "[c]horeonoid_train.py"; }

launch(){ local RUN=$1 XML=$2 S=$3
  if [ -d "single_run/$RUN/log" ]; then log "$RUN は既にある。飛ばす"; return; fi
  while [ "$(njobs)" -ge 2 ]; do sleep 180; done
  mkdir -p "single_run/$RUN"
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/choreonoid_train.py \
    cfg=pusher_gearonly xml_name=$XML num_threads=4 max_epoch_num=200 \
    enable_wandb=false fix_skeleton=true seed=$S +robot_param_scale=1 \
    +reward_specs.ctrl_cost_coeff=0.2 +reward_specs.contact_weight=0 \
    +reward_specs.init_contact_penalty=50 +env_specs.arm_safe_init=true \
    hydra.run.dir=single_run/$RUN > single_run/$RUN/stdout.log 2>&1 &
  log "$RUN launched (PID $!, xml=$XML, seed=$S)"; sleep 60; }

log "=== 結合テスト第2版（棄却側 2 本。助言側は e2e_b1_p108 を流用）開始"
launch cl2_short_pusher      e2e_cl_short  0
launch cl2_short_pusher_s2   e2e_cl_short  1
log "=== キュー終了"
