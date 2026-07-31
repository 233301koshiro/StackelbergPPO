#!/usr/bin/env bash
# viewer(eval_cnoid_viewer.py)起動と同時にChoreonoidウィンドウを画面録画する。
#
# ChoreonoidのMovieRecorderはC++側にしか存在せずPythonバインディングが無い
# （src/Base/pybind11/PyBaseModule.cppにMovie/Record関連の登録なしを確認済み、
# 2026-07-31）ため、スクリプトからは制御できない。かわりにOS側のffmpeg
# x11grabでウィンドウを外部から録画する。
#
# 使い方:
#   VIEWER_RESTORE_DIR=single_run/<run> [VIEWER_EPOCH=best VIEWER_EPISODES=3 ...] \
#   bash scripts/record_cnoid_viewer.sh [出力mp4パス]
#
# 依存: ffmpeg, xdotool（未インストールならこのスクリプトが先にaptで導入する）
set -u
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/cnoid_record_$(date +%Y%m%d_%H%M%S).mp4}"
DISP="${DISPLAY:-:1}"

if [ -z "${VIEWER_RESTORE_DIR:-}" ]; then
  echo "[record] ERROR: VIEWER_RESTORE_DIR を設定してください" >&2
  exit 1
fi

for cmd in ffmpeg xdotool; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[record] $cmd が無いためaptで導入します"
    apt-get install -y "$cmd" >/dev/null 2>&1
  fi
done

CNOID_LOG=$(mktemp)
USE_CHOREONOID=1 OMP_NUM_THREADS=1 DISPLAY="$DISP" \
  /choreonoid_ws/install/bin/choreonoid --python scripts/eval_cnoid_viewer.py \
  > "$CNOID_LOG" 2>&1 &
CNOID_PID=$!
echo "[record] choreonoid起動 PID=$CNOID_PID log=$CNOID_LOG"

# ウィンドウが出現するまで待つ（起動直後はプラグイン初期化中でウィンドウが無い）
WIN_ID=""
for i in $(seq 1 30); do
  WIN_ID=$(DISPLAY="$DISP" xdotool search --pid "$CNOID_PID" 2>/dev/null | head -1)
  [ -n "$WIN_ID" ] && break
  if ! kill -0 "$CNOID_PID" 2>/dev/null; then
    echo "[record] ERROR: choreonoidが起動前に終了しました。$CNOID_LOG を確認してください" >&2
    exit 1
  fi
  sleep 1
done

if [ -z "$WIN_ID" ]; then
  echo "[record] WARNING: ウィンドウを特定できずフルスクリーン録画にフォールバック"
  read W H < <(DISPLAY="$DISP" xdotool getdisplaygeometry)
  X=0; Y=0
  WIDTH=$W; HEIGHT=$H
else
  eval "$(DISPLAY="$DISP" xdotool getwindowgeometry --shell "$WIN_ID")"
fi

# ウィンドウ枠込みの座標が仮想画面の外にはみ出すことがあるため、画面サイズにクランプする。
# xdotool getdisplaygeometryはXineramaのモニタ単位（マルチモニタ環境では実際のroot
# windowより小さい値）を返すことがあるため、x11grabが実際に見るroot windowサイズは
# xdpyinfoから取る（2026-07-31発覚: 4384x2466 のモニタ2枚構成で8768x2466が正しい全体サイズ）。
read SCREEN_W SCREEN_H < <(DISPLAY="$DISP" xdpyinfo | awk -F'[x ]+' '/dimensions:/{print $3, $4}')
if [ $((X + WIDTH)) -gt "$SCREEN_W" ]; then WIDTH=$((SCREEN_W - X)); fi
if [ $((Y + HEIGHT)) -gt "$SCREEN_H" ]; then HEIGHT=$((SCREEN_H - Y)); fi
# ffmpegはvideo_sizeが偶数でないと失敗するため2の倍数に丸める
WIDTH=$((WIDTH / 2 * 2))
HEIGHT=$((HEIGHT / 2 * 2))

echo "[record] 録画領域: ${WIDTH}x${HEIGHT} at (${X},${Y})  [画面: ${SCREEN_W}x${SCREEN_H}] -> $OUT"
ffmpeg -f x11grab -video_size "${WIDTH}x${HEIGHT}" -framerate 25 \
  -i "${DISP}+${X},${Y}" -pix_fmt yuv420p -y "$OUT" \
  > "${OUT}.ffmpeg.log" 2>&1 &
FFMPEG_PID=$!
echo "[record] ffmpeg録画開始 PID=$FFMPEG_PID"

# choreonoidの終了を待って録画を止める
wait "$CNOID_PID"
echo "[record] choreonoid終了、録画を停止します"
kill -INT "$FFMPEG_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null
echo "[record] 完了: $OUT"
