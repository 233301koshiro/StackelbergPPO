"""再生しながら AIST の**接触点**（どのリンクの・どこで・どれだけめり込み・どれだけの力か）を tick ごとに控え、
腕の位置が NaN になったら**直前の tick の接触**を書き出す（2026-10-20、系譜 9-264: 到達の dist seed0 の話 3 が step 871 で NaN）。

⭐ 仕組み: `CNOID_CONTACT_LOG=1` で全リンクに `LinkContactState` の印を付け（シミュレーション開始前。
  ConstraintForceSolver が開始時に印を見る）、`ChoreonoidSimWorld._tick` の後で `link.contactPoints` を読む。
  再生そのものは `record_arm_trace.py` を回す（話の順・初期状態のくじ・方策の復元は同じ）。
⚠️ 記録が物理を変えていないことは、**既知の NaN が同じ話・同じ step で出るか**で確かめる（出力の最後に並べる）。

    EVAL_RESTORE_DIR=single_run/tripo_pjd_dist_phys EVAL_CHECKPOINT=best TRACE_EPISODES=3 \\
    TRACE_OUT=<scratch>/trace.npz CONTACT_LOG_OUT=<scratch>/contact.txt CONTACT_KEEP=60 \\
    USE_CHOREONOID=1 OMP_NUM_THREADS=1 timeout -k 10 900 /choreonoid_ws/install/bin/choreonoid \\
        --no-window --python scripts/probe_contact_log.py > <scratch>/probe.log 2>&1

⚠️ `TRACE_OUT` を必ず渡す（渡さないと run の `trace/arm_trace.npz` を上書きする）。
"""
import os, sys, runpy, traceback
from collections import deque
sys.path.append(os.getcwd())
# CONTACT_LOG_OFF=1 で印を付けずに回す（対照: 記録が物理を変えていないかを NaN の位置で比べる）
if os.environ.get('CONTACT_LOG_OFF') != '1':
    os.environ['CNOID_CONTACT_LOG'] = '1'
assert os.environ.get('TRACE_OUT'), 'TRACE_OUT を渡す（run の arm_trace.npz を上書きしないため）'

import numpy as np
from khrylib.rl.envs.common import mujoco_env_choreonoid as M

OUT = open(os.environ.get('CONTACT_LOG_OUT', 'contact_log.txt'), 'w')
KEEP = int(os.environ.get('CONTACT_KEEP', '60'))       # NaN の前に書き出す tick 数
st = {'ep': 0, 'step': 0, 'tick': 0, 'nan_done': False, 'buf': deque(maxlen=KEEP), 'nan_at': [], 'n_contact_ticks': 0}


def snapshot(world):
    rows, finite = [], True
    for key, sb in world.sim_bodies.items():
        b = sb.body()
        for k in range(b.numLinks):
            lk = b.link(k)
            if key == 'robot' and not np.isfinite(np.asarray(lk.p, dtype=float)).all():
                finite = False
            cps = lk.contactPoints
            if not cps:
                continue
            d = [c.depth for c in cps]
            i = int(np.argmax(d))
            f = [float(np.linalg.norm(np.asarray(c.force, dtype=float))) for c in cps]
            rows.append(f'{key}/{lk.name} n={len(cps)} depth_max={max(d)*1000:.2f}mm |f|max={max(f):.1f}N '
                        f'p={np.round(np.asarray(cps[i].position, dtype=float), 4).tolist()} '
                        f'n={np.round(np.asarray(cps[i].normal, dtype=float), 3).tolist()} '
                        f'v={np.round(np.asarray(cps[i].velocity, dtype=float), 3).tolist()}')
    return rows, finite


_tick0 = M.ChoreonoidSimWorld._tick
def _tick(self, after_start=False):
    _tick0(self, after_start)
    if after_start or not self.sim_bodies:
        return
    st['tick'] += 1
    rows, finite = snapshot(self)
    if rows:
        st['n_contact_ticks'] += 1
        # ⭐ 接触のある tick は毎回書く（話あたり 20〜80 tick）。落ちても直前までが残る
        OUT.write(f'[話 {st["ep"]}] step {st["step"]} tick {st["tick"]} ' + ' | '.join(rows) + '\n'); OUT.flush()
    st['buf'].append((st['step'], st['tick'], rows, finite))
    if not finite and not st['nan_done']:
        st['nan_done'] = True
        st['nan_at'].append((st['ep'], st['step'], st['tick']))
        OUT.write(f'\n=== 話 {st["ep"]}: 腕の位置が NaN（step {st["step"]}・話の中の tick {st["tick"]}）。直前 {KEEP} tick の接触 ===\n')
        for s, t, rs, fin in st['buf']:
            OUT.write(f'step {s} tick {t} {"" if fin else "⛔NaN "}' + (' | '.join(rs) if rs else '接触なし') + '\n')
        OUT.flush()
M.ChoreonoidSimWorld._tick = _tick

_step0 = M.ChoreonoidSimWorld.step
def _step(self, ctrl, n_frames):
    st['step'] += 1
    return _step0(self, ctrl, n_frames)
M.ChoreonoidSimWorld.step = _step

_reset0 = M.ChoreonoidSimWorld.reset
def _reset(self):
    if st['ep']:
        OUT.write(f'話 {st["ep"]}: {st["step"]} step、接触のあった tick {st["n_contact_ticks"]} / {st["tick"]}'
                  f'{"、NaN あり" if st["nan_done"] else ""}\n'); OUT.flush()
    st.update(ep=st['ep'] + 1, step=0, tick=0, nan_done=False, n_contact_ticks=0)
    st['buf'].clear()
    return _reset0(self)
M.ChoreonoidSimWorld.reset = _reset

# ⭐ record_arm_trace は最後に os._exit(0) するので、要約はその前に書く必要がある → os._exit を包む
_exit0 = os._exit
def _exit(code):
    OUT.write(f'話 {st["ep"]}: {st["step"]} step、接触のあった tick {st["n_contact_ticks"]} / {st["tick"]}'
              f'{"、NaN あり" if st["nan_done"] else ""}\n')
    OUT.write(f'\nNaN の位置（話・step・tick）: {st["nan_at"] or "無し"}\n')
    OUT.flush()
    print(f'[contact] NaN の位置（話・step・tick）: {st["nan_at"] or "無し"}', flush=True)
    sys.stdout.flush()
    _exit0(code)
os._exit = _exit

try:
    runpy.run_path(os.path.join(os.getcwd(), 'scripts/record_arm_trace.py'), run_name='__main__')
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)
os._exit(0)
