#!/usr/bin/env python3
"""
Entry point for training inside Choreonoid.

ChoreonoidEnv requires WorldItem / AISTSimulatorItem, which need the Qt
application context.  Run this script via choreonoid --no-window:

  USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    choreonoid --no-window --python scripts/choreonoid_train.py cfg=pusher

Hydra overrides (cfg=..., etc.) are passed as positional args by choreonoid
and appear in sys.argv after this script's path.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('USE_CHOREONOID', '1')

# choreonoid --python does not populate sys.argv with script arguments.
# Recover them from /proc/self/cmdline: everything after "--python <script>" is
# passed to Hydra as overrides.
with open('/proc/self/cmdline', 'rb') as _f:
    _cmdline = [a.decode(errors='replace') for a in _f.read().split(b'\x00') if a]
try:
    _python_idx = _cmdline.index('--python')
    _script = _cmdline[_python_idx + 1]
    sys.argv = [_script] + _cmdline[_python_idx + 2:]
except (ValueError, IndexError):
    pass

# ⭐ 系譜 9-236: 物理に効く環境変数を run のフォルダへ書き残す。
#   ⛔ 環境変数は .hydra に残らず、再生で渡し忘れて別の物理で測った（9-229 の訂正）。
#   ⭐ record_arm_trace.py がこのファイルを読んで同じ環境で再生する。
PHYSICS_ENV = ('CNOID_SELF_COLLISION', 'CNOID_CULLING_DEPTH', 'CNOID_CTRL_CLIP', 'CNOID_HINGE_DAMP_TORQUE', 'CNOID_JOINT_MAX_VEL', 'CNOID_FRICTION', 'CNOID_MATERIAL_TABLE',
               'CNOID_RESTITUTION', 'HOCKEY_WALL_RESTITUTION', 'HOCKEY_ARM_BLOCK', 'ARM_SWEEP_OFF')
_rd = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('hydra.run.dir=')), None)
if _rd:
    import json
    os.makedirs(_rd, exist_ok=True)
    with open(os.path.join(_rd, 'launch_env.json'), 'w') as _f:
        json.dump({k: os.environ[k] for k in PHYSICS_ENV if k in os.environ}, _f, ensure_ascii=False, indent=1)

from design_opt.train import main
main()

# choreonoid は --python スクリプト終了後も Qt イベントループが残り続け、プロセスが
# 終了しないことがある（eval_cross_env.py / eval_cnoid_visual.py と同根の問題）。
# 学習完了後に SIGABRT で落ちて呼び出し元（run_2axis_mvp.sh の set -e 等）に
# 偽の失敗として伝播するのを防ぐため、明示的に正常終了させる。
os._exit(0)
