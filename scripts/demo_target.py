#!/usr/bin/env python3
"""run の目標位置を `x,y,z` で 1 行出す（`demo_cnoid.sh` の目標の印に使う。2026-10-05）。

⭐ `record_arm_trace.py`（mp4 用）と同じ読み方。⚠️ Target-Pusher の目標は cfg の雛形（`design_opt/cfg/*.yml`）にあり、
run の `.hydra/config.yaml` には書かれていないので、`Config` を通して読む。

    python3 scripts/demo_target.py single_run/e2e_a1v_reach      # → 0.8,0.0,0.15
"""
import os
import sys

import yaml

sys.path.insert(0, os.getcwd())
from omegaconf import OmegaConf  # noqa: E402
from design_opt.utils.config import Config  # noqa: E402

run = sys.argv[1]
d = OmegaConf.to_container(OmegaConf.create(yaml.safe_load(open(f'{run}/.hydra/config.yaml'))), resolve=True)
d.pop('restore_dir', None)
cfg = Config(OmegaConf.create(d), os.getcwd(), run)
rs = cfg.reward_specs or {}
src = rs if (rs.get('use_target_reward', False) or rs.get('use_reach', False)) else cfg.env_specs
print(f"{src.get('target_x', 0.8)},{src.get('target_y', 0.0)},{src.get('target_z', 0.15)}")
