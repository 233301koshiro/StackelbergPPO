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

from design_opt.train import main
main()
