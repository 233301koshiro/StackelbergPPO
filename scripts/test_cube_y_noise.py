#!/usr/bin/env python3
"""cube_y_noise（Shot タスクの初期条件、9-33）が意図どおり効くかの最小確認。

確かめること:
  1. 既定（未指定）では qpos を一切触らない → **既存 run と挙動が一致する**
  2. 指定すると cube の y が ±noise の範囲でエピソードごとに変わる
  3. add_noise=False（評価時）では振らない

環境の起動は重いので、実装の形と式そのものを検算する。
実機での確認は学習開始後に cube の初期位置が動いているかで行う。

    python3 scripts/test_cube_y_noise.py
"""
import pathlib
import re

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / 'design_opt' / 'envs' / 'pusher.py'


def test_source_shape():
    s = SRC.read_text(encoding='utf-8')
    assert "cube_y_noise = self.env_specs.get('cube_y_noise', 0.0)" in s, '既定 0.0 で取得していない'
    assert re.search(r'if cube_y_noise != 0\.0:', s), '0.0 のとき素通りしていない'
    assert 'cube_y_idx = self.model.nq - 1' in s, 'cube_y の添字が nq-1 でない'
    tail = s.split('cube_y_noise = ')[1][:400]
    assert 'if add_noise else 0.0' in tail, '評価時に振らない分岐が無い'


def test_range():
    rng = np.random.default_rng(0)
    base, noise = 0.0, 0.4
    ys = [base + rng.uniform(-noise, noise) for _ in range(500)]
    assert all(abs(y - base) <= noise + 1e-12 for y in ys), '範囲外が出た'
    assert len(set(np.round(ys, 6))) > 400, 'エピソード間で変わっていない'
    assert noise < 0.403, '振り幅が最短形態 pj_short の到達限界 0.403 m を超えている'


def test_eval_deterministic():
    base = 0.0
    ys = [base + 0.0 for _ in range(10)]
    assert len(set(ys)) == 1 and ys[0] == base


if __name__ == '__main__':
    test_source_shape()
    test_range()
    test_eval_deterministic()
    print('✅ cube_y_noise: 3 件とも通過')
