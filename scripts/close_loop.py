#!/usr/bin/env python3
"""**判定 → 助言どおりにスケール → 再判定** を 1 本で回す（助教の指摘12 / 実験系譜 9-183）。

⭐ **指摘12**: 「n 倍で届くと分かるなら、①へ戻さず**自動でスケールして③を回せ**」。

⚠️ **当初は「研究の主張は増えない。価値はデモの側」と評価した**（指導教員アドバイス対応方針）。
⭐⭐ **結合テストの連結子として見ると評価が変わる**（[結合テスト_見込み違いの形態.md](../docs/研究応用/設計/結合テスト_見込み違いの形態.md)）:
  **これが無いと「助言に従う」段で人手（描き直し）が入り、閉ループが人手で切れる。**
  ⛔ しかも描き直すと生成側の揺れ（総リーチで 18 %。9-22）が混入して、
  **何が効いたのか分からなくなる。**

**やること**:
  1. 第1層にかける
  2. 棄却されたら**助言された倍率ちょうど**の XML を作る（`make_scaled_arm.py`）
  3. もう一度かけて**通過するか**を確かめる
  4. `--launch` があれば学習も投入する

⚠️ **余裕を上乗せしない。**9-30 で 10 % の上乗せを撤去済みで、
9-11 の梯子が「**余裕は単調に害**」と示している。**助言された最小倍率ちょうどを使う。**

⚠️ **棄却されなかった形態には何もしない**（✅ が出た形態には倍率が無い）。

    python3 scripts/close_loop.py --xml e2e_hockey_s035 --task hockey --spread-y 0.25
    python3 scripts/close_loop.py --xml <短い腕> --task pusher --launch
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnose_morphology as D                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def judge(xml, task, spread_y, tip_ub=None):
    """第1層にかけ、(棄却されたか, 助言された倍率, 所見) を返す。"""
    D.ADVICE.clear()
    tip_ub = D.TIP_RADIUS_UB if tip_ub is None else tip_ub
    geo = D.parse_arm_xml(os.path.join(ROOT, D.ASSET_DIR, f'{xml}.xml'))
    if task in ('pusher', 'hockey') and geo.get('cube') is not None:
        c = np.array(list(map(float, geo['cube']['pos'])), dtype=float)
        if task == 'hockey':
            # ⭐ Bug 50: 先端半径は「設計空間の上限と実値の大きい方」
            d = float(geo['cube']['half']) + D.tip_radius(geo, tip_ub)
            r = float(np.linalg.norm(c[:2]))
            if r > d:
                c[:2] *= (r - d) / r
        target = list(map(float, c))
    else:
        target = [0.8, 0.0, 0.15]
    f1, fatal = D.layer1(geo, task, np.array(target, dtype=float),
                         length_frozen=True, spread_y=spread_y, tip_ub=tip_ub)
    return fatal, (max(D.ADVICE) if D.ADVICE else None), f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', required=True, help='判定する XML 名（拡張子なし）')
    ap.add_argument('--task', default='pusher', choices=['reach', 'pusher', 'hockey'])
    ap.add_argument('--spread-y', type=float, default=0.0)
    ap.add_argument('--tip-ub', type=float, default=None, dest='tip_ub',
                    help='先端カプセル半径の探索上限（cfg の geom_params.size.ub）。Bug 50')
    ap.add_argument('--name', help='修正後の XML 名（既定は <xml>_fix）')
    ap.add_argument('--launch', action='store_true', help='通過したら学習も投入する')
    ap.add_argument('--cfg', default='pusher_tripo_v3')
    a = ap.parse_args()

    print(f'=== ① 判定: {a.xml}（task={a.task}）')
    fatal, scale, f1 = judge(a.xml, a.task, a.spread_y, a.tip_ub)
    for kind, msg in f1:
        if kind in ('fatal', 'warn'):
            print(f'  {D.ICON[kind]} {msg.splitlines()[0]}')
    if not fatal:
        print('\n⭐ **棄却されなかった。**助言が無いので閉ループは回さない。')
        print('   ⚠️ 指摘12 の適用条件: **棄却された形態にしか効かない**')
        return 0
    if scale is None:
        print('\n⛔ **棄却されたが倍率の助言が無い**（可動域・軸など長さ以外の問題）。')
        print('   ⚠️ スケールでは直らない。助言の本文に従うこと')
        return 1

    name = a.name or f'{a.xml}_fix'
    print(f'\n=== ② 助言どおりスケール: **{scale} 倍** → {name}')
    print('   ⚠️ **余裕を上乗せしない**（9-30 で撤去。9-11 が「余裕は単調に害」）')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'make_scaled_arm.py'),
                        '--base', a.xml, '--scale', str(scale), '--name', name],
                       cwd=ROOT, capture_output=True, text=True)
    print('   ' + '\n   '.join(l for l in r.stdout.strip().split('\n') if l.strip())[:600])
    if r.returncode:
        print('⛔ スケールに失敗'); return 1

    print(f'\n=== ③ 再判定: {name}')
    fatal2, scale2, f2 = judge(name, a.task, a.spread_y, a.tip_ub)
    for kind, msg in f2:
        if kind in ('fatal', 'warn'):
            print(f'  {D.ICON[kind]} {msg.splitlines()[0]}')
    if fatal2:
        print(f'\n⛔⛔ **助言に従ったのに、まだ棄却される。**（次の助言: {scale2} 倍）')
        print('   ⚠️ **これは閉ループの欠陥である。**Bug 22（倍率の切り捨て）の再発を疑うこと')
        return 1
    print(f'\n⭐⭐ **通過した。**{a.xml}（棄却）→ {scale} 倍 → {name}（通過）')

    if a.launch:
        print('\n=== ④ 学習を投入')
        print(f'   ⚠️ **投入前に読みを事前登録すること**（§5-2 ⑤-2）')
        print(f'   実行: nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 '
              f'/choreonoid_ws/install/bin/choreonoid --no-window --python '
              f'scripts/choreonoid_train.py cfg={a.cfg} xml_name={name} ...')
        print('   ⭐ **自動では投入しない。**GPU の競合と事前登録を人が確かめるため')
    return 0


sys.exit(main())
