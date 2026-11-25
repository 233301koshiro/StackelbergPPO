#!/usr/bin/env python3
"""**取り直し（系譜 9-351）の事前登録 読み 57 を、規則どおりに表へ並べる**（2026-11-22）。

⭐ なぜ要るか: 読み 57 は 57-1〜57-11 の 11 項あり、47 本の `_vp` を帯・ギア比・リンク長の合計・最後の平均・50 話の再生で読む。
手で写すと取り違えるので、規則（系譜 9-351 とキューの冒頭の文）をそのまま機械で当てる。

    python3 scripts/judge_reading57.py              # 前提（完走・目視記録）だけを出す。⭐ 値は出さない
    python3 scripts/judge_reading57.py --judge      # 前提がそろった項だけ、帯と規則に当てた結果を出す
    python3 scripts/judge_reading57.py --self-check # 同じ定義を取り直す前の run（`_vp` なし）へ当て、既知の値を再現するか

⛔ 結果を見てから読みを変えないために、**値は 47 本すべてが完走し、目視記録に行があるときだけ出す**（項ごと・途中で覗く口は作らない）。
⚠️ 規則に当てた結果は候補であって、判定は人が系譜 9-351 に書く。前提のうち `check_before_conclusion.py`・ワーカーの種・
不変条件・PRETICK の行はここでは見ない（`check_before_conclusion.py <run>` を回す。PRETICK はワーカーのログ、9-351 の 11-17 追記）。
50 話の再生（57-6・57-10）と第3層（57-11）・確定エポック（57-8）は回すコマンドを出す。50 話の再生の結果があれば数える。

判定の量は `d6_bands.py` と同じ（best ＝ exec_R_eps の最大、`trace_metrics` のギア比・リンク長の合計）。関節 4 のギア比は
`gears[3]`（関節 1 の台座は nan で除いた並び。9-318 ㉙・9-349 と同じ）。
"""
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from d6_bands import band, relation, run_info  # noqa: E402

VISUAL = ROOT / 'docs/研究応用/台帳/目視記録.md'
MOVE_MIN = 0.1      # 読み ㊻・57-6・57-10: 箱の x 移動が 0.1 m 以上の話を「動かした」と数える
EP50 = 'trace/arm_trace_ep50.npz'   # 50 話の再生の置き場（run の 5 話の軌跡は上書きしない）

S = lambda base, *suf: [base + s for s in suf]  # noqa: E731
MID = S('tripo_pjp_mid_phys', '', '_s2_ws', '_s3')
LONG = S('tripo_pjp_long_phys', '', '_s2_ws', '_s3')
FROZEN = S('tripo_pjp_mid_frozen_phys_fx', '', '_s2', '_s3')
UB300 = S('tripo_pjp_mid_phys_ub300', '', '_s2')
PROX = S('tripo_pjdp_prox_phys', '', '_s2_ws', '_s3', '_s4', '_s5')
DIST = ['tripo_pjdp_dist_phys_cg', 'tripo_pjdp_dist_phys_s2_ws', 'tripo_pjdp_dist_phys_cg_s3',
        'tripo_pjdp_dist_phys_cg_s4', 'tripo_pjdp_dist_phys_cg_s5']
PROX_FZ = S('tripo_pjdp_prox_frozen_phys_fx', '', '_s2', '_s3')
DIST_FZ = S('tripo_pjdp_dist_frozen_phys_cg_fx', '', '_s2', '_s3')
REACH = S('e2e_a1v_reach_phys', '', '_s2_ws', '_s3', '_s4', '_s5')
PUSH = S('e2e_a1v_pusher_phys', '', '_s2_ws', '_s3', '_s4', '_s5')
R236 = S('tripo_pjr236p_phys', '', '_s2')
R211 = S('tripo_pjr211p_phys', '', '_s2_ws')
A1 = S('e2e_a1_pusher_phys', '', '_s2', '_s3')
FIX3 = S('e2e_a1_fix3_pusher_phys', '', '_s2', '_s3')

PROBLEMS = []   # --judge で値を出す前に見つかった欠け。1 つでもあれば何も出さない（11-25、外部の見直しの指摘）

ALL_SRCS = set(MID + LONG + FROZEN + UB300 + PROX + DIST + PROX_FZ + DIST_FZ + REACH + PUSH + R236 + R211 + A1 + FIX3)

BEST = lambda r: r['best']  # noqa: E731
J4 = lambda r: r['gears'][3]  # noqa: E731
LSUM = lambda r: r['link_sum']  # noqa: E731

# 帯で読む項: (番号, 名前, [(量の名前, key, 桁, 上と読む側の名前, その run, 相手の名前, その run)], 規則)
#   規則は relation(上と読む側, 相手) の '上' / '重なる' / '下' から事前登録の記号へ
BAND_ITEMS = [
    ('57-1', 'R1（中間、3 seed）最適化 対 凍結', [('best', BEST, 2, '最適化', MID, '凍結', FROZEN)],
     {'上': 'a（R1 を保つ）', '重なる': 'b（R1 は押しで満たされない）', '下': 'b（R1 は押しで満たされない）'}),
    ('57-2', 'マトリクスの押し（3 seed）長腕 対 中間', [('best', BEST, 2, '長腕', LONG, '中間', MID)],
     {'上': 'a（長腕 ＞ 中間）', '重なる': 'b（重なる）', '下': 'c（中間 ＞ 長腕）'}),
    ('57-3', '押しの配分（5 seed）根元重 対 先端重', [('best', BEST, 2, '根元重', PROX, '先端重', DIST)],
     {'上': 'a（根元重 ＞ 先端重）', '重なる': 'b（重なる）。「のみが区別した」は取り下げ', '下': 'c（逆。表 4.7 を「先端重が上位」に替える）。「のみが区別した」は取り下げ'}),
    ('57-3 凍結', '押しの配分の凍結（3 seed）根元重 対 先端重', [('best', BEST, 2, '根元重 凍結', PROX_FZ, '先端重 凍結', DIST_FZ)],
     {'上': '離れる → 「形態の最適化を含む判定のみが区別した」を取り下げ', '下': '離れる → 「形態の最適化を含む判定のみが区別した」を取り下げ',
      '重なる': '重なる → 57-3 が a なら「のみが区別した」を保つ（凍結は 3 シードと添える。補い 2）'}),
    ('57-4', '修論 4.5.4 上限 300 N·m（2 seed）対 150 N·m（3 seed）', [('best', BEST, 2, '300', UB300, '150', MID)],
     {'上': 'a（「向上」を保つ）', '重なる': 'b（「向上」を取り下げる）', '下': 'b（「向上」を取り下げる）'}),
    ('57-5', 'R2 縦型（5 seed）到達 対 押し', [('関節 4 のギア比', J4, 1, '到達', REACH, '押し', PUSH),
                                          ('リンク長の合計', LSUM, 3, '到達', REACH, '押し', PUSH)], None),
    ('57-7', '修論 4.5.5 A1 の押し（3 seed）可動 対 関節 3 固定', [('best', BEST, 2, '可動', A1, '固定', FIX3)],
     {'上': 'a（「固定すると押しの性能が低下する」を言える）', '重なる': 'b（今の書き方のまま）', '下': 'c（固定の帯が上。見落としは到達と押しの両方で生じたと書く。補い 2）'}),
]


def vp(names, on):
    return [n + '_vp' if on else n for n in names]


def visual_names():
    """目視記録の表の行の 1 列目にある run 名。判定の列に ⛔（物理の破綻）がある行は数えない（11-25、外部の見直しの指摘:
    以前はファイルのどこかにバッククォートで名前があれば目視済みとしていた）"""
    try:
        lines = VISUAL.read_text().splitlines()
    except OSError:
        return set()
    seen = set()
    for ln in lines:
        c = ln.split('|')
        if not ln.lstrip().startswith('|') or len(c) < 4 or '⛔' in c[2]:
            continue
        seen |= set(re.findall(r'`([A-Za-z0-9_]+)`', c[1]))
    return seen


def ready(names, seen):
    """(そろったか, 1 行の状況)。完走（200 epoch）と目視記録の行の両方"""
    infos = [run_info(n) for n in names]
    done = sum(1 for i in infos if i and i['done'])
    vis = sum(1 for n in names if n in seen)
    ok = done == len(names) and vis == len(names)
    return ok, f'完走 {done}/{len(names)}・目視 {vis}/{len(names)}', infos


def fmt(b, d):
    return '—' if b is None else f'[{b[0]:.{d}f}, {b[1]:.{d}f}]'


def gap(b1, b2):
    """重ならないときの隙間（正）・重なるときの重なり幅（負）"""
    return max(b1[0] - b2[1], b2[0] - b1[1])


def band_item(item, on, seen, show):
    no, name, qs, rule = item
    names = sorted({n for q in qs for n in vp(q[4], on) + vp(q[6], on)})
    ok, st, _ = ready(names, seen)
    print(f'\n=== {no} {name}  （{st}）')
    if not show:
        return None
    if not ok:
        print('    ⏳ 前提がそろわないので値は出さない')
        return None
    rels = []
    for qn, key, d, la, ra, lb, rb in qs:
        A = [run_info(n) for n in vp(ra, on)]
        B = [run_info(n) for n in vp(rb, on)]
        # 補い 2（11-25）: 表示する桁に丸めた値で帯を作り、端が一致したら「重なる」（relation は端を含む区間）
        rk = lambda x, key=key, d=d: round(key(x), d)  # noqa: E731
        ba, bb = band(*A, key=rk), band(*B, key=rk)
        rel = relation(ba, bb)
        rels.append(rel)
        if rel is None:
            PROBLEMS.append(f'{no} {qn}: 帯が作れない（軌跡・値の欠けか nan）')
            continue
        each = lambda xs: ', '.join(f'{key(x):.{d}f}' for x in xs)  # noqa: E731
        print(f'    {qn}: {la} {fmt(ba, d)}（{each(A)}） 対 {lb} {fmt(bb, d)}（{each(B)}） → {la}が{rel}'
              f'（{"隙間" if rel != "重なる" else "重なり"} {abs(gap(ba, bb)):.{d}f}）')
    if None in rels:
        return rels
    if rule:
        print(f'    → 規則に当てると {rule[rels[0]]}')
    else:  # 57-5: 両方離れる a ／ 片方 b ／ 両方重なる c
        k = sum(r != '重なる' for r in rels)
        print(f'    → 規則に当てると {"a（両方とも重ならない）" if k == 2 else "b（片方だけ重ならない）" if k == 1 else "c（両方重なる）"}')
    return rels


def last_mean(r, n):
    log = ROOT / 'single_run' / r / 'log' / 'log_train.txt'
    v = [float(x) for x in re.findall(r'exec_R_eps\s+(-?[\d.]+)', log.read_text(errors='replace'))]
    return float(np.mean(v[-n:]))


def item_57_9(on, seen, show):
    print('\n=== 57-9 修論 4.2.2 の 1（最良値と最後の 50・20 エポックの平均）押しの 2 組')
    for la, ra, lb, rb in (('長腕', LONG, '中間', MID), ('根元重', PROX, '先端重', DIST)):
        names = vp(ra, on) + vp(rb, on)
        ok, st, _ = ready(names, seen)
        print(f'  [{la} 対 {lb}]（{st}）')
        if not (show and ok):
            continue
        A, B = [run_info(n) for n in vp(ra, on)], [run_info(n) for n in vp(rb, on)]
        r2 = lambda i: round(i['best'], 2)  # noqa: E731  補い 2: 表示の桁に丸めて比べる
        rb_ = relation(band(*A, key=r2), band(*B, key=r2))
        if rb_ is None:
            PROBLEMS.append(f'57-9 {la} 対 {lb}: 最良値の帯が作れない')
            continue
        diffs = [i['best'] - last_mean(i['name'], 50) for i in A + B]
        print(f'    最良値の帯 → {la}が{rb_}。最良値と最後の 50 エポックの平均の差 {min(diffs):.2f}〜{max(diffs):.2f}（記録のみ）')
        if rb_ == '重なる':
            print('    → 最良値の帯が重なるので、この組は 57-9 の対象外')
            continue
        # a は 50 と 20 の両方が最良値と同じ向きに分かれたとき。片方でも重なるか逆なら b（9-351 の「50・20 エポックの平均の帯も」。
        # 11-25 まで 2 行に別々の記号を出していて、結果を見てから片方を選べた。外部の見直しの指摘）
        rels = []
        for n in (50, 20):
            ba = band(*A, key=lambda i, n=n: round(last_mean(i['name'], n), 2))
            bb = band(*B, key=lambda i, n=n: round(last_mean(i['name'], n), 2))
            rel = relation(ba, bb)
            rels.append(rel)
            print(f'    最後の {n} エポックの平均: {la} {fmt(ba, 2)} 対 {lb} {fmt(bb, 2)} → {la}が{rel}')
        if None in rels:
            PROBLEMS.append(f'57-9 {la} 対 {lb}: 最後の平均の帯が作れない')
            continue
        print(f'    → 規則に当てると {"a（50・20 とも同じ向き）" if all(r == rb_ for r in rels) else "b（重なるか逆。但し書きに加える）"}')


def count50(r):
    p = ROOT / 'single_run' / r / EP50
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    dx = np.asarray(z['ep_cube_dx'], float)
    return int((dx >= MOVE_MIN).sum()), len(dx)


def replay_cmd(r):
    return (f'EVAL_RESTORE_DIR=single_run/{r} EVAL_CHECKPOINT=best TRACE_EPISODES=50 TRACE_OUT=single_run/{r}/{EP50} '
            f'USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid --no-window '
            f'--python scripts/record_arm_trace.py > single_run/{r}/trace/ep50.log 2>&1')


def replay_items(on, seen, show):
    for no, name, groups in (
            ('57-6', '修論 4.5.4 の 2.36 倍の押し（50 話で箱を 0.1 m 以上）。a 2 seed とも 48 / 50 以上 ／ b どちらかが 48 未満。2.11 倍は記録のみ',
             (('2.36 倍', R236), ('2.11 倍（記録のみ）', R211))),
            ('57-10', '修論 4.4.1 の 50 話。a 最適化 3 本とも 50/50・凍結のどれかが 50 未満 ／ b 凍結も 3 本とも 50/50 ／ c 最適化のどれかが 50 未満',
             (('最適化', MID), ('凍結', FROZEN)))):
        print(f'\n=== {no} {name}')
        missing = []
        res = {}
        for lab, rs in groups:
            for r in vp(rs, on):
                ok, st, _ = ready([r], seen)
                c = count50(r) if ok else None
                res[r] = c
                if ok and c is None:
                    missing.append(r)
                shown = f'{c[0]} / {c[1]}' if (show and c) else ('あり' if c else '—')
                print(f'    {lab} {r}: {st}・50 話の再生 {shown}')
        if missing:
            print('    50 話の再生がまだ。完走した run から 1 本ずつ（学習と並べると遅くなる。CPU のみ、1 本数分）:')
            for r in missing:
                print('      ' + replay_cmd(r))
    print('\n=== 57-11 修論 4.5.1 の第3層の押し（seed 0 の best、中間 0.47 m・長腕 1.09 m の差し替え）')
    print('    `diagnose_morphology.py` で回す（系譜 9-278 の回し方）。57-2 の結果と合わせて読む:')
    for r in vp([MID[0], LONG[0]], on):
        print(f'      EVAL_RESTORE_DIR=single_run/{r} USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid '
              f'--no-window --python scripts/diagnose_morphology.py > single_run/{r}/trace/diag.log 2>&1')
    print('\n=== 57-8 R4（較正のコホートB の押し 2 対、seed 0・1 の確定エポック）。a 2 対とも両 seed で ≤ 30 ／ b どれかが 30 超')
    # run を名前で渡すと rank_settle.py は `--ws` と同じ定義（running best）で測る。`--ws` は既定のコホートの seed 1 を
    # `_s2_ws` に替えるだけで、ここでは seed 1 に `_s2_ws` の名前をそのまま渡している
    for a, b in ((LONG, MID), (PROX, DIST)):
        for k in (0, 1):
            print(f'      python3 scripts/rank_settle.py {vp([a[k]], on)[0]} {vp([b[k]], on)[0]}')
    print('    記録のみ（コホートA の押し 3 組・短腕を含む押しの対〔短腕は取り直す前のログ〕・コホートC の押し）:')
    for g in (LONG, MID, PROX):
        print(f'      python3 scripts/rank_settle.py {vp([g[0]], on)[0]} {vp([g[1]], on)[0]}')
    for g in (LONG, MID):
        print(f'      python3 scripts/rank_settle.py {vp([g[0]], on)[0]} tripo_pjp_short_phys_cg')
    print(f'      python3 scripts/rank_settle.py {vp([LONG[0]], on)[0]} {vp([MID[0]], on)[0]} tripo_pjp_short_phys_cg')


def self_check():
    """同じ定義を取り直す前の run へ当てて、修論・系譜の既知の帯を再現するか"""
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f'  {"✅" if cond else "⛔"} {name}')
    I = lambda ns: [run_info(n) for n in ns]  # noqa: E731
    rd = lambda b, d: tuple(round(x, d) for x in b)  # noqa: E731
    chk('57-1 の組: 凍結 [6.57, 6.77] 対 最適化 [9.37, 10.44]（予備審査論文 5.2・系譜 9-318）',
        rd(band(*I(FROZEN)), 2) == (6.57, 6.77) and rd(band(*I(MID)), 2) == (9.37, 10.44))
    chk('57-2 の組: 長腕 [20.71, 24.92]（系譜 9-318 ㉘）', rd(band(*I(LONG)), 2) == (20.71, 24.92))
    chk('57-3 の組: 根元重 [9.60, 11.94] 対 先端重 [5.38, 9.52]、隙間 0.08（修論 4.4.3.2）',
        rd(band(*I(PROX)), 2) == (9.6, 11.94) and rd(band(*I(DIST)), 2) == (5.38, 9.52))
    chk('57-5 の組: 関節 4 のギア比 到達 [25.0, 142.1] 対 押し [141.5, 150.0]（系譜 9-349）',
        rd(band(*I(REACH), key=J4), 1) == (25.0, 142.1) and rd(band(*I(PUSH), key=J4), 1) == (141.5, 150.0))
    chk('57-5 の組: リンク長の合計 到達 [1.103, 1.78] 対 押し [1.957, 3.211]（系譜 9-349）',
        rd(band(*I(REACH), key=LSUM), 3) == (1.103, 1.78) and rd(band(*I(PUSH), key=LSUM), 3) == (1.957, 3.211))
    chk('run の名前がキュー 2 本の 47 本と一致する', check_names())
    return 0 if ok else 1


def check_names():
    """BAND_ITEMS と 50 話の組の run が、キューの SRCS（37 本＋10 本）と過不足なく一致するか"""
    q = set()
    for f in ('queue_1116_bug106_retake.sh', 'queue_1116b_bug106_extra.sh'):
        t = (ROOT / 'scripts' / f).read_text()
        q |= set(re.search(r'SRCS="([^"]*)"', t).group(1).split())
    mine = set(ALL_SRCS)
    if q != mine:
        print(f'    キューにだけある: {sorted(q - mine)}  ここにだけある: {sorted(mine - q)}')
    return q == mine and len(q) == 47


def gear_ok(r):
    """57-5 の run の軌跡で、関節 4 を gears[3] で取ってよいか（台座の nan を除いて 4 つ・5 話とも同じ形）。値は出さない。
    ⚠️ 5 話のギア比は完全には一致しない（取り直す前の run で話ごとに 0.001〜0.003 ずれる。判定の桁 0.1 より 2 桁小さい）ので、
    表示の桁の半分（0.05）未満のずれを同じ形とみなす（11-25）"""
    p = ROOT / 'single_run' / r / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return False
    z = np.load(p, allow_pickle=True)
    eps = [np.asarray(g, float) for g in z['ep_gear']]
    fin = [g[np.isfinite(g)] for g in eps]
    return len(fin) > 0 and all(len(f) == 4 for f in fin) and all(np.max(np.abs(f - fin[0])) < 0.05 for f in fin)


def main():
    a = sys.argv[1:]
    if a == ['--self-check']:
        return self_check()
    show = a == ['--judge']
    if a and not show:
        print(__doc__)
        return 2
    seen = visual_names()
    if not check_names():
        print('⛔ run の名前がキューと合わない')
        return 1
    if show:  # 前提は 47 本すべて（9-351: 「37 本とも完走・…目視」、続きの 10 本は「前提は 57 と同じ」）。項ごとに覗く口は作らない
        allv = [n + '_vp' for n in sorted(ALL_SRCS)]
        ok, st, _ = ready(allv, seen)
        if not ok:
            print(f'⏳ 47 本の前提がそろっていない（{st}）。値は出さない。--judge なしで項ごとの状況を見る')
            return 1
        PROBLEMS.extend(f'57-5 {r}: 軌跡のギア比が 4 関節・5 話同じになっていない（gears[3] が関節 4 と言えない）'
                        for r in vp(REACH + PUSH, True) if not gear_ok(r))
        # 全部を先に組み立て、欠けが 1 つも無いときだけ出す（途中で落ちると前の項の値だけ見えてしまう。11-25、外部の見直しの指摘）
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                for it in BAND_ITEMS:
                    band_item(it, True, seen, show)
                item_57_9(True, seen, show)
                replay_items(True, seen, show)
        except Exception as e:  # noqa: BLE001  例外の文は値を含みうるので、型だけ出す
            PROBLEMS.append(f'組み立ての途中で {type(e).__name__}')
        if PROBLEMS:
            print('⛔ 値を出す前に欠けが見つかったので、何も出さない（直してから回す）:')
            for m in PROBLEMS:
                print('    ' + m)
            return 1
        sys.stdout.write(buf.getvalue())
        return 0
    for it in BAND_ITEMS:
        band_item(it, True, seen, show)
    item_57_9(True, seen, show)
    replay_items(True, seen, show)
    if not show:
        print('\n（値は出していない。そろったら --judge。前提の残りは `python3 scripts/check_before_conclusion.py <run>` で確かめる）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
