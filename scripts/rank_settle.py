#!/usr/bin/env python3
"""判定所要学習量の較正（第4章 4.4.2）— 順位が最終順位に固定される最小 epoch を測る。

判定に使う代表値は「そのエポックまでの best exec_R_eps」（= running best）である。
第4章 4.1.3 が各 run の代表値を best checkpoint と定めているため、判定を epoch E で
打ち切ったときに読む値は「ep0..E の best」になる。生の各エポック値ではノイズで順位が
何度も入れ替わるので、較正の物差しとしては running best を用いる。

「確定 epoch」の定義: そのエポック以降、最終エポックまで一度も最終順位から外れない
最小のエポック。途中で一度でも崩れたらカウントし直す（＝逆転が完全に止まった点）。

使い方:
    python3 scripts/rank_settle.py                # 既定のコホートを全部測る
    python3 scripts/rank_settle.py runA runB ...  # 任意の run を比較（引数順は無関係）

Choreonoid 不要・学習不要。stdout.log だけを読む。
"""
import re
import sys
import os

RUN_DIR = 'single_run'


def running_best(run):
    """run の各 epoch における「そこまでの best exec_R_eps」を返す。

    ⚠️ データ源は `log/log_train.txt` を優先する。`stdout.log` は再開（resume）で
    上書きされ ep0 付近が欠落することがあり、欠落したまま順位を測ると
    「共通 epoch の先頭」を確定点と誤読する（2026-08-06 に実際に誤った）。
    log_train.txt は追記されるので再開をまたいで ep0 から残る。
    同じ epoch が複数回現れる場合（再開で midway から再実行された区間）は、
    最後まで走り切った系列である**後の方**を採用する。
    """
    d = os.path.join(RUN_DIR, run)
    path = os.path.join(d, 'log', 'log_train.txt')
    if not os.path.exists(path):
        path = os.path.join(d, 'stdout.log')
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    per_epoch = {}
    for line in open(path, errors='ignore'):
        # log_train.txt は先頭に "[YYYY-MM-DD HH:MM:SS,mmm] " が付く
        m = re.search(r'(?:^|\]\s*)(\d+)\s+T_sample', line)
        if not m:
            continue
        v = re.search(r'exec_R_eps\s+([-\d.]+)', line)
        if v:
            per_epoch[int(m.group(1))] = float(v.group(1))   # 後勝ち
    if not per_epoch:
        raise ValueError(f'{run}: exec_R_eps を含む行が無い')
    out, best = {}, float('-inf')
    for e in sorted(per_epoch):
        best = max(best, per_epoch[e])
        out[e] = best
    return out


def settle_epoch(runs, labels):
    """最終順位に固定される最小 epoch と、最終順位・最終値を返す。"""
    curves = [running_best(r) for r in runs]
    epochs = sorted(set.intersection(*[set(c) for c in curves]))
    if not epochs:
        raise ValueError('共通の epoch が無い（run 同士の進捗がずれている）')
    rank = lambda vals: tuple(sorted(range(len(vals)), key=lambda i: -vals[i]))
    final_vals = [c[epochs[-1]] for c in curves]
    final_rank = rank(final_vals)

    settled = None
    for e in epochs:
        if rank([c[e] for c in curves]) == final_rank:
            if settled is None:
                settled = e
        else:
            settled = None          # 逆転したのでカウントし直す
    order = ' > '.join(labels[i] for i in final_rank)
    # 共通 epoch が 0 から始まらない場合、確定点が観測範囲の外にありうる。
    # settled == epochs[0] なら「その epoch 以前は見えていない」＝上界としてしか読めない。
    truncated = epochs[0] > 0 and settled == epochs[0]
    return settled, order, [round(v, 2) for v in final_vals], epochs[-1], epochs[0], truncated


# (runs, labels, コホート名) — 較正の根拠として第4章 4.4.2 が参照する
COHORTS = [
    # A: 同一条件・別シード対（差は乱数のみ＝最も判別が難しい＝上界）
    (['rrbot_arm_pusher_L2', 'rrbot_arm_pusher_L2_s2'], ['L2', 'L2_s2'],
     'A1 シード対 Pusher (L2)'),
    (['rrbot_arm_reach_L1', 'rrbot_arm_reach_L1_s2'], ['L1', 'L1_s2'],
     'A2 シード対 Reach (L1)'),
    (['rrbot_arm_tp_TP2', 'rrbot_arm_tp_TP2_s2'], ['TP2', 'TP2_s2'],
     'A3 シード対 Target-Pusher (TP2)'),
    # B: 形態が実際に異なる対（判定器の実用ケース）
    (['tripo_pj_long', 'tripo_pj_short'], ['long', 'short'],
     'B1 PJ Reach 長腕 vs 短腕'),
    (['tripo_pj_mid', 'tripo_pj_short'], ['mid', 'short'],
     'B2 PJ Reach 中間 vs 短腕'),
    (['tripo_pj_mid', 'tripo_pj_long'], ['mid', 'long'],
     'B3 PJ Reach 中間 vs 長腕（反転の要）'),
    (['tripo_pjp_long', 'tripo_pjp_short'], ['long', 'short'],
     'B4 PJ Pusher 長腕 vs 短腕'),
    (['tripo_pjp_mid', 'tripo_pjp_short'], ['mid', 'short'],
     'B5 PJ Pusher 中間 vs 短腕'),
    (['tripo_pjp_long', 'tripo_pjp_mid'], ['long', 'mid'],
     'B6 PJ Pusher 長腕 vs 中間'),
    # C: 3形態同時（マトリクスの各列）
    (['tripo_pj_mid', 'tripo_pj_long', 'tripo_pj_short'], ['mid', 'long', 'short'],
     'C1 マトリクス Reach 3形態'),
    (['tripo_pjp_long', 'tripo_pjp_mid', 'tripo_pjp_short'], ['long', 'mid', 'short'],
     'C2 マトリクス Pusher 3形態'),
]


def main():
    args = sys.argv[1:]
    if args:
        cohorts = [(args, args, '指定 run')]
    else:
        cohorts = COHORTS

    print(f'{"コホート":36s} {"確定ep":>8s}  {"最終順位":24s} 最終値')
    print('-' * 100)
    warned = False
    for runs, labels, name in cohorts:
        try:
            ep, order, vals, last, first, truncated = settle_epoch(runs, labels)
            if ep is None:
                ep_s = '未確定'
            elif truncated:
                ep_s = f'≤{ep}'       # 観測開始点で既に確定済み＝上界しか言えない
                warned = True
            else:
                ep_s = str(ep)
            rng = f'(ep{first}〜{last})' if first > 0 else f'(ep{last}まで)'
            print(f'{name:36s} {ep_s:>8s}  {order:24s} {vals}  {rng}')
        except (FileNotFoundError, ValueError) as e:
            print(f'{name:36s} {"SKIP":>8s}  {type(e).__name__}: {e}')
    if warned:
        print('\n⚠️ 「≤N」は観測開始 epoch で既に最終順位だったことを示す（それ以前は'
              'ログが無く確認できない）。確定点の上界としてのみ読むこと。')


if __name__ == '__main__':
    main()
