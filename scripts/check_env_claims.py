#!/usr/bin/env python3
"""**道具と環境の振る舞いについての主張**を、実際に動かして確かめ直す（系譜 9-257、2026-10-14）。

なぜ: CLAUDE.md・デバッグ戦記・スクリプトの注意書きには「Choreonoid は起動に約 105 秒」のような
**現在形の主張**があり、AI はそれを前提に判断する。だが **10-14 のファクトチェック A〜F はどれもこの層を対象にしておらず**、
他の検査も見ない（ログと md の数値の突き合わせでも、既知の古い主張の検索でもない）。
実際に 2 件が古いか誤っていた: 起動 105 秒（実測 2 秒）・「sys.exit(0) のせいで exit 137」（実際は Choreonoid が自分では終わらず timeout が殺す）。

    docker exec akita_sp_ppo bash -c 'cd /userdir/StackelbergPPO && python3 scripts/check_env_claims.py'   # 約 1 分半（Choreonoid を 4 回起動）

⚠️ コンテナで回す。ホストには Choreonoid が無く、以前は 3 件とも exit 127 のまま「⛔ 成り立たない」と出ていた（2026-10-22）。
   Choreonoid が無いときは主張を判定せずに止まる（終了コード 2）。

主張を足すときは CLAIMS に 1 行（主張・書いてある場所・確かめ方）。確かめられない主張は足さない（プロンプト ② の G で人が見る）。
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHOREONOID = '/choreonoid_ws/install/bin/choreonoid'


def _cnoid(src, limit):
    """src を Choreonoid で回す → (終了コード, 秒, 出力)。出力はファイルへ（Bug 43: パイプに繋がない）。"""
    with tempfile.TemporaryDirectory() as d:
        py, log = Path(d) / 'p.py', Path(d) / 'out.log'
        py.write_text(src)
        t = time.time()
        with open(log, 'w') as f:
            rc = subprocess.call(['timeout', '-k', '5', str(limit), CHOREONOID, '--no-window', '--python', str(py)],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        return rc, time.time() - t, log.read_text(errors='replace')


def startup():
    rc, s, out = _cnoid('import os\nprint("ALIVE", flush=True)\nos._exit(0)\n', 300)
    ok = rc == 0 and 'ALIVE' in out and s < 20
    return ok, f'os._exit(0) で終えて {s:.1f} 秒・exit {rc}（20 秒未満なら「約 2 秒」は成り立つ）'


def no_self_exit():
    rc, s, out = _cnoid('import sys\nprint("ALIVE", flush=True)\nsys.exit(0)\n', 30)
    typeerr = 'Invoked with: 0' in out
    hung = rc in (124, 137, -9)   # -9: timeout の KILL はプロセス群に飛ぶので timeout 自身も落ちる
    return typeerr and hung, (f'sys.exit(0) → TypeError {"あり" if typeerr else "なし"}・'
                              f'{"自分では終わらず timeout が止めた" if hung else "自分で終わった"}（exit {rc}・{s:.0f} 秒）')


def ignores_sigterm():
    """6 秒で SIGTERM、効かなければ 10 秒後に KILL。KILL まで生きていれば「SIGTERM を無視する」（2026-10-20 に足した）"""
    out = []
    for name, src in (('待機', 'print("ALIVE", flush=True)\n'),
                      ('Python 実行中', 'import time\nprint("ALIVE", flush=True)\nwhile True:\n    time.sleep(0.2)\n')):
        with tempfile.TemporaryDirectory() as d:
            py = Path(d) / 'p.py'
            py.write_text(src)
            t = time.time()
            rc = subprocess.call(['timeout', '-s', 'TERM', '-k', '10', '6', CHOREONOID, '--no-window', '--python', str(py)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
            out.append((name, rc in (137, -9) and time.time() - t > 14, rc, time.time() - t))
    return all(o[1] for o in out), '・'.join(f'{n}: exit {rc}・{s:.0f} 秒' for n, _, rc, s in out) + \
        '（16 秒＝SIGTERM の後も生き、KILL で落ちた）'


def no_push_hook():
    hooks = [p for p in (ROOT / '.git' / 'hooks').iterdir() if not p.name.endswith('.sample')]
    bad = [p.name for p in hooks if 'push' in p.read_text(errors='replace')]
    return not bad, f'push する hook: {bad or "無し"}'


CLAIMS = [
    ('Choreonoid の起動は約 2 秒（旧記述 105 秒）', 'CLAUDE.md §5-2 ⑤-3・Bug 43', startup),
    ('Choreonoid は sys.exit(0) で TypeError、スクリプトが終わっても自分では終わらない → 末尾は os._exit(0)',
     'CLAUDE.md §5-2 ⑤-3・Bug 43 追記・record_arm_trace.py 末尾', no_self_exit),
    ('Choreonoid は SIGTERM を無視する（待機中も Python 実行中も）→ 止めるなら kill -9',
     'デバッグ戦記 Bug 26・新しいチャットの始め方 ②', ignores_sigterm),
    ('コミットは自動では push されない', 'CLAUDE.md §6 権限表', no_push_hook),
]


def main():
    if not Path(CHOREONOID).exists():   # 無いまま回すと exit 127 を「主張が誤り」と読んでしまう
        print(f'⛔ {CHOREONOID} が無い（ホストで回した？）。主張は判定していない。コンテナで回す:\n'
              "    docker exec akita_sp_ppo bash -c 'cd /userdir/StackelbergPPO && python3 scripts/check_env_claims.py'")
        return 2
    bad = unknown = 0
    for claim, where, fn in CLAIMS:
        try:
            ok, msg = fn()
        except Exception as ex:   # 確かめられなかったことを黙らない
            ok, msg = None, f'確かめられない: {type(ex).__name__}: {ex}'
        bad += ok is False
        unknown += ok is None
        print(f"{'✅' if ok else ('⚠️' if ok is None else '⛔')} {claim}\n    場所: {where}\n    実測: {msg}")
    if bad:
        print(f'\n⛔ {bad} 件が成り立たない → 書いてある場所を直す')
    if unknown:
        print(f'\n⚠️ {unknown} 件は確かめられなかった → 原因を見て回し直す')
    if not bad and not unknown:
        print('\n✅ 主張はすべて今も成り立つ')
    return 1 if bad or unknown else 0


if __name__ == '__main__':
    sys.exit(main())
