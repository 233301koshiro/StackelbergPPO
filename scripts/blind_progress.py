#!/usr/bin/env python3
"""**取り直し（系譜 9-351）の進み具合を、報酬を見ずに出す**（2026-11-24、ユーザー「全部やっていいよ」）。

⭐ なぜ要るか: 読み 57 は 47 本の `_vp` がそろってから判定する事前登録で、判定まで報酬を読まない。
11-24 に `single_run/postrun_summary.log` を run の名前で grep して「完走（best 35.02）」を見てしまった（系譜 9-351 の 11-24 18:30）。
散文の「grep -v best で見る」は守られない前提なので（CLAUDE.md §5-1-2）、見る口をこのスクリプトだけにし、数値を伏せて出す。

    python3 scripts/blind_progress.py            # 47 本の状態（未起動・学習中 第 N エポックと残り時間・完走）と目視・再生の済み
    python3 scripts/blind_progress.py --logs     # キューのログ（queue_1116*.log）と途中の見張り（midrun_check.log）の末尾、postrun_summary.log の ⛔ の行を伏せて出す
    python3 scripts/blind_progress.py --premise  # 読み 57 の前提のうち judge が見ないもの（種・PRETICK・launch_env・凍結・check_before_conclusion の ⛔/⚠️）を 47 本ぶん（約 11 分）
    python3 scripts/blind_progress.py <run>      # その run の結論前の検査と postrun_summary.log の段を、小数・百分率・mm を伏せて出す

伏せるもの: 小数（`-1.85`・`0.583`）と百分率と mm の値（到達の誤差は「誤差 80 mm」と整数で出る）。報酬・箱の移動・誤差・比はすべて小数で出るので、これで全部隠れる。
整数（エポック・話の数・項目の番号）は残す。⚠️ 物理の量（関節の速さ・トルク）も小数なので伏せる。物理が正しいかを数値で見たいときは
`check_before_conclusion.py` の ✅ / ⛔ の判定で見る（判定は伏せない）。
⭐ 判定の後（`judge_reading57.py` の前提がそろった後）は、このスクリプトではなく普段の道具を使う。前提がそろうと冒頭にそう出す。
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import judge_reading57 as j57  # noqa: E402

RUNS = [n + '_vp' for n in sorted(j57.ALL_SRCS)]
SUMMARY = ROOT / 'single_run' / 'postrun_summary.log'
MASK = re.compile(r'-?\d+(?:\.\d+)?\s*(?:%|mm)|-?\d*\.\d+')


def mask(line):
    return MASK.sub('▒', line)


def state(run):
    """(状態の語, 最後のエポック, 残り時間)。log_train.txt の行頭の番号と ETA だけを読み、報酬の列には触れない"""
    p = ROOT / 'single_run' / run / 'log' / 'log_train.txt'
    if not p.exists():
        return '未起動', None, ''
    ep, eta, done = None, '', False
    for line in p.open(errors='ignore'):
        if 'training done' in line:
            done = True
        m = re.search(r'\]\s+(\d+)\s+T_sample.*?ETA\s+([0-9:]+)', line)
        if m:
            ep, eta = int(m.group(1)), m.group(2)
    return ('完走' if done else '学習中'), ep, ('' if done else eta)


def summary_block(run):
    """postrun_summary.log のうち、その run の「--- <run> 完走」から次の「---」「===」の手前まで"""
    if not SUMMARY.exists():
        return []
    out, on = [], False
    for line in SUMMARY.open(errors='ignore'):
        if re.search(r'---\s+' + re.escape(run) + r'\s', line):
            on = True
        elif on and re.search(r'(---|===)\s', line) and run not in line:
            on = False
        if on:
            out.append(line.rstrip('\n'))
    return out


def overview():
    seen = j57.visual_names()
    ok, st, _ = j57.ready(RUNS, seen)
    if ok:
        print(f'⭐ 47 本の前提がそろった（{st}）。`python3 scripts/judge_reading57.py --judge` を回し、判定は人が系譜 9-351 に書く')
    print(f'{"run":44s} {"状態":6s} {"エポック":>8s} {"残り":>9s}  目視  再生')
    n = {'完走': 0, '学習中': 0, '未起動': 0}
    for r in RUNS:
        s, ep, eta = state(r)
        n[s] += 1
        post = (ROOT / 'single_run' / r / 'trace' / '.postrun_done').exists()
        if s == '完走' and r in seen and post:
            continue                       # そろったものは数だけ出す
        print(f'{r:44s} {s:6s} {("—" if ep is None else str(ep) + "/199"):>8s} {eta:>9s}  '
              f'{"済" if r in seen else "—":4s}  {"済" if post else "—"}')
    print(f'\n完走 {n["完走"]}・学習中 {n["学習中"]}・未起動 {n["未起動"]}（全 {len(RUNS)} 本）。'
          f'完走して目視と再生が済んだ run は上の表から省いた')


def logs(n=12):
    """キューのログ・途中の見張りの末尾と、postrun_summary.log の直近の ⛔ の行。どれも best の行を含むので伏せて出す"""
    for p in sorted((ROOT / 'single_run').glob('queue_1116*.log')) + [ROOT / 'single_run' / 'midrun_check.log']:
        if not p.exists():
            continue
        print(f'\n[{p.name} の末尾 {n} 行]')
        for line in p.read_text(errors='ignore').splitlines()[-n:]:
            print('  ' + mask(line))
    if SUMMARY.exists():
        bad = [ln for ln in SUMMARY.read_text(errors='ignore').splitlines()[-400:] if '⛔' in ln]
        print(f'\n[postrun_summary.log の直近 400 行のうち ⛔ の行: {len(bad)} 行]')
        for line in bad[-n:]:
            print('  ' + mask(line))


def premise():
    """読み 57 の前提のうち judge_reading57.py が見ないもの（ワーカーの種・PRETICK の行・launch_env の 2 つの直し・
    凍結の robot_param_scale・check_before_conclusion.py の ⛔ / ⚠️）を 47 本ぶん、値を伏せて出す（11-25 追加）。
    種と PRETICK は stdout.log から決まった文字列だけを -o 相当で抜く（キューの起動 60 秒後の確かめと同じ文字列）"""
    import json
    bad = 0
    for r in RUNS:
        d = ROOT / 'single_run' / r
        out = (d / 'stdout.log').read_text(errors='ignore') if (d / 'stdout.log').exists() else ''
        ov = (d / '.hydra' / 'overrides.yaml').read_text() if (d / '.hydra' / 'overrides.yaml').exists() else ''
        m = re.search(r'^- seed=(\d+)', ov, re.M)
        seed = int(m.group(1)) if m else 0
        want = '[' + ', '.join(str(1000 * seed + i) for i in range(1, 5)) + ']'
        s = re.search(r'ワーカーの乱数の種 \[[0-9, ]*\]', out)
        # PRETICK の行は学習の stdout.log ではなくワーカーのログにだけ出る（系譜 9-351 の 11-17 追記。キューの ⛔⛔ は誤報）。
        # 親 PID はキューのログの「<run> launched (PID N)」から取り、ワーカー 4 つすべてに行があるかを見る
        pid = next((m.group(1) for q in sorted((ROOT / 'single_run').glob('queue_1116*.log'))
                    for m in re.finditer(re.escape(r) + r' launched \(PID (\d+)\)', q.read_text(errors='ignore'))), None)
        wl = sorted(Path('/tmp').glob(f'cnoid_worker_{pid}_*.log')) if pid else []
        p = len(wl) == 4 and all(re.search(r'tick の前にトルクを絞って速度の上限を守る: [0-9]+ 関節', w.read_text(errors='ignore'))
                                  for w in wl)
        try:
            env = json.loads((d / 'launch_env.json').read_text())
            env_ok = env.get('CNOID_JOINT_MAX_VEL_PRETICK') == '1' and env.get('CNOID_CAPSULE_INERTIA_ROT') == '1'
        except Exception:
            env_ok = False
        items = [('種', bool(s) and want in s.group(0)), ('PRETICK（ワーカー 4/4）', bool(p)), ('launch_env', env_ok)]
        if 'frozen' in r:
            items.append(('凍結 0.0001', 'robot_param_scale = 0.0001' in out))
        cbc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'check_before_conclusion.py'), r],
                             capture_output=True, text=True, cwd=ROOT)
        flags = [mask(ln.strip()) for ln in (cbc.stdout + cbc.stderr).splitlines()
                 if re.match(r'\s*(⛔|⚠️)', ln) and '結論を書く前に' not in ln]
        ok = all(v for _, v in items)
        bad += (not ok) or any(f.startswith('⛔') for f in flags)
        print(f'{"✅" if ok else "⛔"} {r:44s} ' + '  '.join(f'{k}{"✅" if v else "⛔"}' for k, v in items), flush=True)
        for f in flags:
            print('      ' + f[:240], flush=True)
    print(f'\n前提で ⛔ のある run: {bad} / {len(RUNS)}')


def detail(run):
    if run not in RUNS:
        print(f'⚠️ {run} は取り直しの 47 本に無い。伏せずに見てよい run なら普段の道具で見る')
        return 1
    s, ep, eta = state(run)
    print(f'{run}: {s}（第 {ep} エポック{"、残り " + eta if eta else ""}）')
    print('\n[結論前の検査]（小数・百分率・mm は ▒ に伏せた）')
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'check_before_conclusion.py'), run],
                       capture_output=True, text=True, cwd=ROOT)
    for line in (r.stdout + r.stderr).splitlines():
        print('  ' + mask(line))
    print('\n[postrun_summary.log の段]（同じく伏せた）')
    for line in summary_block(run) or ['（まだ無い）']:
        print('  ' + mask(line))
    return 0


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        overview()
    elif a == ['--logs']:
        logs()
    elif a == ['--premise']:
        premise()
    elif len(a) == 1 and not a[0].startswith('-'):
        sys.exit(detail(a[0]))
    else:
        print(__doc__)
        sys.exit(2)
