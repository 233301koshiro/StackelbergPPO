#!/usr/bin/env python3
"""指摘13の自動化: 走行中の学習を監視し、「使用率が低い関節」が安定したら
固定版を自動生成して並行学習を起動し、完走後に帯の重なりで判定する。

**指摘の中身**（指導教員アドバイス対応方針.md 指摘13、ノートより）:
    使用率が10%未満で安定しだしたら、その関節を固定した学習を並行で開始し、
    遜色なければその助言を採用する。
現状は 9-27・9-61・9-62 で**すべて事後に人手**でやっていた
（完走を待つ → 再生して使用率を見る → 固定 XML を手で作る → 再学習 → 手で比較）。
本スクリプトはこの一連を自動化する。

**⚠️ 事前に実測して決めた設計判断**（9-95）:
学習序盤〜中盤のチェックポイントは使用率が大きく振動する
（`e2e_a1v_reach` で ep10=96% → ep30=40% → ep50=7%、関節4）。
同一チェックポイントを複数回評価すると完全に一致する（決定的）ので、
振動の原因は**評価ノイズではなく方策がまだ収束していないこと**。
したがって:
  1. 学習の**前半 20 %** は判定に使わない（`--warmup-frac`）
  2. **直近 3 回連続**で閾値未満のときだけ「安定」とみなす（`--stable-n`）
どちらも実データ（e2e_a1v_reach の関節4: ep50 以降 6〜12 % で推移）で
このルールが機能することを確認済み。

使い方:
    nohup python3 scripts/joint_fix_watch.py --run e2e_a1v_reach2 > single_run/e2e_a1v_reach2/joint_fix_watch.log 2>&1 &

状態は `single_run/<run>/joint_fix_state.json` に保存し、再起動しても重複起動しない。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNUSED_FRAC = 0.10        # diagnose_morphology.py の UNUSED_FRAC と揃える
STABLE_N = 3              # 直近何回連続で閾値未満なら安定とみなすか
WARMUP_FRAC = 0.20        # 学習の最初の何割は判定に使わないか
POLL_SEC = 60
MAXJOBS = 2


def run_dir(name):
    return os.path.join(ROOT, 'single_run', name)


def log(state_path, msg):
    line = f'[{time.strftime("%F %T")}] {msg}'
    print(line, flush=True)
    with open(state_path.replace('joint_fix_state.json', 'joint_fix_watch.log'), 'a') as f:
        f.write(line + '\n')


def load_hydra(name):
    import yaml
    cfg = yaml.safe_load(open(f'{run_dir(name)}/.hydra/config.yaml'))
    overrides = open(f'{run_dir(name)}/.hydra/overrides.yaml').read()
    return cfg, overrides


def latest_epoch(name):
    """models/ にある最新の epoch_XXXX.p の番号。無ければ None。"""
    d = os.path.join(run_dir(name), 'models')
    eps = [int(m.group(1)) for f in os.listdir(d)
           if (m := re.match(r'epoch_(\d+)\.p$', f))]
    return max(eps) if eps else None


def joint_usage(name, epoch, n_joints_hint=None):
    """diagnose_morphology.py を EVAL_CHECKPOINT=<epoch> で呼び、使用率のリストを返す。"""
    env = dict(os.environ, EVAL_RESTORE_DIR=run_dir(name), EVAL_CHECKPOINT=str(epoch),
               USE_CHOREONOID='1', OMP_NUM_THREADS='1')
    out = subprocess.run(
        ['/choreonoid_ws/install/bin/choreonoid', '--no-window', '--python',
         os.path.join(ROOT, 'scripts', 'diagnose_morphology.py')],
        env=env, capture_output=True, text=True, timeout=90, cwd=ROOT).stdout
    m = re.search(r'関節1 (\d+)%((?:、関節\d+ \d+%)*)', out)
    if not m:
        return None
    vals = [int(m.group(1))] + [int(x) for x in re.findall(r'関節\d+ (\d+)%', m.group(2))]
    return [v / 100.0 for v in vals]


def njobs():
    out = subprocess.run(['pgrep', '-fc', 'choreonoid_train.py'],
                          capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def launch_fixed(base_name, joints, cfg_overrides, state_path):
    """joints（1始まり）を固定した XML を作り、同条件で学習を起動する。"""
    fix_name = f'{base_name}_fix{"".join(map(str, joints))}'
    xml_name = re.search(r'xml_name=(\S+)', cfg_overrides).group(1)
    fixed_xml = f'{xml_name}_fix{"".join(map(str, joints))}'
    if not os.path.exists(f'{ROOT}/assets/mujoco_envs/{fixed_xml}.xml'):
        subprocess.run([sys.executable, 'make_fixed_joint_arm.py',
                         '--base', xml_name, '--fix', *map(str, joints), '--name', fixed_xml],
                        cwd=os.path.join(ROOT, 'scripts'), check=True)
    if os.path.isdir(run_dir(fix_name)):
        log(state_path, f'{fix_name} は既に存在する。起動しない')
        return fix_name
    os.makedirs(run_dir(fix_name), exist_ok=True)
    new_overrides = re.sub(r'xml_name=\S+', f'xml_name={fixed_xml}', cfg_overrides)
    args = [a.strip('- ') for a in new_overrides.split('\n') if a.strip().startswith('-')]
    log(state_path, f'GPU 枠待ち → {fix_name} を起動予定（{" ".join(args)}）')
    while njobs() >= MAXJOBS:
        time.sleep(180)
    with open(f'{run_dir(fix_name)}/stdout.log', 'w') as out:
        subprocess.Popen(
            ['env', 'USE_CHOREONOID=1', 'OMP_NUM_THREADS=1',
             '/choreonoid_ws/install/bin/choreonoid', '--no-window', '--python',
             'scripts/choreonoid_train.py', *args, f'hydra.run.dir=single_run/{fix_name}'],
            cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    log(state_path, f'⭐ {fix_name} 起動した（指摘13 の自動並行学習）')
    return fix_name


def training_done(name):
    p = f'{run_dir(name)}/log/log_train.txt'
    return os.path.exists(p) and 'training done!' in open(p).read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='監視対象の run 名（走行中でも完走後でもよい）')
    ap.add_argument('--warmup-frac', type=float, default=WARMUP_FRAC)
    ap.add_argument('--stable-n', type=int, default=STABLE_N)
    ap.add_argument('--once', action='store_true', help='1回だけ判定して終了（テスト用）')
    a = ap.parse_args()

    state_path = f'{run_dir(a.run)}/joint_fix_state.json'
    state = json.load(open(state_path)) if os.path.exists(state_path) else \
        {'history': [], 'launched': [], 'seen_epochs': []}

    cfg, overrides = load_hydra(a.run)
    max_ep = cfg.get('max_epoch_num', 200)
    warmup_ep = int(max_ep * a.warmup_frac)

    while True:
        ep = latest_epoch(a.run)
        if ep is not None and ep >= warmup_ep and ep not in state['seen_epochs']:
            use = joint_usage(a.run, ep)
            if use:
                state['history'].append({'epoch': ep, 'use': use})
                state['seen_epochs'].append(ep)
                log(state_path, f'ep{ep}: ' + '、'.join(f'関節{i+1} {u*100:.0f}%' for i, u in enumerate(use)))
                recent = state['history'][-a.stable_n:]
                if len(recent) == a.stable_n:
                    n_j = len(recent[0]['use'])
                    stable = [j for j in range(n_j)
                              if all(r['use'][j] < UNUSED_FRAC for r in recent)]
                    key = tuple(j + 1 for j in stable)
                    if stable and list(key) not in state['launched']:
                        log(state_path, f'⭐ 関節{list(key)} が {a.stable_n} 回連続で '
                                         f'{UNUSED_FRAC*100:.0f}% 未満 → 固定版を起動する')
                        launch_fixed(a.run, list(key), overrides, state_path)
                        state['launched'].append(list(key))
            json.dump(state, open(state_path, 'w'), ensure_ascii=False, indent=2)
        if a.once:
            return 0
        if training_done(a.run) and all(training_done(f'{a.run}_fix{"".join(map(str, k))}')
                                          for k in state['launched']):
            log(state_path, '元 run・固定版とも完走。監視を終了する')
            for k in state['launched']:
                fix_name = f'{a.run}_fix{"".join(map(str, k))}'
                sub = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'band_overlap.py'),
                                       '--run-a', run_dir(a.run), '--run-b', run_dir(fix_name)],
                                      capture_output=True, text=True)
                log(state_path, f'関節{list(k)} の判定:\n' + sub.stdout)
            return 0
        time.sleep(POLL_SEC)


if __name__ == '__main__':
    sys.exit(main())
