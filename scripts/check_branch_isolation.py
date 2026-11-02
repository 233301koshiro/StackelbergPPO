#!/usr/bin/env python3
"""main と軸6 のブランチ（feat/layer4-robotassembler）の設定が混ざっていないかを確かめる（2026-11-02、ユーザー「何回も言っているけど」）。

    python3 scripts/check_branch_isolation.py          # ⛔ が 1 つでもあれば終了コード 1
    python3 scripts/check_branch_isolation.py --reproduce   # 既知の混ざり方を人工的に作り、全部 ⛔ になるかを確かめる

混ざってはいけない 3 つの層（境界の説明・共用のファイルの一覧・チェックリストは docs/リポジトリ説明/mainと軸6ブランチの境界.md）:
  ① RobotAssembler の設定（部品・スケール表・部品の質量・組み方）… ブランチだけ
  ② 幾何のリンクの設定（カプセルの骨・太さ・横ずれ。khrylib/robot/xml_robot.py）… 共用。ブランチは cfg の口で足す
  ③ 物理の環境（CNOID_* の環境変数・*_phys の XML）… 共用。軸6 だけの値は環境変数の口で足し、既定では効かない

検査:
  A. ファイル: 両方にあって中身が違うファイルが、既知の一覧（KNOWN_DIFF）の外に増えていないか。ブランチだけのファイルと、
     main にあって軸6 の口を持つ共用ファイルは一覧として出す（共用のファイルを変えたら、ブランチの run にも効く）
  B. run: 軸6 の run（ブランチだけの XML か cfg を使う）は worktree で、それ以外は main で起動したか（hydra の runtime.cwd）
  C. 設定: 軸6 だけの cfg キーが軸6 以外の run の config に無いか。軸6 だけの環境変数が軸6 以外の run の launch_env に無いか。
     軸6 の run の launch_env に要るもの（REQUIRED_RA_ENV）があるか
  D. ブランチが main の今の共用コードを取り込んでいるか（取り込んでいないと、次の軸6 の run は古い共用コードで走る）
"""
import glob
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BRANCH = 'feat/layer4-robotassembler'
MAIN_DIR = '/userdir/StackelbergPPO'
RA_DIR = '/userdir/StackelbergPPO_ra'
# 両方にあって中身が違ってよいファイル（理由つき）。増やすときは境界の md の表にも書く
KNOWN_DIFF = {
    'khrylib/robot/xml_robot.py': '② 幾何: cfg の capsule_lateral（カプセルの横ずれを保つ）と part_mass_file（長さから部品の質量を作る）。どちらも cfg に無ければ従来どおり（系譜 9-328）',
    'scripts/demo_cnoid.sh': '見た目だけ: ra_* のタスク（部品のメッシュ。XML がブランチにしか無い）',
    'scripts/watch_midrun.sh': '見張り: RUN_FILTER で軸6 の run だけを見る（系譜 9-329）',
    'scripts/queue_postrun_replay.sh': '自動再生: 軸6 の run の一覧（系譜 9-329）',
}
# 軸6 だけが使う環境変数（main の既定では効かない）。物理のものは軸6 の run の launch_env に要る
RA_ENV = {
    'CNOID_JOINT_LIMIT_TORQUE': '③ 物理: 可動域をトルクで掛ける（Bug 85）',
    'CNOID_JOINT_LIMIT_SLACK': '③ 物理: 同上の遊び',
    'CNOID_VISUAL_RAW': '見た目だけ: 部品のメッシュを間引かず出す（画面のデモ）',
}
REQUIRED_RA_ENV = {'CNOID_JOINT_LIMIT_TORQUE'}
RA_HOOK_WORDS = ('軸6', 'ra15', 'RobotAssembler', 'robot_assembler')

bad, warn = [], []


def git(*a):
    return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout


def code_files(lines):
    return [l for l in lines if l and not l.startswith('docs/') and not l.endswith('.md')]


def flat_keys(d):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out.add(k)
            out |= flat_keys(v)
    return out


def check_run(run, cwd, cfg, le, ra_xml, ra_cfg, ra_keys):
    """1 本の run の B・C。（軸6 の run か, 見つけた問題）を返す。"""
    probs = []
    is_ra = cfg.get('xml_name') in ra_xml or cfg.get('cfg') in ra_cfg
    if is_ra:
        if cwd != RA_DIR:
            probs.append(f'B: 軸6 の run {run} が {cwd} で起動した（worktree {RA_DIR} のはず）')
        miss = REQUIRED_RA_ENV - set(le or {})
        if miss:
            probs.append(f'C: 軸6 の run {run} の launch_env に {sorted(miss)} が無い（書き戻しの可動域で学習・再生している）')
    else:
        if cwd != MAIN_DIR:
            probs.append(f'B: 軸6 でない run {run} が {cwd} で起動した（main {MAIN_DIR} のはず）')
        k = flat_keys(cfg) & ra_keys
        if k:
            probs.append(f'C: 軸6 でない run {run} の config に軸6 だけのキー {sorted(k)}')
        e = set(le or {}) & set(RA_ENV)
        if e:
            probs.append(f'C: 軸6 でない run {run} の launch_env に軸6 だけの環境変数 {sorted(e)}')
    return is_ra, probs


def reproduce():
    """既知の混ざり方を人工的に作り、check_run が全部 ⛔ を出すかを確かめる（出せない検査の ✅ は根拠にならない）。"""
    X, C, K = {'e2e_a1v_ra15'}, {'pusher_ra15'}, {'capsule_lateral', 'part_mass_file'}
    ra = {'xml_name': 'e2e_a1v_ra15', 'cfg': 'pusher_ra15'}
    mn = {'xml_name': 'e2e_b1_phys_arm171', 'cfg': 'pusher_gearonly_real'}
    cases = [
        ('軸6 の run を main で起動した', ('r', MAIN_DIR, ra, {'CNOID_JOINT_LIMIT_TORQUE': '1'})),
        ('軸6 の run の可動域トルクが無い', ('r', RA_DIR, ra, {'CNOID_SELF_COLLISION': '1'})),
        ('main の run を worktree で起動した', ('m', RA_DIR, mn, {})),
        ('main の run に軸6 の cfg キー', ('m', MAIN_DIR, {**mn, 'robot': {'capsule_lateral': True}}, {})),
        ('main の run に軸6 の環境変数', ('m', MAIN_DIR, mn, {'CNOID_JOINT_LIMIT_TORQUE': '1'})),
    ]
    ok = True
    for name, (run, cwd, cfg, le) in cases:
        _, p = check_run(run, cwd, cfg, le, X, C, K)
        print(f'  {"✅ 検出" if p else "⛔ 見逃し"}: {name}  {p[:1]}')
        ok &= bool(p)
    for name, (run, cwd, cfg, le) in [('正しい軸6 の run', ('r', RA_DIR, ra, {'CNOID_JOINT_LIMIT_TORQUE': '1'})),
                                      ('正しい main の run', ('m', MAIN_DIR, mn, {'CNOID_SELF_COLLISION': '1'}))]:
        _, p = check_run(run, cwd, cfg, le, X, C, K)
        print(f'  {"✅ 通る" if not p else "⛔ 誤検出"}: {name}  {p[:1]}')
        ok &= not p
    print('✅ 既知の混ざり方をすべて検出した' if ok else '⛔ 検査が壊れている')
    sys.exit(0 if ok else 1)


def main():
    if '--reproduce' in sys.argv:
        reproduce()
    os.chdir(ROOT)
    if not git('rev-parse', '--verify', '--quiet', BRANCH).strip():
        print(f'⛔ ブランチ {BRANCH} が無い'); sys.exit(1)

    # ---- A. ファイル -------------------------------------------------------
    st = [l.split('\t') for l in git('diff', '--name-status', 'main', BRANCH).splitlines()]
    only_branch = code_files([p[-1] for p in st if p[0] == 'A'])
    only_main = code_files([p[-1] for p in st if p[0] == 'D'])
    differ = code_files([p[-1] for p in st if p[0].startswith('M')])
    print('【A】ファイルの境界')
    print(f'  ブランチだけ（① RobotAssembler の設定・道具）: {len(only_branch)} 本')
    for f in only_branch:
        print(f'    {f}')
    print(f'  両方にあって中身が違う: {len(differ)} 本')
    for f in differ:
        if f in KNOWN_DIFF:
            print(f'    ✅ {f} — {KNOWN_DIFF[f]}')
        else:
            print(f'    ⛔ {f} — 既知の一覧（KNOWN_DIFF）に無い。どちらが正かを決め、理由を書く')
            bad.append(f'A: 中身が違うファイルが増えた {f}')
    for f in only_main:
        if not f.startswith('scripts/check_branch_isolation'):
            warn.append(f'A: main だけにあり、ブランチへまだ取り込んでいない {f}')
    # ---- 軸6 だけの XML・cfg・cfg キー ---------------------------------------
    ra_xml = {os.path.splitext(os.path.basename(f))[0] for f in only_branch
              if f.startswith('assets/mujoco_envs/') and f.endswith('.xml')}
    ra_cfg = {os.path.splitext(os.path.basename(f))[0] for f in only_branch
              if f.startswith('design_opt/cfg/') and f.endswith('.yml')}
    main_keys = set()
    for f in glob.glob('design_opt/cfg/*.yml'):
        try:
            main_keys |= flat_keys(yaml.safe_load(open(f)))
        except Exception:
            pass
    ra_keys = set()
    for c in ra_cfg:
        ra_keys |= flat_keys(yaml.safe_load(git('show', f'{BRANCH}:design_opt/cfg/{c}.yml')))
    ra_keys -= main_keys
    print(f'\n  軸6 だけの XML {sorted(ra_xml)}・cfg {sorted(ra_cfg)}・cfg のキー {sorted(ra_keys)}')
    # 共用ファイルのうち、軸6 の語・軸6 だけの環境変数・軸6 だけの cfg キー（'キー' の形）を持つもの
    pat = '|'.join(list(RA_HOOK_WORDS) + sorted(RA_ENV) + [f"'{k}'" for k in sorted(ra_keys)])
    hooks = [f for f in code_files(git('grep', '-l', '-E', pat, 'main', '--', '*.py', '*.sh', '*.yml')
                                   .replace('main:', '').splitlines()) if not f.startswith('scripts/check_')]
    print(f'  main にあって軸6 の口を持つ共用ファイル（変えるとブランチの run にも効く）: {len(hooks)} 本')
    for f in hooks:
        print(f'    {f}')

    # ---- B・C. run ---------------------------------------------------------
    print('\n【B・C】run の境界（single_run/*/.hydra）')
    n_ra = n_main = 0
    for h in sorted(glob.glob('single_run/*/.hydra/hydra.yaml')):
        run = h.split('/')[1]
        try:
            cwd = yaml.safe_load(open(h))['hydra']['runtime']['cwd']
            cfg = yaml.safe_load(open(os.path.join(os.path.dirname(h), 'config.yaml'))) or {}
        except Exception as e:
            warn.append(f'B: {run} の hydra の記録が読めない（{type(e).__name__}）')
            continue
        le_p = os.path.join('single_run', run, 'launch_env.json')
        le = json.load(open(le_p)) if os.path.exists(le_p) else None
        is_ra, probs = check_run(run, cwd, cfg, le, ra_xml, ra_cfg, ra_keys)
        n_ra += is_ra
        n_main += not is_ra
        bad.extend(probs)
    print(f'  軸6 の run {n_ra} 本（worktree で起動・launch_env に {sorted(REQUIRED_RA_ENV)}）・それ以外 {n_main} 本（main で起動・軸6 のキーと環境変数なし）')

    # ---- D. 共用コードの取り込み ---------------------------------------------
    print('\n【D】ブランチが main の共用コードを取り込んでいるか')
    behind = code_files(git('diff', '--name-only', f'{BRANCH}...main').splitlines())
    if behind:
        print(f'  ⚠️ main で変わってブランチにまだ無い共用コード {len(behind)} 本（次の軸6 の run の前に worktree で git merge main）:')
        for f in behind:
            print(f'    {f}')
        warn.append(f'D: ブランチが main の共用コード {len(behind)} 本を取り込んでいない')
    else:
        print('  ✅ 取り込み済み（main の共用コードはブランチと同じ）')

    print('\n========================================================')
    for w in warn:
        print(f'⚠️ {w}')
    for b in bad:
        print(f'⛔ {b}')
    if bad:
        print(f'⛔ 混ざっている疑い {len(bad)} 件'); sys.exit(1)
    print('✅ main と軸6 のブランチは混ざっていない（A〜C）。' + ('⚠️ は上のとおり' if warn else ''))


if __name__ == '__main__':
    main()
