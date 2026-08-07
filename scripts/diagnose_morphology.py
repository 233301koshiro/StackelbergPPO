#!/usr/bin/env python3
"""
diagnose_morphology.py: スケッチ由来の形態が、指定タスクに対してなぜ向く/向かないかを診断し、
専門知識のないユーザーにも読める言葉で助言を返す（第3章 M7、方針レビュー_2026-08-03.md ③⑦）。

判定器（良い/悪いのスコアを返す）から診断器（なぜ悪いか・次に何をすべきかを返す）への拡張。

3層構成:
  第1層（本スクリプト、学習不要・1秒）: XML の幾何だけで即答できる不適合
      - 目標に届かない / 対象に最初からめり込む / 目標が可動範囲の外（関節軸の想定違い）
  第2層（--run 指定時）: 学習後の収束形態から読む
      - gear と太さが揃って下限に張り付く = タスクに根本的に不適合
  第3層（--run 指定時）: 行動トレースから読む
      - 一撃で吹き飛ばす / 押して止める / 静止保持 のどれに収束したか

使い方:
  # 第1層のみ（学習前に数秒で判定。Choreonoid 不要）
  python3 scripts/diagnose_morphology.py --xml tripo_arm_v2c_pj_short --task reach

  # 第1〜3層（学習済み run に対して。Choreonoid が必要）
  EVAL_RESTORE_DIR=single_run/tripo_pj_short USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
    /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/diagnose_morphology.py
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

# Choreonoid 経由（--python）で起動されると cwd がプロジェクト直下でも
# sys.path に入らないため、design_opt / khrylib を import できるようにする
sys.path.insert(0, os.getcwd())

ASSET_DIR = 'assets/mujoco_envs'

# 設計変数の探索範囲（design_opt/cfg/*.yml の geom_params / actuator_params と対応）
GEAR_LB, GEAR_UB = 20.0, 400.0
SIZE_LB, SIZE_UB = 0.03, 0.10
# 「境界に張り付いている」とみなす幅（探索範囲に対する割合）。
# 実測例: tripo_pj_short の size は 0.039/0.030/0.030（範囲 0.03〜0.10）で、
# これを「下限に張り付き」と読めるよう 20% を採る（5% では 0.039 を取りこぼした）
BOUND_TOL = 0.20


def parse_arm_xml(xml_path):
    """XML から腕の幾何とタスク対象の情報を取り出す（Choreonoid 不要）。"""
    root = ET.parse(xml_path).getroot()
    wb = root.find('worldbody')

    # 腕: worldbody の最初の body を根とし、子を辿ってカプセル長を集める
    arm_root = None
    cube = None
    for b in wb.findall('body'):
        if b.get('name') == 'cube':
            cube = b
        elif arm_root is None:
            arm_root = b
    if arm_root is None:
        raise ValueError('腕の body が見つかりません')

    base_pos = np.array([float(x) for x in arm_root.get('pos', '0 0 0').split()])

    lengths, radii, axes = [], [], []
    node = arm_root
    while True:
        child = node.find('body')
        if child is None:
            break
        g = child.find('geom')
        if g is not None and g.get('type') == 'capsule':
            ft = [float(x) for x in g.get('fromto').split()]
            vec = np.array(ft[3:6]) - np.array(ft[0:3])
            lengths.append(float(np.linalg.norm(vec)))
            radii.append(float(g.get('size')))
        j = child.find('joint')
        if j is not None:
            axes.append(tuple(round(float(x)) for x in j.get('axis', '0 0 1').split()))
        node = child

    cube_info = None
    if cube is not None:
        cpos = np.array([float(x) for x in cube.get('pos', '0 0 0').split()])
        cg = cube.find("geom[@name='cube_geom']")
        half = float(cg.get('size').split()[0]) if cg is not None else 0.15
        cube_info = dict(pos=cpos, half=half)

    return dict(base_pos=base_pos, lengths=lengths, radii=radii, axes=axes, cube=cube_info)


def reach_annulus(lengths):
    """N リンク直列アームの到達可能な円環 (最小半径, 最大半径)。
    1本が他の総和より長いと中心付近に届かない穴ができる。"""
    total = sum(lengths)
    longest = max(lengths) if lengths else 0.0
    r_min = max(0.0, 2 * longest - total)
    return r_min, total


# bone_offset の探索範囲（cfg の body_params.offset、rel: true・±OFFSET_HALF）
OFFSET_HALF = 0.5


def max_reach_after_design(lengths):
    """リンク長が設計変数のとき、最適化で到達しうる最大リーチの「上限の目安」。

    offset は初期値相対 ±0.5 m（2成分）なので、単純には1リンクあたり
    sqrt((L+0.5)^2 + 0.5^2) まで伸ばせる計算になる。ただしこれは**過大評価**である:
    rrbot（初期 0.30 + 0.25 m）でこの式は 1.845 m を返すが、実測された最大リーチは
    1.44 m だった（目標 1.5 m に対し残距離 6.02 cm、実験系譜.md 4.3.1）。
    根ボディの offset が運動学的に不活性なこと等が効いていると見られるが、
    正確な上限は解析していない。したがって本関数の値は「これ以上は絶対に届かない」
    という上限としてのみ使い、「届く」の断定には使わない。"""
    return sum(float(np.hypot(l + OFFSET_HALF, OFFSET_HALF)) for l in lengths)


def layer1(geo, task, target, length_frozen=True):
    """第1層: 幾何だけで即答できる不適合。(所見リスト, 致命的か) を返す。"""
    findings = []
    fatal = False
    base = geo['base_pos']
    lengths = geo['lengths']
    r_min, r_now = reach_annulus(lengths)
    axes = geo['axes']
    planar = len(set(axes)) == 1  # 全関節が同一軸 = 単一平面内でしか動かない

    findings.append(('info', f'腕の構成: {len(lengths)}関節・設計図の総リーチ {r_now:.3f} m'
                             f'（各リンク {", ".join(f"{l:.3f}" for l in lengths)} m）'))
    if length_frozen:
        r_max = r_now
        findings.append(('info', 'リンク長は固定されているため、この長さのまま判定します'))
    else:
        r_max = max_reach_after_design(lengths)
        findings.append(('warn', f'リンク長が最適化対象のため、設計図の長さだけでは判定できません。\n'
                                 f'      理論上の上限は約 {r_max:.3f} m ですが、これは**過大評価**です'
                                 f'（rrbot では同じ計算が 1.845 m を返す一方、実測の最大リーチは 1.44 m でした）。\n'
                                 f'      → 以下の判定は「届かない」側だけが確実で、「届く」側は保証できません。'))
    if r_min > 1e-6:
        findings.append(('info', f'到達可能な範囲は半径 {r_min:.3f}〜{r_max:.3f} m のドーナツ状'
                                 f'（1本のリンクが長すぎて中心付近には届きません）'))

    if task == 'reach':
        d = float(np.linalg.norm(target[:2] - base[:2]))
        dz = abs(float(target[2] - base[2]))
        findings.append(('info', f'目標までの水平距離: {d:.3f} m'))

        if d > r_max:
            fatal = True
            need = d / r_max
            findings.append(('fatal',
                f'**腕が短すぎて目標に届きません**（届く範囲は {r_max:.3f} m まで、目標は {d:.3f} m 先）。\n'
                f'      → 腕全体を **{need:.2f} 倍以上**に伸ばしてください。'))
        elif d < r_min:
            fatal = True
            findings.append(('fatal',
                f'**目標が近すぎて届きません**（内側の穴 {r_min:.3f} m の中にあります）。\n'
                f'      → リンクの長さを揃えるか、対象を遠ざけてください。'))
        else:
            margin = (r_max - d) / r_max
            if length_frozen:
                findings.append(('ok', f'目標は可動範囲の内側です（余裕 {margin*100:.0f}%）'))
            else:
                findings.append(('warn', f'上限の概算では届く計算ですが（余裕 {margin*100:.0f}%）、'
                                         f'上記の通り概算は過大評価なので**確実ではありません**。'))
            if length_frozen and margin > 0.4:
                findings.append(('warn',
                    f'ただし腕が目標に対してかなり長めです（余裕 {margin*100:.0f}%）。\n'
                    f'      → 過剰な長さは慣性を増やし、到達までに時間がかかる原因になります。'))

        if planar and dz > 1e-3:
            fatal = True
            findings.append(('fatal',
                f'**目標の高さが腕の動く平面から {dz:.3f} m ずれています**。\n'
                f'      全関節が同じ軸を向いているため、腕は1つの平面内でしか動けません。\n'
                f'      → 関節の向きが想定と違う可能性があります。スケッチを描き直すか、目標の高さを合わせてください。'))

    elif task == 'pusher':
        if geo['cube'] is None:
            findings.append(('warn', '押す対象（cube）が XML に見つかりません'))
        else:
            cpos, half = geo['cube']['pos'], geo['cube']['half']
            near = float(np.linalg.norm(cpos[:2] - base[:2])) - half
            findings.append(('info', f'対象の手前の面までの距離: {near:.3f} m'))
            if near > r_max:
                fatal = True
                findings.append(('fatal',
                    f'**腕が短すぎて対象に触れません**（届く範囲 {r_max:.3f} m、対象は {near:.3f} m 先）。\n'
                    f'      → 腕全体を **{near/r_max:.2f} 倍以上**に伸ばしてください。'))
            elif r_max > near + 2 * half:
                findings.append(('warn',
                    f'静止状態で腕の先端が対象にめり込む可能性があります'
                    f'（腕 {r_max:.3f} m > 対象の奥の面 {near + 2*half:.3f} m）。\n'
                    f'      → 学習の起動時に `arm_safe_init=true` を付けて、腕を対象から逸らした姿勢で始めてください。'))
            else:
                findings.append(('ok', '対象に届き、かつ初期状態でめり込みません'))
    return findings, fatal


def layer2(bodies, task):
    """第2層: 学習後の収束形態から読む。"""
    findings = []
    gears, sizes = [], []
    for b in bodies[1:]:
        g = None
        for j in b.joints:
            if j.actuator:
                g = float(j.actuator.gear)
        if g is not None:
            gears.append(g)
        g0 = b.geoms[0] if b.geoms else None
        if g0 is not None and getattr(g0, 'size', None) is not None:
            sizes.append(float(np.asarray(g0.size, dtype=float).flatten()[0]))

    def at(vals, bound, span):
        return vals and all(abs(v - bound) <= span * BOUND_TOL for v in vals)

    gspan, ssz = GEAR_UB - GEAR_LB, SIZE_UB - SIZE_LB
    findings.append(('info', f'収束したギア比: {", ".join(f"{g:.0f}" for g in gears)}'))
    findings.append(('info', f'収束したリンク太さ: {", ".join(f"{s:.3f}" for s in sizes)} m'))

    if at(gears, GEAR_LB, gspan) and at(sizes, SIZE_LB, ssz):
        if task == 'reach':
            # ⚠️ Reach では「弱く・細く」は正常な最適解でもある（確率的方策下のノイズを
            # 抑えるため、実験系譜.md 第6段）。実測でも、到達できない tripo_pj_short と
            # サブミリで到達する tripo_pj_mid が**同じ署名**を示した（2026-08-03）。
            # したがってこの署名だけでは成否を判定できない。
            findings.append(('warn',
                'ギア比も太さも下限に張り付いています（省エネ設計に収束）。\n'
                '      ただし Reach では**これは正常な最適解でもあります**'
                '（弱く細い方がブレずに狙いを定めやすいため）。\n'
                '      → この署名だけでは成否を判定できません。第1層（届くか）と第3層（実際に到達したか）で確認してください。'))
        else:
            findings.append(('fatal',
                '**ギア比も太さも、選べる範囲の下限に全部張り付いています**。\n'
                '      対象を押すタスクでは強い出力が有利なはずなのに最弱を選んでいます。\n'
                '      これは「何をどう調整しても成績が上がらない」＝この形ではタスクを達成できない、という兆候です。\n'
                '      → 形そのものを変えてください。'))
    elif at(gears, GEAR_UB, gspan):
        findings.append(('warn',
            'ギア比が選べる範囲の上限に張り付いています。\n'
            '      → 出力が足りていない可能性があります。もっと強い（太い）設計を許せば伸びる余地があります。'))
    elif at(gears, GEAR_LB, gspan):
        findings.append(('info',
            'ギア比が下限に張り付いています。Reach のように「そっと正確に止める」タスクでは正常な収束です\n'
            '      （強すぎるとブレて狙いを外すため、弱い方が有利）。'))
    else:
        # 9-10 の検証で判明: この肯定判定は**単独では誤る**。tripo_pjp_short は
        # 対象に触れられていない（実測 −0.0）のに内部解へ収束しており、ここが ✅ になった。
        # 「与えられた形の中では設計が飽和していない」ことしか意味しないので、そう書く。
        findings.append(('ok',
            'ギア比が範囲の内側で収束しています（設計の飽和は起きていません）。\n'
            '      → ただしこれは**タスクを達成できたという意味ではありません**。'
            '届かない形でも内部解に収束することがあります。\n'
            '      　成否は第1層（届くか）と第3層（実際に動かせたか）で確認してください。'))
    return findings


def layer3(trace, task):
    """第3層: 行動トレースから戦略を分類する。"""
    findings = []
    if task == 'pusher' and trace.get('cube_x'):
        xs = trace['cube_x']
        moved = xs[-1] - xs[0]
        peak = trace.get('peak_v', 0.0)
        start = trace.get('move_start')
        if abs(moved) < 1e-3:
            findings.append(('warn', '対象がまったく動いていません（触れられていない可能性）'))
        elif peak > 5.0:
            findings.append(('info',
                f'**一撃で吹き飛ばす戦略**に収束しました（最大 {peak:.1f} m/s、{moved:.2f} m 移動）。\n'
                f'      狙った位置に止めたい場合は、Pusher ではなく Target-Pusher タスクを選んでください。'))
        else:
            findings.append(('info',
                f'穏やかに押す戦略に収束しました（最大 {peak:.1f} m/s、{moved:.2f} m 移動）。'))
        if start is not None:
            findings.append(('info', f'動き出しは {start} ステップ目（それ以前は振りかぶり）'))
    elif task == 'reach' and trace.get('dist'):
        d = trace['dist']
        final_mm = d[-1] * 1000
        conv = trace.get('conv_step')
        if conv is None:
            findings.append(('fatal',
                f'**目標にまったく到達できていません**（最後まで {final_mm:.0f} mm 離れたまま）。\n'
                f'      第1層で指摘した幾何的な問題が、実際の動きにもそのまま現れています。'))
        elif final_mm < 5:
            findings.append(('info',
                f'**目標で静止し続ける戦略**に収束しました'
                f'（最終誤差 {final_mm:.1f} mm、{conv} ステップで到達）。'))
        else:
            findings.append(('warn',
                f'目標付近には行きますが、{final_mm:.0f} mm ずれた位置で止まっています'
                f'（{conv} ステップで 10 mm 以内には入りました）。'))
    return findings


ICON = {'fatal': '❌', 'warn': '⚠️ ', 'ok': '✅', 'info': '  '}


def report(title, groups):
    print(f'\n{"="*72}\n  診断レポート: {title}\n{"="*72}')
    for name, findings in groups:
        if not findings:
            continue
        print(f'\n【{name}】')
        for kind, msg in findings:
            print(f'  {ICON[kind]} {msg}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', help='assets/mujoco_envs 内の XML 名（拡張子なし）')
    ap.add_argument('--task', default='reach', choices=['reach', 'pusher'])
    ap.add_argument('--target', nargs=3, type=float, default=[0.8, 0.0, 0.15])
    ap.add_argument('--length-free', action='store_true',
                    help='リンク長も最適化対象として判定する（--run 指定時は cfg から自動判定）')
    args, _ = ap.parse_known_args()
    length_frozen = not args.length_free

    run = os.environ.get('EVAL_RESTORE_DIR', '')
    if not args.xml and not run:
        ap.error('--xml か環境変数 EVAL_RESTORE_DIR のどちらかを指定してください')

    # run 指定なら、その run の設定から XML 名とタスクを復元する
    if run:
        import yaml
        cfgd = yaml.safe_load(open(f'{run}/.hydra/config.yaml'))
        args.xml = args.xml or cfgd.get('xml_name')
        rs = cfgd.get('reward_specs') or {}
        if rs.get('use_reach'):
            args.task = 'reach'
            args.target = [rs.get('target_x', 0.8), rs.get('target_y', 0.0), rs.get('target_z', 0.15)]
        else:
            args.task = 'pusher'
        # cfg の robot.body_params が空なら bone_offset は凍結されている
        bp = ((cfgd.get('robot') or {}).get('body_params')) or {}
        length_frozen = not bool(bp)

    geo = parse_arm_xml(os.path.join(ASSET_DIR, f'{args.xml}.xml'))
    f1, fatal = layer1(geo, args.task, np.array(args.target, dtype=float), length_frozen)
    groups = [('第1層: 設計図だけで分かること（学習不要）', f1)]

    if run:
        groups += _run_layers23(run, args.task)

    report(f'{args.xml} / タスク={args.task}', groups)
    print()
    if fatal:
        print('  → 総合判定: **この形ではタスクを達成できません。** 上の指摘に沿って形を直してください。')
    else:
        print('  → 総合判定: 幾何的な障害はありません。学習を実行して性能を確かめてください。')
    print()
    # Bug 21: Choreonoid 経由（--python）だとレポート出力後もプロセスが終了せず、
    # 呼び出し側の timeout に当たるまで居座る。診断本体は数秒で終わっているので、
    # ここで明示的に落とす。`extract_gear.py` と同じ扱い。
    sys.stdout.flush()
    if run:
        os._exit(0)


def _run_layers23(run, task):
    """Choreonoid 上でのみ動く第2・3層。import 時点で Choreonoid が要るため関数内で読み込む。"""
    import yaml
    import torch
    from omegaconf import OmegaConf
    from design_opt.utils.config import Config
    from design_opt.agents.genesis_agent import BodyGenAgent, tensorfy
    from design_opt.utils.tools import set_global_seed

    F = OmegaConf.create(yaml.safe_load(open(f'{run}/.hydra/config.yaml')))
    d = OmegaConf.to_container(F, resolve=True)
    d.pop('restore_dir', None)
    cfg = Config(OmegaConf.create(d), os.getcwd(), run)
    cfg.restore_dir = run
    cfg.control_prior = False   # Bug 10: 再評価時に転用フィルタが残ると重みが読まれない
    cfg.morph_prior = False
    torch.set_default_dtype(torch.float64)
    set_global_seed(cfg.seed)

    ag = BodyGenAgent(cfg=cfg, dtype=torch.float64, device=torch.device('cpu'),
                      seed=cfg.seed, num_threads=1, training=False, checkpoint='best')
    env = ag.env
    st = env.reset()
    for _ in range(cfg.skel_transform_nsteps + 2):
        if env.stage == 'execution':
            break
        sv = tensorfy([st])
        if ag.obs_norm is not None:
            sv = ag.normalize_observation(sv)
        with torch.no_grad():
            a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        st, _, _, _, _ = env.step(a)

    f2 = layer2(env.robot.bodies, task)

    # 行動トレース
    trace, xs, dists = {}, [], []
    target = np.array([cfg.env_specs.get('target_x', 0.8),
                       cfg.env_specs.get('target_y', 0.0),
                       cfg.env_specs.get('target_z', 0.15)])
    for _ in range(1100):
        sv = tensorfy([st])
        if ag.obs_norm is not None:
            sv = ag.normalize_observation(sv)
        with torch.no_grad():
            a = ag.policy_net.select_action(sv, mean_action=True).numpy().astype(np.float64)
        st, _, done, _, _ = env.step(a)
        if task == 'pusher':
            xs.append(float(env.get_body_com('cube')[0]))
        else:
            b = env.robot.bodies[-1]
            p = np.array(env._body_xpos.get(b.name, np.zeros(3)))
            R = np.array(env._body_xmat.get(b.name, np.eye(3)))
            dists.append(float(np.linalg.norm(p + R @ np.asarray(b.bone_offset, float) - target)))
        if done:
            break
    if xs:
        v = [abs(xs[i+1]-xs[i]) / 0.01 for i in range(len(xs)-1)]
        trace['cube_x'] = xs
        trace['peak_v'] = max(v) if v else 0.0
        trace['move_start'] = next((i for i, d in enumerate(v) if d > 0.05), None)
    if dists:
        trace['dist'] = dists
        trace['conv_step'] = next((i for i, d in enumerate(dists) if d < 0.01), None)
    f3 = layer3(trace, task)
    return [('第2層: 学習が選んだ設計から分かること', f2),
            ('第3層: 実際の動きから分かること', f3)]


if __name__ == '__main__':
    main()
