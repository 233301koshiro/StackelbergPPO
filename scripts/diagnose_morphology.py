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
import math
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

    lengths, radii, axes, offsets, dirs, ranges = [], [], [], [], [], []
    node = arm_root
    while True:
        child = node.find('body')
        if child is None:
            break
        # ⚠️ Bug 23: リンクの見た目の長さ（geom fromto）と、関節の取り付け位置（body pos）は
        # 別物である。運動学が決まるのは後者なので、両方を読んで突き合わせる。
        offsets.append(float(np.linalg.norm(
            np.array([float(x) for x in child.get('pos', '0 0 0').split()]))))
        g = child.find('geom')
        if g is not None and g.get('type') == 'capsule':
            ft = [float(x) for x in g.get('fromto').split()]
            vec = np.array(ft[3:6]) - np.array(ft[0:3])
            lengths.append(float(np.linalg.norm(vec)))
            dirs.append(vec / (np.linalg.norm(vec) + 1e-12))
            radii.append(float(g.get('size')))
        j = child.find('joint')
        if j is not None:
            axes.append(tuple(round(float(x)) for x in j.get('axis', '0 0 1').split()))
            rg = j.get('range')
            ranges.append(tuple(float(x) for x in rg.split()) if rg else None)
        node = child

    cube_info = None
    if cube is not None:
        cpos = np.array([float(x) for x in cube.get('pos', '0 0 0').split()])
        cg = cube.find("geom[@name='cube_geom']")
        half = float(cg.get('size').split()[0]) if cg is not None else 0.15
        cube_info = dict(pos=cpos, half=half)

    return dict(base_pos=base_pos, lengths=lengths, radii=radii, axes=axes,
                offsets=offsets, dirs=dirs, cube=cube_info, ranges=ranges)


def kinematic_reach(lengths, offsets):
    """関節の取り付け位置の連鎖から出る**実効リーチ**。

    第1層は長らくリンク長（geom）の総和だけでリーチを判定していたが、それは
    「各関節が前のリンクの先端にある」という**暗黙の仮定**に依存している。
    Bug 23 では生成器がこの仮定を破った XML を作り、診断は実効 0.504 m の腕に
    「総リーチ 0.802 m・余裕 0%」と答えた。仮定は明示して検算する。
    """
    if not lengths or not offsets:
        return sum(lengths)
    # offsets[i] = 「i 番目の子 body が親のどこに付いているか」。
    # 正常なら offsets[i+1] == lengths[i] になる。先端は offsets の連鎖 + 末端リンク。
    return sum(offsets[:len(lengths)]) + lengths[-1]


def reach_annulus(lengths):
    """N リンク直列アームの到達可能な円環 (最小半径, 最大半径)。
    1本が他の総和より長いと中心付近に届かない穴ができる。"""
    total = sum(lengths)
    longest = max(lengths) if lengths else 0.0
    r_min = max(0.0, 2 * longest - total)
    return r_min, total


# bone_offset の探索範囲（cfg の body_params.offset、rel: true・±OFFSET_HALF）
OFFSET_HALF = 0.5


def max_reach_after_design(lengths, offset_half=OFFSET_HALF):
    """リンク長が設計変数のとき、最適化で到達しうる最大リーチの「上限の目安」。

    offset は初期値相対 ±0.5 m（2成分）なので、単純には1リンクあたり
    sqrt((L+0.5)^2 + 0.5^2) まで伸ばせる計算になる。ただしこれは**過大評価**である:
    rrbot（初期 0.30 + 0.25 m）でこの式は 1.845 m を返すが、実測された最大リーチは
    1.44 m だった（目標 1.5 m に対し残距離 6.02 cm、実験系譜.md 4.3.2.1）。
    根ボディの offset が運動学的に不活性なこと等が効いていると見られるが、
    正確な上限は解析していない。したがって本関数の値は「これ以上は絶対に届かない」
    という上限としてのみ使い、「届く」の断定には使わない。"""
    return sum(float(np.hypot(l + offset_half, offset_half)) for l in lengths)


def scale_advice(need):
    """必要倍率を返す。**判定を覆すのに必要な最小の変更**そのもの。

    Bug 22: 以前は `f'{need:.2f}'` で表示していたため**四捨五入で切り捨てられ**、
    助言どおりに直しても届かない形態が出た（必要 1.9846 倍 → 表示「1.98 倍以上」
    → 実際は 2 mm 足りず、同じ診断が再び却下した）。倍率は必ず**切り上げる**。
    この切り上げは維持する。

    ⚠️ 2026-09-07: **10% の余裕を上乗せする処理を撤去した。**
    導入時の理由は「下限ちょうどでは伸ばしきった特異姿勢でしか目標に触れられず、
    Reach では実質的に達成できない」だったが、実験で**否定された**。

      余裕 0%   −0.27 / −0.33      ← 最良
      余裕 5.3% −1.19 / −1.41
      余裕 10%  −1.66 / −3.42
      （各 2 seed・隣接条件も域が重ならない。実験系譜 9-11）

    到達タスクでは**余裕を広げるほど単調に悪化**し、別系統の縦型 3 点でも
    同じ傾向が再現した（9-29）。制御コストを 0 にしても差は残る（9-15）ので、
    ctrl コスト由来でもない。押しタスクでは腕が長いほど有利なので余裕は無害だが、
    **一律に上乗せする根拠は無い**。

    Wachter et al. (2018) / Ustun et al. (2019) の反実仮想説明・実行可能な救済が
    要請するのも「判定を覆すのに必要な**最小の**変更」であり、
    理論上の要件と実験結果がここで一致した（第2章 2.4）。
    """
    return math.ceil(need * 100) / 100



def planar_chain_reach(lengths, ranges_deg, d, dz, steps=61):
    """B（9-72）: 可動域を入れた平面チェーンが点 (d, dz) に届くか。

    零姿勢でチェーンは「上」(0, 1) を向き、関節 i は累積角 Σθ で回る
    （縦型アームはリンクが +Z、根元ピッチが ±90°）。

    ⚠️ **棄却の側だけが確実になるように作る。** 格子は連続な姿勢空間の標本でしかないので、
    「格子上で届かない」だけでは棄却できない。順運動学のリプシッツ定数
    （関節を δ 動かすと先端は高々 R·δ 動く）から**格子の隙間で届きうる余地**を見積もり、
    最小残距離がその余地を超えたときだけ「届かない」と言う。

    返り値: (判定できたか, 届くか, 最小残距離, 余地)
    """
    n = len(lengths)
    if n == 0 or n > 4:
        return False, True, 0.0, 0.0          # 想定外の構成では判定しない
    R = float(sum(lengths))
    if R <= 1e-9:
        return False, True, 0.0, 0.0
    grids = []
    for i in range(n):
        r = ranges_deg[i] if i < len(ranges_deg) and ranges_deg[i] else (-180.0, 180.0)
        grids.append(np.deg2rad(np.linspace(r[0], r[1], steps)))
    # 各関節の寄与は「その関節までの累積角」なので、順に積み上げる
    acc = grids[0]
    px = lengths[0] * np.sin(grids[0])
    py = lengths[0] * np.cos(grids[0])
    for i in range(1, n):
        acc = acc[:, None] + grids[i][None, :]
        px = px[:, None] + lengths[i] * np.sin(acc)
        py = py[:, None] + lengths[i] * np.cos(acc)
        acc = acc.reshape(-1); px = px.reshape(-1); py = py.reshape(-1)
    dist = np.hypot(px - d, py - dz)
    dmin = float(dist.min())
    # 格子の隙間で届きうる余地: 1 関節あたり半刻み × リプシッツ定数 R、n 関節ぶん
    step_rad = max(float(np.deg2rad((ranges_deg[i][1] - ranges_deg[i][0]) if i < len(ranges_deg)
                                    and ranges_deg[i] else 360.0) / (steps - 1))
                   for i in range(n))
    slack = R * step_rad * n / 2.0
    return True, dmin <= slack, dmin, slack


def layer1(geo, task, target, length_frozen=True, spread_y=0.0, offset_half=OFFSET_HALF):
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

    # Bug 23: リンク長の総和は「各関節が前のリンクの先端にある」ときしか実効リーチと一致しない。
    # 一致しない XML は、リンクが重なるか離れるかしており、判定の前提が崩れている。
    r_kin = kinematic_reach(lengths, geo.get('offsets', []))
    if abs(r_kin - r_now) > 1e-4:
        fatal = True
        findings.append(('fatal',
            f'**この XML は壊れています**（リンク長と関節の取り付け位置が矛盾）。\n'
            f'      リンク長の総和は {r_now:.4f} m ですが、関節位置の連鎖から出る'
            f'**実効リーチは {r_kin:.4f} m** しかありません。\n'
            f'      → リンクが互いに重なっています。XML の `body pos` を各リンクの長さに合わせてください。\n'
            f'      　（この不一致を見逃すと、届かない腕を「届く」と判定します。Bug 23 の再発）'))
        r_now = r_kin      # 以降の判定は実効値で行う
    if length_frozen:
        r_max = r_now
        findings.append(('info', 'リンク長は固定されているため、この長さのまま判定します'))
    else:
        r_max = max_reach_after_design(lengths, offset_half)
        findings.append(('warn', f'リンク長が最適化対象のため、設計図の長さだけでは判定できません。\n'
                                 f'      理論上の上限は約 {r_max:.3f} m ですが、これは**過大評価**です'
                                 f'（rrbot では同じ計算が 1.845 m を返す一方、実測の最大リーチは 1.44 m でした）。\n'
                                 f'      → 以下の判定は「届かない」側だけが確実で、「届く」側は保証できません。'))
    if r_min > 1e-6:
        findings.append(('info', f'到達可能な範囲は半径 {r_min:.3f}〜{r_max:.3f} m のドーナツ状'
                                 f'（1本のリンクが長すぎて中心付近には届きません）'))

    # 非平面アームでは「リンク長の総和＝水平リーチ」が成り立たない。根元の関節軸に平行なリンクは、
    # その軸まわりの回転では水平に伸びず**肩の高さを上げるだけ**である
    # （tripo_arm_v3 の根元 0.202 m の鉛直支柱がこれ。総和 1.002 m に対し水平限界は 0.797 m）。
    # 判定は変えず警告に留める: 正確な水平リーチは関節可動域込みの順運動学が要り、本層の範囲外。
    # ⚠️ geom の fromto は**そのボディのローカル系**なので、世界座標の軸と比べても意味がない。
    # 各リンクを「自分自身の関節軸」と比べる: 軸に平行なリンクはその関節では振れない。
    if not planar and geo.get('dirs') and len(axes) == len(lengths):
        par, perp, perp_rng = [], [], []
        rngs = (geo.get('ranges') or []) + [None] * len(lengths)
        for l, dv, ax, rg in zip(lengths, geo['dirs'], axes, rngs):
            a = np.array(ax, dtype=float)
            a = a / (np.linalg.norm(a) + 1e-12)
            if abs(float(np.dot(dv, a))) > 0.99:
                par.append(l)
            else:
                perp.append(l); perp_rng.append(rg)
        if par and perp:
            # ⚠️ **設計モードでは「寄与するリンクだけ」を伸ばす**（2026-09-11 修正、9-73）。
            #   旧実装は r_max に全リンクを伸ばした値を入れた直後、ここで**伸ばしていない**
            #   perp の合計から lim を出して min を取っていた。結果として
            #   **設計モードでも凍結モードと同じ判定になり**、「届かない側だけが確実」と
            #   表示しながら成長で救えるはずの形態を棄却していた（e2e_b1v で実測）。
            #   根元の鉛直リンクは伸ばしても水平到達に効かない（Bug 24 と同じ理由）ので、
            #   伸ばす対象は perp に限る。
            if length_frozen:
                l_h = sum(perp)
                shoulder_z = float(base[2]) + sum(par)
                grow_note = ''
            else:
                l_h = sum(float(np.hypot(l + offset_half, offset_half)) for l in perp)
                shoulder_z = float(base[2]) + sum(par)
                grow_note = (f'（水平に効くリンクだけを最大まで伸ばした場合。'
                             f'伸ばす前は {sum(perp):.3f} m）')
            msg = (f'この腕は非平面です（関節軸が複数種類）。**上の総リーチ {r_max:.3f} m は水平リーチではありません**。\n'
                   f'      根元の関節軸に平行なリンク {sum(par):.3f} m は肩の高さを上げるだけで、'
                   f'水平に伸びるのは残り {l_h:.3f} m です{grow_note}（肩の高さ {shoulder_z:.3f} m）。')
            if target is not None and len(target) >= 3:
                if length_frozen:
                    dz_s = abs(float(target[2]) - shoulder_z)
                else:
                    # 設計モードでは par も伸縮でき肩の高さを選べるので、
                    # **上限としては目標と同じ高さに置けると仮定する**（最も有利な側）。
                    dz_s = 0.0
                lim = float(np.sqrt(max(0.0, l_h ** 2 - dz_s ** 2)))
                msg += (f'\n      → 目標の高さ {float(target[2]):.3f} m での**水平到達限界は約 {lim:.3f} m**。'
                        f'以降の判定はこの値で行います。')
                # 総和で判定すると「届く」と誤答する（v3・目標 0.8 m がまさにこれ）。
                r_max = min(r_max, lim)

                # ⭐ B（9-72）: 可動域を入れた到達判定。**棄却の側だけが確実**になるよう、
                #   格子の分解能から保証できるときだけ「届かない」と言う。
                #   A（全長）は「伸ばせば届くか」を見るが、**関節が曲がらなければ届かない**
                #   ことは見ていない。両者は独立で、どちらかに掛かれば棄却する。
                d_b = float(np.linalg.norm(target[:2] - base[:2]))
                if task == 'pusher' and geo.get('cube') is not None:
                    d_b -= float(geo['cube']['half'])      # 対象は手前の面まで
                #   ⚠️ **設計モードでは B を適用しない。** A は「長いほど遠くへ届く」ので
                #   最大まで伸ばした値が最良ケースになるが、**B では長さは単調に有利ではない**。
                #   関節が ±90° しか曲がらないと、伸ばしすぎた腕は**畳みきれず近くに届かない**
                #   （実測: e2e_a1v_fix4 を最大まで伸ばすと肩から最短 2.14 m。目標 0.805 m に
                #   残り 1.35 m）。最適化器はもっと短い長さも選べるので、
                #   **最大値ひとつを根拠に棄却すると健全でない**。長さも探索するなら
                #   offset の空間も一緒に探す必要があり、本層の計算量を超える。
                ok_b, reach_b, dmin_b, slack_b = (
                    planar_chain_reach(list(perp), perp_rng, d_b, float(target[2]) - shoulder_z)
                    if length_frozen else (False, True, 0.0, 0.0))
                if ok_b and not reach_b:
                    fatal = True
                    findings.append(('fatal',
                        f'**関節の可動域では目標に届きません**（残り {dmin_b:.3f} m、'
                        f'格子の余地 {slack_b:.3f} m）。\n'
                        f'      腕の長さは足りていても、**関節が必要な角度まで曲がりません**。\n'
                        f'      → 可動域（XML の `joint range`）を広げるか、目標を動かしてください。'))
                elif ok_b:
                    findings.append(('info',
                        f'可動域を入れても目標に届く姿勢があります（残り {dmin_b:.3f} m）'))
            findings.append(('warn', msg))

    if task == 'reach':
        d = float(np.linalg.norm(target[:2] - base[:2]))
        dz = abs(float(target[2] - base[2]))
        findings.append(('info', f'目標までの水平距離: {d:.3f} m'))

        if d > r_max:
            fatal = True
            need = d / r_max
            findings.append(('fatal',
                f'**腕が短すぎて目標に届きません**（届く範囲は {r_max:.3f} m まで、目標は {d:.3f} m 先）。\n'
                f'      → 腕全体を **{scale_advice(need):.2f} 倍**に伸ばしてください。\n'
                f'      　（これが判定を覆すのに必要な最小の倍率です。'
                f'余裕を上乗せすると到達精度はむしろ単調に悪化します。実験系譜 9-11）'))
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
                    f'      → 実測では**余裕が大きいほど到達精度は単調に悪化**します'
                    f'（余裕 0 / 5.3 / 10 % で −0.27〜−0.33 / −1.19〜−1.41 / −1.66〜−3.42、各2seed。9-11）。\n'
                    f'      　⚠️ **機序は未解明です。**「慣性が増える」では説明できません'
                    f'（長さ 1.6 倍でも実効慣性の増加は 1.5 %。9-49）'))

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
                    f'      → 腕全体を **{scale_advice(near / r_max):.2f} 倍**に伸ばしてください。\n'
                    f'      　（これが判定を覆すのに必要な最小の倍率です。'
                    f'余裕を上乗せすると到達精度はむしろ単調に悪化します。実験系譜 9-11）'))
            elif r_max > near + 2 * half:
                findings.append(('warn',
                    f'静止状態で腕の先端が対象にめり込む可能性があります'
                    f'（腕 {r_max:.3f} m > 対象の奥の面 {near + 2*half:.3f} m）。\n'
                    f'      → 学習の起動時に `arm_safe_init=true` を付けて、腕を対象から逸らした姿勢で始めてください。'))
            else:
                findings.append(('ok', '対象に届き、かつ初期状態でめり込みません'))

            # 9-46: タスクが対象に分布を持たせる場合（Shot の cube_y_noise）、
            # 判定すべきは中央の一点ではなく**分布の最遠点**である。
            # 既定 0.0 なので、指定しない限り既存の判定は一切動かない。
            if spread_y > 1e-9:
                far = float(np.linalg.norm(
                    np.array([cpos[0] - base[0], cpos[1] + spread_y - base[1]]))) - half
                findings.append(('info',
                    f'このタスクは対象を y 方向に ±{spread_y:.2f} m ばらつかせます'
                    f'（分布の幅 {2*spread_y:.2f} m）。**判定は分布の最遠点で行います**。\n'
                    f'      中央 {near:.3f} m（伸展率 {near/r_max*100:.0f}%）'
                    f' → 端 {far:.3f} m（伸展率 {far/r_max*100:.0f}%）'))
                # 届く範囲に収まる最大のばらつき: |[dx, s]| - half <= r_max
                dx = float(cpos[0] - base[0])
                s_ok = float(np.sqrt(max(0.0, (r_max + half) ** 2 - dx ** 2)))
                if far > r_max:
                    fatal = True
                    findings.append(('fatal',
                        f'**対象が端に来たとき腕が届きません**'
                        f'（届く範囲 {r_max:.3f} m、端の対象は {far:.3f} m 先）。\n'
                        f'      中央だけなら届くので、**この形態はこのタスクには使えません**。\n'
                        f'      → 腕を **{scale_advice(far / r_max):.2f} 倍**に伸ばすか、'
                        f'対象のばらつきを **±{s_ok:.2f} m 以下**に狭めてください。'))
                elif far / r_max > 0.9:
                    findings.append(('warn',
                        f'端では腕をほぼ伸ばしきります（伸展率 {far/r_max*100:.0f}%）。\n'
                        f'      到達はしますが、**伸びきった姿勢で速度を出せるかは本層では判定できません**。\n'
                        f'      　（打つ・弾く系のタスクでは実測が要ります。実験系譜 9-46）'))
                else:
                    findings.append(('ok',
                        f'対象が端に来ても余裕があります（伸展率 {far/r_max*100:.0f}%）'))
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
    findings += _joint_usage(trace)
    return findings


# 使われていないと見なす閾値。可動域の何割しか動かなかったら報告するか。
UNUSED_FRAC = 0.10


def _joint_usage(trace):
    """各関節が可動域のどれだけを使ったかを報告する（2026-09-02 追加）。

    **動機**: 第1層は「腕を N 倍に伸ばせ」という反実仮想的な助言を返せるが、
    第2層・第3層は「何が起きたか」を報告できるだけで、
    「設計をどう変えれば結果が変わるか」を出せていなかった（3.12.5 節）。
    関節の使用率は、その穴を**設計を減らす方向**で部分的に埋める。
    使われていない関節が分かれば「固定してよい」と言えるからである。

    ⚠️ **断定はしない。** 「動いていない」と「動く必要がない」は違い、
    学習が不十分で動かせていないだけの可能性がある。第3層は学習の帰結を見る
    確率的な層なので、断定する権限を持たない（5.5.1 節「確率的な層は
    断定してはならない」）。したがって報告は警告に留め、成否の判断はしない。

    検証は第1層の助言と同じ閉ループで行う。すなわち提案どおり関節を固定した
    形態を作って再学習し、性能が落ちないことを確認する（4.4.2 節と同じ手順）。
    """
    use = trace.get('joint_use')
    if not use:
        return []
    findings = []
    detail = '、'.join(f'関節{i+1} {u*100:.0f}%' for i, u in enumerate(use))
    idle = [i for i, u in enumerate(use) if u < UNUSED_FRAC]
    if idle:
        names = '・'.join(f'関節{i+1}' for i in idle)
        findings.append(('warn',
            f'**{names} がほとんど動いていません**（{detail}）。\n'
            f'      → その関節を固定すれば、質量と制御の自由度を減らせる可能性があります。\n'
            f'      　（ただし「動かせていないだけ」の可能性もあります。'
            f'固定した形で学習し直して確かめてください）'))
    else:
        findings.append(('info',
            f'すべての関節が可動域を使っています（{detail}）。'))
    return findings


def _has_unreached(groups):
    """第3層が「達成できていない」旨を報告しているか。

    総合判定を出す権限は第1層のみだが（5.5.1）、第3層が未達を報告しているのに
    総合判定が「幾何的な障害はありません」だけだと**読み手には矛盾に見える**。
    判定は変えず、第3層の報告があることだけを添えるための判定子。
    """
    for title, findings in groups:
        if not title.startswith('第3層'):
            continue
        for level, _ in findings:
            if level == 'fatal':
                return True
    return False


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
    ap.add_argument('--target', nargs=3, type=float, default=None,
                    help='目標位置。既定は reach が [0.8,0,0.15]、'
                         'pusher は XML の対象位置（9-46）')
    ap.add_argument('--offset-half', type=float, default=None,
                    help='bone_offset の探索幅（片側）。run 指定時は cfg から自動取得。'
                         f'既定 {OFFSET_HALF}（9-72）')
    ap.add_argument('--spread-y', type=float, default=0.0,
                    help='対象の y 方向のばらつき（Shot の cube_y_noise）。'
                         '指定すると分布の最遠点で判定する（9-46）')
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
            # 9-46: pusher 系は目標を XML の対象位置から取る（既定 [0.8,0,0.15] は
            # 実際のパック高さ 0.2125 と食い違い、非平面の水平限界を誤らせていた）。
            es = cfgd.get('env_specs') or {}
            if not args.spread_y:
                args.spread_y = float(es.get('cube_y_noise', 0.0) or 0.0)
        # ⚠️ **保存された hydra config に `robot` は入っていない**（`cfg` という名前だけ）。
        #   旧実装は `cfgd.get('robot')` を見ており**常に空＝常に凍結モード**になっていた。
        #   長さを最適化する縦型の run でも凍結として判定していた（Bug 40）。
        #   正は `design_opt/cfg/<cfg>.yml`（探索範囲の写しは c2499fa で削除済み）。
        bp = {}
        cfg_file = os.path.join(os.getcwd(), 'design_opt', 'cfg', f'{cfgd.get("cfg")}.yml')
        if os.path.exists(cfg_file):
            bp = ((yaml.safe_load(open(cfg_file)) or {}).get('robot') or {}).get('body_params') or {}
        length_frozen = not bool(bp)
        # ⭐ 9-72: 探索幅はスクリプトに直書きせず**その run の cfg から読む**。
        #   直書きだと cfg を変えても判定器が追随せず、また食い違いが生まれる。
        if args.offset_half is None:
            ub = ((bp.get('offset') or {}).get('ub')) or []
            if ub:
                args.offset_half = max(abs(float(v)) for v in ub)

    geo = parse_arm_xml(os.path.join(ASSET_DIR, f'{args.xml}.xml'))
    if args.target is None:
        # 9-46: pusher 系の既定は XML の対象位置。以前の既定 [0.8,0,0.15] は
        # 実際の対象の高さと食い違い、非平面の水平限界を誤らせていた。
        # ⚠️ **明示的に --target が渡されたら上書きしない**（渡した値を黙って捨てない）。
        if args.task == 'pusher' and geo.get('cube') is not None:
            args.target = list(map(float, geo['cube']['pos']))
        else:
            args.target = [0.8, 0.0, 0.15]
    f1, fatal = layer1(geo, args.task, np.array(args.target, dtype=float),
                       length_frozen, spread_y=args.spread_y,
                       offset_half=(OFFSET_HALF if args.offset_half is None else args.offset_half))
    groups = [('第1層: 設計図だけで分かること（学習不要）', f1)]

    if run:
        groups += _run_layers23(run, args.task, geo)

    report(f'{args.xml} / タスク={args.task}', groups)
    print()
    if fatal:
        print('  → 総合判定: **この形ではタスクを達成できません。** 上の指摘に沿って形を直してください。')
    else:
        print('  → 総合判定: 幾何的な障害はありません。')
        # 第3層が未達を報告しているのに総合判定が「障害なし」だけだと矛盾して見える。
        # 判定を出す権限は第1層のみ（5.5.1「確率的な層は断定してはならない」）なので
        # 判定自体は変えず、**第3層の報告があることだけを添える**（2026-09-02）。
        if _has_unreached(groups):
            print('  　　ただし**第3層は、実際に動かした結果が未達であると報告しています**。')
            print('  　　幾何が通っていても学習や制御の側で解けていない場合があります。'
                  '第3層の記述を確認してください。')
        else:
            print('  　　学習を実行して性能を確かめてください。')
    print()
    # Bug 21: Choreonoid 経由（--python）だとレポート出力後もプロセスが終了せず、
    # 呼び出し側の timeout に当たるまで居座る。診断本体は数秒で終わっているので、
    # ここで明示的に落とす。`extract_gear.py` と同じ扱い。
    sys.stdout.flush()
    if run:
        os._exit(0)


def _run_layers23(run, task, geo=None):
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
    n_j = len((geo or {}).get('axes') or [])   # 腕の関節数（qpos の先頭がこの順に並ぶ）
    qhist = []
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
        try:                         # 関節角の履歴（第3層の使用率判定に使う）
            qhist.append(np.array(env.data.qpos[:n_j], dtype=float))
        except Exception:
            pass
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
    if qhist:
        Q = np.array(qhist)
        span = Q.max(axis=0) - Q.min(axis=0)          # 実際に振れた幅 [rad]
        lim = []
        for r in (geo or {}).get('ranges') or []:
            # XML の range は度。指定が無ければ ±180° とみなす
            lim.append(np.deg2rad(r[1] - r[0]) if r else 2 * np.pi)
        lim = np.array(lim[:len(span)]) if lim else np.full(len(span), 2 * np.pi)
        trace['joint_use'] = [float(min(1.0, s_ / l)) for s_, l in zip(span, lim) if l > 0]
    f3 = layer3(trace, task)
    return [('第2層: 学習が選んだ設計から分かること', f2),
            ('第3層: 実際の動きから分かること', f3)]


if __name__ == '__main__':
    main()
