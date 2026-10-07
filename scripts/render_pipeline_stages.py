#!/usr/bin/env python3
"""M3〜M5 の**処理の途中経過**を図にする（スライド用。2026-10-07、ユーザー依頼）。

⭐ 絵を描き起こすのではなく、**パイプラインの関数そのもの**（`glb_to_links.py` / `mesh_to_params.py`）を
  呼んで、実際に何が起きているかを描く。手描き A1 を例にする（修論 4.5 の E2E 実走の 1 例）。

  figures/pipeline_m2_pose.png  真横に描いた絵（A1）と斜めに描いた絵（A3）の 3D 化の比較（M2 の失敗例）
  figures/pipeline_m3.png   GLB の頂点 → マゼンタの頂点と関節（重心）→ 関節の高さで切ったリンク
  figures/pipeline_m4.png   リンクごとのメッシュとカプセル（長さ・太さ・はみ出し）
  figures/pipeline_m5.png   XML のカプセル模型（初期姿勢）と、関節軸の割り当て

    python3 scripts/render_pipeline_stages.py              # A1
    python3 scripts/render_pipeline_stages.py --case B2    # 別の手描き（data/test/<case>/meshes が要る）

⚠️ M4・M5 は**いまの** `mesh_to_params.py` → `topology_to_xml.py` で作り直してから描く（data/test/A1 の topology.json・e2e_a1v.xml は 9-75 より前の抽出器の版で、半径が違う）。
"""
import argparse
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glb_to_links as G       # noqa: E402
import mesh_to_params as MP    # noqa: E402

# ⚠️ 名前で指定するだけでは見つからず豆腐になる。render_arm_video.py と同じく Noto CJK を明示登録する
for _ttc in sorted(pathlib.Path('/usr/share/fonts').rglob('*CJK*')):
    try:
        matplotlib.font_manager.fontManager.addfont(str(_ttc))
    except Exception:
        pass
for _f in ('Noto Sans CJK JP', 'Noto Serif CJK JP'):
    if any(_x.name == _f for _x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = _f
        break
COL = ['#888888', '#d94040', '#3070d0', '#30a050', '#d0a020']   # 台座・上腕・前腕・先端
JA = {'base': '台座', 'upper_arm': '上腕', 'forearm': '前腕', 'hand': '先端'}


def m2_pose(out):
    """真横に描いた A1 と、斜め（3/4）に描いた A3 の 3D 化を並べる（系譜 9-23）。

    ⭐ A3 は Tripo3D が斜めの線を「奥へ伸びる」と解釈し、腕が寝た（関節がほぼ同じ高さに並ぶ）。
    """
    from PIL import Image
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, case in enumerate(['A1', 'A3']):
        ax[row, 0].imshow(Image.open(f'data/test/{case}/sketch/{case}_hand.webp')); ax[row, 0].axis('off')
        ax[row, 1].imshow(Image.open(f'data/test/{case}/sketch/{case}_m1.jpeg')); ax[row, 1].axis('off')
        mesh = G._apply_yup_zup(G._load_concat(f'data/test/{case}/3D/{case}.glb'))
        V = np.asarray(mesh.vertices); C = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255
        idx = np.random.default_rng(0).choice(len(V), min(30000, len(V)), replace=False)
        a = ax[row, 2]
        # 横（x-z）と奥行き（y-z）のうち、腕が長く見える方を描く
        k = 0 if np.ptp(V[:, 0]) >= np.ptp(V[:, 1]) else 1
        a.scatter(V[idx, k], V[idx, 2], c=C[idx], s=0.5)
        a.set_aspect('equal'); a.grid(alpha=0.3); a.set_xlabel('横 [m]'); a.set_ylabel('高さ z [m]')
        a.set_title(f'3D（glb）を横から\n高さの広がり {np.ptp(V[:, 2]):.2f} m')
        ax[row, 0].set_title(f'{case} 手描き（{"真横" if case == "A1" else "斜め 3/4"}）')
        ax[row, 1].set_title('M1 の整形画像（マーカー 3 個）')
    fig.suptitle('M2 の失敗例 — 斜めに描くと、Tripo3D が斜線を「奥へ伸びる」と読み、腕が寝る（A3。2 回とも）', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f'[M2] → {out}')


def m3(case, names, out):
    """GLB → マーカー → 関節 → 分割。glb_to_links.main() と同じ順に関数を呼ぶ。"""
    glb = f'data/test/{case}/3D/{case}.glb'
    mesh = G._apply_yup_zup(G._load_concat(glb))
    color, tol = G._auto_calibrate_joint_color(mesh)
    mask = G._marker_mask(mesh, color, tol)
    joints = G._detect_joints_3d(mesh, color, tol)
    segs = G._split_by_z(mesh, [float(p[2]) for p in joints])
    # ⭐ パイプラインが実際に出した関節座標（joints.json）と一致するかを確かめてから描く
    saved = np.asarray(json.load(open(f'data/test/{case}/meshes/joints.json'))['joint_positions'])
    err = np.abs(np.asarray(joints) - saved).max()
    print(f'[M3] 関節 {len(joints)} 個。joints.json との差の最大 {err*1000:.3f} mm')
    # ⚠️ A1 の joints.json は関数の改修前に作られており 0.5 mm ずれる（図には影響しない量）。2 mm を超えたら止める
    assert err < 2e-3, 'パイプラインの出力と 2 mm 以上ずれる（関数か入力が変わった）'

    V = np.asarray(mesh.vertices); C = np.asarray(mesh.visual.vertex_colors[:, :3]) / 255
    rng = np.random.default_rng(0); idx = rng.choice(len(V), min(40000, len(V)), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.2))
    ax[0].scatter(V[idx, 0], V[idx, 2], c=C[idx], s=0.6)
    ax[0].set_title('① 3D モデル（glb）の頂点\nテクスチャの色を頂点に移したもの')
    ax[1].scatter(V[idx, 0], V[idx, 2], c='#dddddd', s=0.4)
    ax[1].scatter(V[mask, 0], V[mask, 2], c='#ff00ff', s=1.2)
    for k, p in enumerate(joints):
        ax[1].plot(p[0], p[2], 'kx', ms=12, mew=2.5)
        ax[1].annotate(f'関節{k+1}', (p[0], p[2]), xytext=(10, -4), textcoords='offset points', fontsize=11)
        for a in ax[1:]:
            a.axhline(p[2], color='k', ls='--', lw=0.8)
    ax[1].set_title(f'② マゼンタの頂点（色 {color}・許容幅 {tol}）\n'
                    '3 次元で塊に分け、重心＝関節（×）。小さい塊（全体の 2 % 未満）は捨てる')
    for k, s in enumerate(segs):
        sv = np.asarray(s.vertices)
        ax[2].scatter(sv[:, 0], sv[:, 2], c=COL[k % len(COL)], s=0.6)
        c = sv.mean(axis=0)
        ax[2].annotate(JA.get(names[k], names[k]), (c[0], c[2]), xytext=(25, 0), textcoords='offset points',
                       fontsize=12, color=COL[k % len(COL)], weight='bold')
    ax[2].set_title('③ 関節の高さ（破線）でメッシュを切る\n→ リンクごとの STL')
    fig.text(0.99, 0.01, f'関節位置: 実行時の joints.json との差 最大 {err*1000:.1f} mm', ha='right', fontsize=8, color='gray')
    for a in ax:
        a.set_aspect('equal'); a.set_xlabel('x [m]'); a.grid(alpha=0.3)
    ax[0].set_ylabel('z（高さ）[m]')
    fig.suptitle('M3 関節検出とリンク分割（手描き A1）', fontsize=15)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'[M3] → {out}')


def m4(case, names, out):
    """リンクごとに、メッシュとカプセル（mesh_to_params と同じ軸・半径・はみ出しの計算）。"""
    mdir = f'data/test/{case}/meshes'
    fo = json.load(open(f'{mdir}/joints.json'))['frame_origins']
    lengths = json.load(open(f'{mdir}/joints.json')).get('link_lengths', {})
    fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 5.6))
    for i, n in enumerate(names):
        mesh = MP.load_mesh(f'{mdir}/{n}.stl')
        u = MP.bone_axis(names, i, fo, mesh); u = u / np.linalg.norm(u)
        V = np.asarray(mesh.vertices)
        # ⭐ 値は mesh_to_params.build_topology と同じ計算（半径は 4 桁に丸める・先端の長さは OBB）
        L = float(lengths[n]) if n in lengths else float(MP.obb_params(mesh, 1.0)['length'])
        r = round(MP.perp_radius(mesh, u, 1.0), 4)
        over = MP.capsule_overhang(mesh, u, L, r, 1.0)
        # ⭐ 横から見た断面（骨からの距離 × 骨に沿った位置）で描く。単純な投影だと奥行き方向の
        #   はみ出しが隠れて「カプセルの内側なのに黒い」点になる
        v = np.cross(u, [0, 1, 0] if abs(u[1]) < 0.9 else [1, 0, 0]); v /= np.linalg.norm(v)
        t = V @ u
        radial = np.linalg.norm(V - np.outer(t, u), axis=1) * np.sign(V @ v + 1e-12)
        tc = np.clip(t, 0, L); out_m = np.linalg.norm(V - np.outer(tc, u), axis=1) > r
        a = ax[i]
        a.scatter(radial[~out_m], t[~out_m], c=COL[i % len(COL)], s=0.6, alpha=0.5)
        a.scatter(radial[out_m], t[out_m], c='k', s=0.8)
        th = np.linspace(0, np.pi, 40)
        cap = np.r_[np.c_[r * np.ones(2), [0, L]], np.c_[r * np.cos(th), L + r * np.sin(th)],
                    np.c_[-r * np.ones(2), [L, 0]], np.c_[-r * np.cos(th), -r * np.sin(th)]]
        a.plot(cap[:, 0], cap[:, 1], 'k-', lw=1.5)
        a.plot([0, 0], [0, L], 'k--', lw=1)
        a.plot(0, 0, 'm^', ms=10)
        a.set_title(f'{JA.get(n, n)}\n長さ {L:.3f} m・太さ（半径）{r:.3f} m\nはみ出し {over:.1f} %（黒い点）')
        a.set_aspect('equal'); a.grid(alpha=0.3); a.set_xlabel('骨からの距離 [m]')
    ax[0].set_ylabel('骨の軸に沿った距離 [m]（▲＝自分の関節）')
    fig.suptitle('M4 長さと太さの抽出 — 長さ＝関節と関節の距離、太さ＝骨に垂直な断面の幅（黒線がカプセル）', fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'[M4] → {out}')


def m5(xml_path, names, out):
    """XML のカプセル模型（初期姿勢＝全関節 0°）を、XML の入れ子どおりに積んで描く。"""
    root = ET.parse(xml_path).getroot()
    body0 = root.find('worldbody/body')
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1, 1.5]})
    a = ax[0]
    base = np.array(list(map(float, body0.get('pos').split())))
    p, k, b = base.copy(), 0, body0.find('body')
    th = np.linspace(0, 2 * np.pi, 60)
    while b is not None:
        p = p + np.array(list(map(float, b.get('pos').split())))
        g, j = b.find('geom'), b.find('joint')
        ft = np.array(list(map(float, g.get('fromto').split())))
        r = float(g.get('size'))
        q0, q1 = p + ft[:3], p + ft[3:]
        a.add_patch(plt.Rectangle((q0[0] - r, q0[2]), 2 * r, q1[2] - q0[2], color=COL[k % len(COL)], alpha=0.75))
        for q in (q0, q1):
            a.add_patch(plt.Circle((q[0], q[2]), r, color=COL[k % len(COL)], alpha=0.75))
        ax_v = list(map(float, j.get('axis').split()))
        kind = 'ヨー（鉛直軸）' if ax_v[2] else 'ピッチ（水平軸）'
        a.plot(p[0], p[2], 'ko', ms=6)
        a.annotate(f'{JA.get(names[k], names[k])}：{kind}  ±{j.get("range").split()[1].lstrip("-")}°\n'
                   f'長さ {q1[2]-q0[2]:.3f}・半径 {r:.3f}',
                   (p[0] + r, (q0[2] + q1[2]) / 2), xytext=(18, 0), textcoords='offset points', fontsize=10.5)
        k += 1
        b = b.find('body')
    a.set_xlim(-0.4, 1.2); a.set_ylim(-0.05, p[2] + 0.45); a.set_aspect('equal'); a.grid(alpha=0.3)
    a.set_xlabel('x [m]'); a.set_ylabel('z [m]')
    a.set_title('XML の模型（初期姿勢：全関節 0°）\n※ 描いた角度は持ち込まない。長さと太さだけ')
    b1 = body0.find('body'); b2 = b1.find('body'); b3 = b2.find('body')
    def _g(b):
        g = b.find('geom'); return g.get('fromto').split()[-1], g.get('size')
    (l1, r1), (l2, r2) = _g(b1), _g(b2)
    snippet = (f'<body name="{b1.get("name")}" pos="0 0 0">          ← 台座\n'
               f'  <joint axis="0 0 1" range="-180 180"/>     ← ヨー\n'
               f'  <geom type="capsule" fromto="0 0 0  0 0 {l1}" size="{r1}"/>\n'
               f'  <body name="{b2.get("name")}" pos="0 0 {l1}">     ← 親の先端に取り付け\n'
               f'    <joint axis="0 1 0" range="-90 90"/>     ← ピッチ\n'
               f'    <geom type="capsule" fromto="0 0 0  0 0 {l2}" size="{r2}"/>\n'
               f'    <body name="{b3.get("name")}" pos="0 0 {l2}">  ← 親の先端に取り付け\n'
               '      ...\n    </body>\n  </body>\n</body>\n<actuator>\n'
               '  <motor joint="1_joint" gear="100" ctrlrange="-1 1"/>  ← ギア比つきモーター\n  ...')
    ax[1].axis('off')
    ax[1].text(0, 1, snippet, fontsize=9.5, va='top', transform=ax[1].transAxes)
    ax[1].set_title('XML（抜粋）— 子の <body> を親の中に入れ子にし、\npos＝親の骨の長さ にすると「先端に取り付け」になる', loc='left')
    fig.suptitle('M5 シミュレーションモデル生成（topology.json → XML）', fontsize=15)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'[M5] → {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='A1')
    ap.add_argument('--out-dir', default='figures')
    a = ap.parse_args()
    names = ['base', 'upper_arm', 'forearm', 'hand']
    # ⭐ data/test の topology / XML は古い抽出器の版がある（9-75 前）。図の値を揃えるため、
    #   **いまの** mesh_to_params → topology_to_xml を一時ディレクトリで回し、その XML を描く
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp(prefix='pipeline_fig_')
    mdir = f'data/test/{a.case}/meshes'
    subprocess.run([sys.executable, 'scripts/mesh_to_params.py', '--parts',
                    *[f'{mdir}/{n}.stl' for n in names], '--names', *names,
                    '--joints-json', f'{mdir}/joints.json', '--vertical',
                    '--output', f'{tmp}/topology.json'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, 'scripts/topology_to_xml.py', '--topology', f'{tmp}/topology.json',
                    '--output', f'{tmp}/model.xml', '--no-cube'], check=True, stdout=subprocess.DEVNULL)
    m2_pose(f'{a.out_dir}/pipeline_m2_pose.png')
    m3(a.case, names, f'{a.out_dir}/pipeline_m3.png')
    m4(a.case, names, f'{a.out_dir}/pipeline_m4.png')
    m5(f'{tmp}/model.xml', names, f'{a.out_dir}/pipeline_m5.png')


if __name__ == '__main__':
    main()
