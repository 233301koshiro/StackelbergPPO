"""軸6 L-6: 部品で組んだ形（build_arm3.py の部品の json）を、run の best が箱を打つ瞬間の姿勢にして、真横から並べて描く。

    python3 scripts/render_ra_forms.py <出力.png> <見出し> <部品の json> <run> [<見出し> <部品の json> <run> ...]

姿勢: <run>/trace/arm_trace.npz の 1 話目で、箱が初めて 2 mm 動いた step の 1 つ前。上腕・前腕・手先の
「腕の平面の中での仰角」をシミュレーションの関節の位置から取り、部品の腕の各サーボの軸まわりに回して合わせる
（関節の角の符号の約束に頼らない）。旋回（関節 1）は描かない（真横から見るので平面の中だけ）。
箱はシミュレーションの位置（関節 1 の軸からの前方の距離と床からの高さ）に 60 mm の立方体で置く。
同じ step のシミュレーションのカプセル（上腕・前腕・手先）を青の点線の輪郭で重ね、真横から見た箱との隙間を
部品の腕とカプセルで並べて出す（高さは床から。部品の腕は関節 1 のサーボの原点を床から 39.1 mm に置く）。
部品の腕はシミュレーションのモデルと台→肩の高さ・手先の形が違うので、手先と箱がぴったり接するとは限らない。
質量: 上腕・前腕の構造（Hinge＋LEGO＋Frame）の質量を部品の mass-param から足し、build_arm3.py の json の値と並べる。
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import yaml
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NotoSerifCJK-Regular' in f:
        font_manager.fontManager.addfont(f)
names = [f.name for f in font_manager.fontManager.ttflist if 'CJK JP' in f.name]
plt.rcParams['font.family'] = sorted(names, key=lambda n: 'Sans' not in n)[0] if names else 'DejaVu Sans'

PROJ = '/choreonoid_ws/install/share/choreonoid-2.3/robot_assembler/irsl'
P = {p['type']: p for p in yaml.safe_load(open(PROJ + '/irsl_assembler_config.yaml'))['PartsSettings']}
FLOOR = -39.1        # 関節 1 のサーボの原点から床まで（render_arm.py と同じ。XML の台の高さ 39 mm）
LIGHT = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
_meshes = {}


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def mesh(v, Tp):
    if 'box' in v:
        m = trimesh.creation.box(extents=v['box'])
    else:
        if v['url'] not in _meshes:
            _meshes[v['url']] = trimesh.load(v['url'], force='mesh', process=False)
        m = _meshes[v['url']].copy()
        m.apply_scale(v['scale_mm'])
    m.apply_transform(Tp @ np.array(v['T']))
    return m


def sim_pose(run):
    """1 話目で箱が動き出す直前の、上腕・前腕・手先の仰角（rad）と、関節 1 の軸から見た箱の前方の距離・高さ（mm）。"""
    d = np.load(f'{run}/trace/arm_trace.npz', allow_pickle=True)
    xp = np.asarray(d['ep_xpos'][0], float)
    xm = np.asarray(d['ep_xmat'][0], float)
    bo = np.asarray(d['ep_bone'][0], float)
    cube = np.asarray(d['ep_cubes'][0], float)
    t = max(int(np.where(np.linalg.norm(cube - cube[0], axis=1) > 0.002)[0][0]) - 1, 0)
    names = [str(n) for n in d['body_names']]
    i1, i2, i3, i4 = (names.index(n) for n in ('1', '11', '111', '1111'))
    tip = xp[t, i4] + xm[t, i4] @ bo[i4]
    hinge = xm[t, i2][:, 1]                       # 関節 2〜4 の軸（y 軸のヒンジ）
    f = np.cross(hinge, [0, 0, 1.0])
    f[2] = 0
    f /= np.linalg.norm(f)
    if f @ (cube[t] - xp[t, i1]) < 0:
        f = -f
    pts = [xp[t, i2], xp[t, i3], xp[t, i4], tip]
    elev = [np.arctan2(b[2] - a[2], (b - a) @ f) for a, b in zip(pts[:-1], pts[1:])]
    c = cube[t] - xp[t, i1]
    cap = np.asarray(d['ep_cap'][0], float)
    rad = np.asarray(d['ep_geom'][0], float).ravel()
    to2d = lambda q: np.array([(q - xp[t, i1]) @ f * 1000, q[2] * 1000])   # 関節 1 の軸から前方・床から上（mm）
    caps = [(to2d(xp[t, i] + xm[t, i] @ cap[i, :3]), to2d(xp[t, i] + xm[t, i] @ cap[i, 3:]), rad[i] * 1000, names[i])
            for i in (i1, i2, i3, i4)]
    return t, elev, (c @ f * 1000, cube[t][2] * 1000, c @ np.cross([0, 0, 1.0], f) * 1000), caps   # 箱の中心の高さは床（XML の z = 0）から


def pose_parts(parts, elev):
    T = [np.array(p['T']) for p in parts]
    servo = [i for i, p in enumerate(parts) if p['servo']]

    def apply(k, ang):
        i0 = servo[k]
        o, ax = T[i0][:3, 3].copy(), T[i0][:3, 2].copy()
        M = np.eye(4)
        M[:3, :3] = rot(ax, ang)
        M[:3, 3] = o - M[:3, :3] @ o
        for i in range(i0 + 1, len(T)):
            T[i] = M @ T[i]

    def tip():
        o, ax = T[servo[-1]][:3, 3], T[servo[-1]][:3, 2]
        v = np.vstack([mesh(vv, T[i]).vertices for i in range(servo[-1] + 1, len(parts)) for vv in parts[i]['visual']])
        w = v - o
        return v[np.argmax(np.linalg.norm(w - np.outer(w @ ax, ax), axis=1))]

    def signed(u, w, ax):
        u = u - (u @ ax) * ax
        w = w - (w @ ax) * ax
        return np.arctan2(np.cross(u, w) @ ax, u @ w)

    ax2 = T[servo[1]][:3, 2]
    fwd = np.cross(ax2, [0, 0, 1.0])
    if fwd @ (T[servo[2]][:3, 3] - T[servo[1]][:3, 3]) < 0:
        fwd = -fwd
    for k, el in zip((1, 2, 3), elev):
        a = T[servo[k]][:3, 3]
        b = T[servo[k + 1]][:3, 3] if k + 1 < len(servo) else tip()
        want = np.cos(el) * fwd + np.sin(el) * np.array([0, 0, 1.0])
        apply(k, signed(b - a, want, T[servo[k]][:3, 2]))
    # 仰角が合ったかを出す（⑤-3-2: 期待と実測を並べる）
    J = [T[i][:3, 3] for i in servo] + [tip()]
    got = [np.arctan2(b[2] - a[2], (b - a) @ fwd) for a, b in zip(J[1:-1], J[2:])]
    return T, fwd, J, got


def stadium(a, b, r, n=24):
    """線分 a→b を半径 r で太らせた輪郭（カプセルを真横から見た形）。"""
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    th0 = np.arctan2(u[1], u[0])
    arc = lambda c, s: c + r * np.stack([np.cos(s), np.sin(s)], 1)
    return np.vstack([arc(b, np.linspace(th0 - np.pi / 2, th0 + np.pi / 2, n)), arc(a, np.linspace(th0 + np.pi / 2, th0 + 3 * np.pi / 2, n))])


def box_gap(xy, cx, cz, h=30.0):
    """真横から見た点と箱（中心 cx, cz、半辺 h）の距離の最小（mm）。中に入っていれば 0。"""
    q = np.stack([np.clip(xy[:, 0], cx - h, cx + h), np.clip(xy[:, 1], cz - h, cz + h)], 1)
    return float(np.linalg.norm(xy - q, axis=1).min())


def struct_mass(parts, tag):
    return sum(P[p['type']]['mass-param']['mass'] for p in parts if p['name'].startswith(tag))   # 設定は g


def main():
    out, args = sys.argv[1], sys.argv[2:]
    forms = [args[i:i + 3] for i in range(0, len(args), 3)]
    drawn = []
    for label, pj, run in forms:
        d = json.load(open(pj))
        parts = d['parts']
        t, elev, (cx, cz, cy), caps = sim_pose(run)
        T, fwd, J, got = pose_parts(parts, elev)
        side = np.cross(fwd, [0, 0, 1.0])          # 見る人の側（画面の右が前方・上が z）
        polys, cols, depth, gaps = [], [], [], {}
        for i, p in enumerate(parts):
            for v in p['visual']:
                m = mesh(v, T[i])
                c = np.array(v['color'][:3]) if v.get('color') else (np.array([0.18, 0.18, 0.2]) if p['servo'] else np.array([0.88, 0.88, 0.86]))
                shade = 0.45 + 0.55 * np.clip(np.abs(m.face_normals @ LIGHT), 0, 1)
                tri = m.vertices[m.faces]
                polys.append(np.stack([tri @ fwd, tri[..., 2] - FLOOR], -1))   # 高さは床から
                if i >= 3:   # J1・肩フレーム・J2（動かない根元）は除く
                    gaps[p['name']] = min(gaps.get(p['name'], 1e9), box_gap(polys[-1].reshape(-1, 2), cx, cz))
                cols.append(np.clip(c[None, :] * shade[:, None], 0, 1))
                depth.append((tri @ side).mean(1))
        polys, cols, depth = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
        o = np.argsort(depth)
        J = np.array(J)
        J[:, 2] -= FLOOR
        g_parts = min(gaps.items(), key=lambda kv: kv[1])
        g_caps = [(nm, box_gap(stadium(a, b, r), cx, cz)) for a, b, r, nm in caps]
        print(f'{label}: step {t}（1 話目）  仰角 シミュレーション {np.degrees(elev).round(1)} / 部品 {np.degrees(got).round(1)} °  '
              f'箱 前方 {cx:.0f} mm・中心の高さ {cz:.0f} mm・平面からの横ずれ {cy:.0f} mm  三角形 {len(polys)}')
        print(f'  真横から見た箱との隙間: 部品の腕で最も近い {g_parts[0]} {g_parts[1]:.1f} mm ／ シミュレーションのカプセル '
              + '・'.join(f'{nm} {g:.1f}' for nm, g in g_caps) + ' mm')
        for k, tag in enumerate(('リンク1', 'リンク2')):
            ln = d['links'][k]
            print(f'  {tag}: 組めた {ln["組めた_mm"]} mm  {ln["Hinge"]} 段 {ln["段"] or "なし"} {ln["Frame"]}  構造の質量（部品の和）{struct_mass(parts, tag):.2f} g')
        drawn.append((label, polys[o], cols[o], J @ fwd, J[:, 2], cx, cz, d['links'], caps))

    allx = np.concatenate([p[1][..., 0].ravel() for p in drawn] + [[p[5] + 30 for p in drawn]])
    allz = np.concatenate([p[1][..., 1].ravel() for p in drawn])
    x0, x1 = allx.min() - 15, allx.max() + 15
    z0, z1 = -6, allz.max() + 15
    w = 9.6 / len(drawn)              # 修論の版面に貼る図の幅は 8〜9.6 in（第4章の執筆メモ）
    fig, axs = plt.subplots(1, len(drawn), figsize=(9.6, w * (z1 - z0) / (x1 - x0) + 1.3), dpi=300)
    for ax, (label, pol, col, jx, jz, cx, cz, links, caps) in zip(np.atleast_1d(axs), drawn):
        ax.add_patch(Rectangle((x0, z0), x1 - x0, -z0, color=(0.85, 0.85, 0.82), zorder=0))
        ax.add_patch(Rectangle((cx - 30, cz - 30), 60, 60, facecolor=(0.95, 0.55, 0.45), edgecolor='k', lw=0.5, zorder=1))
        ax.add_collection(PolyCollection(pol, facecolors=col, edgecolors='none', zorder=2))
        for a, b, r, _ in caps[1:]:
            st = stadium(a, b, r)
            ax.plot(*np.vstack([st, st[:1]]).T, color='tab:blue', lw=0.8, ls=':', zorder=5)
        ax.plot(jx, jz, color='tab:red', lw=1.0, ls='--', zorder=3)
        ax.scatter(jx[1:-1], jz[1:-1], color='tab:red', s=8, zorder=4)
        ax.set_xlim(x0, x1); ax.set_ylim(z0, z1); ax.set_aspect('equal')
        ax.set_title(f'{label}\n上腕 {links[0]["組めた_mm"]:.1f} mm・前腕 {links[1]["組めた_mm"]:.1f} mm', fontsize=10)
        ax.set_xlabel('前方 [mm]', fontsize=9); ax.tick_params(labelsize=8)
    np.atleast_1d(axs)[0].set_ylabel('高さ [mm]', fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color='tab:red', ls='--', lw=1.0), Line2D([], [], color='tab:blue', ls=':', lw=1.0),
                Rectangle((0, 0), 1, 1, facecolor=(0.95, 0.55, 0.45), edgecolor='k', lw=0.5)],
               ['関節を結ぶ線（部品の腕）', 'シミュレーションのカプセル（同じ時刻）', '箱'],
               loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    print('書き出した:', out)


if __name__ == '__main__':
    main()
