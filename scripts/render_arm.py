"""build_bow.py が書いた部品の姿勢から、お辞儀の姿勢にして描く（左: RobotAssembler の部品、右: 同じ姿勢の 1/5 のシミュレーションのカプセル）。"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NotoSerifCJK-Regular' in f:
        font_manager.fontManager.addfont(f)
names = [f.name for f in font_manager.fontManager.ttflist if 'CJK JP' in f.name]
plt.rcParams['font.family'] = sorted(names, key=lambda n: 'Sans' not in n)[0] if names else 'DejaVu Sans'

ELEV = np.radians([65.0, 5.0, -60.0])   # リンク 1・2・手先の向き（水平からの仰角）。手先が下を向く＝お辞儀
YAW = np.radians(0.0)

IN = os.environ.get('IN', 'bow_parts.json')
SCALE = float(os.environ.get('SCALE', '1.0'))
d = json.load(open(os.path.join(HERE, IN)))
parts = d['parts']
T = [np.array(p['T']) for p in parts]
servo_idx = [i for i, p in enumerate(parts) if p['servo']]


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def apply(k, ang):
    """k 番目のサーボのホーンより先の部品をすべて、ホーンの軸まわりに回す。"""
    i0 = servo_idx[k]
    o, ax = T[i0][:3, 3].copy(), T[i0][:3, 2].copy()
    M = np.eye(4)
    M[:3, :3] = rot(ax, ang)
    M[:3, 3] = o - M[:3, :3] @ o
    for i in range(i0 + 1, len(T)):
        T[i] = M @ T[i]


meshes = {}


def mesh(v):
    if 'box' in v:
        m = trimesh.creation.box(extents=v['box'])
        m.apply_transform(np.array(v['T']))
        return m
    if v['url'] not in meshes:
        meshes[v['url']] = trimesh.load(v['url'], force='mesh', process=False)
    m = meshes[v['url']].copy()
    m.apply_scale(v['scale_mm'])
    m.apply_transform(np.array(v['T']))
    return m


def tip_point():
    o, ax = T[servo_idx[-1]][:3, 3], T[servo_idx[-1]][:3, 2]
    vs = []
    for i in range(servo_idx[-1] + 1, len(parts)):
        for v in parts[i]['visual']:
            m = mesh(v)
            m.apply_transform(T[i])
            vs.append(m.vertices)
    v = np.vstack(vs)
    w = v - o
    perp = w - np.outer(w @ ax, ax)
    return v[np.argmax(np.linalg.norm(perp, axis=1))]


def signed(u, w, ax):
    u = u - (u @ ax) * ax
    w = w - (w @ ax) * ax
    return np.arctan2(np.cross(u, w) @ ax, u @ w)


# 腕が伸びる平面の前方向（J2 → J3 を軸に垂直な水平成分で）
ax2 = T[servo_idx[1]][:3, 2]
fwd = np.cross(ax2, [0, 0, 1.0])
v0 = T[servo_idx[2]][:3, 3] - T[servo_idx[1]][:3, 3]
if fwd @ v0 < 0:
    fwd = -fwd
for k, el in zip((1, 2, 3), ELEV):
    a = T[servo_idx[k]][:3, 3]
    b = T[servo_idx[k + 1]][:3, 3] if k + 1 < len(servo_idx) else tip_point()
    want = np.cos(el) * fwd + np.sin(el) * np.array([0, 0, 1.0])
    apply(k, signed(b - a, want, T[servo_idx[k]][:3, 2]))
apply(0, YAW)

joints = [T[i][:3, 3] for i in servo_idx] + [tip_point()]
floor = -39.1

LIGHT = np.array([0.4, -0.6, 0.7])
LIGHT /= np.linalg.norm(LIGHT)


def faces_of(i, p):
    out = []
    for v in p['visual']:
        m = mesh(v)
        m.apply_transform(T[i])
        if v.get('color'):
            c = np.array(v['color'][:3])
        elif p['servo']:
            c = np.array([0.18, 0.18, 0.2])
        else:
            c = np.array([0.88, 0.88, 0.86])
        n = m.face_normals
        shade = 0.45 + 0.55 * np.clip(np.abs(n @ LIGHT), 0, 1)
        out.append((m.vertices[m.faces], np.clip(c[None, :] * shade[:, None], 0, 1)))
    return out


def setup(ax, title, lim):
    ax.set_title(title, fontsize=11)
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=18, azim=-35)
    ax.set_xlabel('x [mm]', fontsize=8); ax.set_ylabel('y [mm]', fontsize=8); ax.set_zlabel('z [mm]', fontsize=8)
    ax.tick_params(labelsize=7)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
    ax.plot_surface(gx, gy, np.full_like(gx, floor), color=(0.93, 0.93, 0.9), alpha=0.5, zorder=-1)


allv = np.vstack([np.vstack([f.reshape(-1, 3) for f, _ in faces_of(i, p)]) for i, p in enumerate(parts)])
lo, hi = allv.min(0) - 15, allv.max(0) + 15
c = (lo + hi) / 2
span = (hi - lo).max() / 2
lim = [(c[0] - span * 0.6, c[0] + span * 0.6), (c[1] - span, c[1] + span), (floor, floor + 2 * span)]

fig = plt.figure(figsize=(17, 6.4), dpi=150)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
setup(ax1, 'RobotAssembler の部品（IRSL）で組んだ形 — XL430 × 4', lim)
F, C = [], []
for i, p in enumerate(parts):
    for f, col in faces_of(i, p):
        F.append(f); C.append(col)
pc = Poly3DCollection(np.concatenate(F), facecolors=np.concatenate(C), edgecolors='none', linewidths=0)
ax1.add_collection3d(pc)
J = np.array(joints)
ax1.plot(J[:, 0], J[:, 1], J[:, 2], color='tab:red', lw=1.2, ls='--', zorder=10)
ax1.scatter(J[:-1, 0], J[:-1, 1], J[:-1, 2], color='tab:red', s=12, zorder=11)
axs = fig.add_subplot(1, 3, 2, projection='3d')
setup(axs, '同じ形を真横から（関節の軸の向きから）', lim)
axs.view_init(elev=0, azim=0)
axs.add_collection3d(Poly3DCollection(np.concatenate(F), facecolors=np.concatenate(C), edgecolors='none', linewidths=0))
axs.plot(J[:, 0] + 60, J[:, 1], J[:, 2], color='tab:red', lw=1.2, ls='--', zorder=10)
axs.scatter(J[:-1, 0] + 60, J[:-1, 1], J[:-1, 2], color='tab:red', s=12, zorder=11)

# 右: 同じ関節の位置で、1/5 の A1v のカプセル（シミュレーションのモデル）
ax2 = fig.add_subplot(1, 3, 3, projection='3d')
setup(ax2, f'シミュレーションのモデル（{"A1v の 1/5" if SCALE == 1.0 else f"A1v の 1/5 × {SCALE:g}"}、カプセル）を真横から', lim)
ax2.view_init(elev=0, azim=0)
base = np.array([0, 0, floor])
sim = [base, base + [0, 0, 219.5 * 0.2 * SCALE]]
dirs = [np.cos(e) * fwd + np.sin(e) * np.array([0, 0, 1.0]) for e in ELEV]
for L, dv in zip((403.1, 330.8, 276.3), dirs):
    sim.append(sim[-1] + L * 0.2 * SCALE * dv)
rad = [0.0865, 0.0501, 0.0466, 0.0535]
u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:9j]
for a, b, r in zip(sim[:-1], sim[1:], rad):
    r = r * 1000 * 0.2
    t = np.linspace(0, 1, 12)
    w = b - a
    e1 = np.cross(w, [1, 0, 0]) if abs(w[0]) < 0.9 * np.linalg.norm(w) else np.cross(w, [0, 1, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(w / np.linalg.norm(w), e1)
    th = np.linspace(0, 2 * np.pi, 18)
    tt, hh = np.meshgrid(t, th)
    X = a[:, None, None] + w[:, None, None] * tt + r * (np.cos(hh) * e1[:, None, None] + np.sin(hh) * e2[:, None, None])
    ax2.plot_surface(X[0], X[1], X[2], color=(0.55, 0.65, 0.8), shade=True, linewidth=0, alpha=0.9)
S = np.array(sim)
ax2.plot(S[:, 0], S[:, 1], S[:, 2], color='tab:red', lw=1.2, ls='--')
ax2.scatter(S[1:-1, 0], S[1:-1, 1], S[1:-1, 2], color='tab:red', s=12)

dist = [np.linalg.norm(joints[i + 1] - joints[i]) for i in range(len(joints) - 2)]
_ax = T[servo_idx[-1]][:3, 2]
_w = joints[-1] - joints[-2]
dist.append(np.linalg.norm(_w - (_w @ _ax) * _ax))   # 手先は軸に垂直な成分（系譜 9-319 と同じ）
tg = [x * 0.2 * SCALE for x in (219.5, 403.1, 330.8, 276.3)]
lab = 'A1v の 1/5' if SCALE == 1.0 else f'A1v の 1/5 × {SCALE:g}'
txt = (f'関節の間（RA で組めた / {lab}）: '
       f'台→肩 {dist[0]:.1f} / {tg[0]:.1f} mm、上腕 {dist[1]:.1f} / {tg[1]:.1f}、前腕 {dist[2]:.1f} / {tg[2]:.1f}、手先 {dist[3]:.1f} / {tg[3]:.1f} mm。'
       '赤の破線は関節を結ぶ線。姿勢は見せるための手置き（学習の結果ではない）。')
fig.text(0.5, 0.03, txt, ha='center', fontsize=9)
fig.suptitle('お辞儀型アーム（根元が旋回＋お辞儀の関節 3 つ）を研究室の部品で組んだときの見た目' + ('' if SCALE == 1.0 else f' — リンクを {SCALE:g} 倍（LEGO を組み立ての基準で挟む）'), fontsize=13)
plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.08, wspace=0.0)
out = os.path.join(HERE, os.environ.get('PNG', 'bow_arm_ra.png'))
fig.savefig(out)
print(out, [round(x, 1) for x in dist])
