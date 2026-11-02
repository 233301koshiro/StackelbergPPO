"""軸6: 部品で組んだ腕（build_arm3.py の部品の json）を、Choreonoid の画面のデモ（demo_cnoid.sh）の見た目に使える形へ書き出す。

    python3 scripts/make_ra_visual_meshes.py <部品の json> <出力のフォルダ>

出力は visual_mesh.py（CNOID_VISUAL_MESHES / CNOID_VISUAL_GLB）が読む形: base / upper_arm / forearm / hand の STL（m、各リンクの原点から）・
joints.json（frame_origins）・色の arm.glb。⚠️ 見た目だけで、物理はカプセルのまま（visual_mesh.py の説明）。
並べ方: 部品の腕をまっすぐ上へ伸ばした姿勢にし（render_ra_forms.py の pose_parts、仰角 3 つとも 90°）、関節 2〜4 の軸を +y、
腕の前を +x へ向ける。シミュレーションの腕も伸ばした姿勢で骨が +z を向き、関節 2〜4 の軸は y（e2e_a1v_ra15.xml）なので、
visual_mesh.py の回転（メッシュの骨 → +z → いまの骨）が軸まわりにねじれない。
リンクの割り当て: サーボ k の後ろの部品はそのサーボの関節で動くので、body の k 番目（1 = 台、関節 1 の後ろ）は
サーボ k の次からサーボ k+1 まで。関節 1 のサーボは動かない側だが、visual_mesh は取り付けの球を描かないので台に入れる（旋回で一緒に回って見える）。
リンクの原点: 関節 2〜4 のサーボの原点は軸の方向へずれている（19 mm）。軸の上を関節 1 の軸の面まで動かす（骨が横へ傾かないように）。
"""
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_ra_forms import mesh, pose_parts   # noqa: E402

NAMES = ['base', 'upper_arm', 'forearm', 'hand']


def main():
    src, out = sys.argv[1], sys.argv[2]
    parts = json.load(open(src))['parts']
    servo = [i for i, p in enumerate(parts) if p['servo']]
    assert len(servo) == 4, servo
    T, fwd, J, got = pose_parts(parts, [np.pi / 2] * 3)
    print(f'まっすぐにした仰角（期待 90°）: {np.round(np.degrees(got), 3)}')
    ax = T[servo[1]][:3, 2] / np.linalg.norm(T[servo[1]][:3, 2])
    up = np.array([0, 0, 1.0])
    y = np.cross(up, fwd)
    R = np.vstack([fwd, y, up])                    # 部品の座標 → 前 x・軸 y・上 z
    print(f'関節 2 の軸 → {np.round(R @ ax, 4)}（期待 ±y）')
    o1 = T[servo[0]][:3, 3]
    to = lambda q: R @ (q - o1) / 1000.0           # mm → m、関節 1 のサーボの原点が 0
    org = []
    for k in range(4):
        q = to(T[servo[k]][:3, 3])
        if k > 0:
            q[1] = 0.0                             # 軸（y）の上を関節 1 の軸の面へ
        org.append(q)
    bounds = servo[1:] + [len(parts)]
    os.makedirs(out, exist_ok=True)
    allV, allF, allC, n = [], [], [], 0
    for k, nm in enumerate(NAMES):
        lo = 0 if k == 0 else servo[k] + 1
        ms, cs = [], []
        for i in range(lo, bounds[k] + (1 if k < 3 else 0)):
            for v in parts[i]['visual']:
                m = mesh(v, T[i])
                m.vertices = np.array([to(p) for p in m.vertices])
                ms.append(m)
                cs.append(np.tile((v.get('color') or [0.6, 0.6, 0.6])[:3], (len(m.vertices), 1)))
        m = trimesh.util.concatenate(ms)
        C = np.vstack(cs)
        trimesh.Trimesh(m.vertices - org[k], m.faces, process=False).export(f'{out}/{nm}.stl')
        allV.append(np.asarray(m.vertices)); allF.append(np.asarray(m.faces) + n); allC.append(C); n += len(m.vertices)
        bone = (org[k + 1] - org[k]) if k < 3 else None
        print(f'  {nm:9}: 部品 {sum(len(parts[i]["visual"]) for i in range(lo, bounds[k] + (1 if k < 3 else 0)))} 個・'
              f'面 {len(m.faces)}・' + (f'骨 {np.round(bone * 1000, 1)} mm（長さ {np.linalg.norm(bone) * 1000:.1f}）'
                                       if bone is not None else f'先端の最遠点 {np.round((m.vertices - org[k])[np.argmax(np.linalg.norm(m.vertices - org[k], axis=1))] * 1000, 1)} mm'))
    tip = to(J[-1])                                # 軸から最も遠い手先の点（pose_parts）。最遠点は手先の角で軸の方向へずれている
    tip[1] = 0.0
    print(f'  先端（軸の面へ移した）: {np.round((tip - org[3]) * 1000, 1)} mm（長さ {np.linalg.norm(tip - org[3]) * 1000:.1f}）')
    fo = {nm: org[k].tolist() for k, nm in enumerate(NAMES)}
    fo['tip'] = tip.tolist()                       # visual_mesh.py・render_arm_video.py は先端のリンクの骨にこれを使う
    json.dump({'frame_origins': fo}, open(f'{out}/joints.json', 'w'), indent=1)
    V = np.vstack(allV)
    g = trimesh.Trimesh(np.c_[V[:, 0], V[:, 2], -V[:, 1]], np.vstack(allF), process=False,
                        vertex_colors=(np.clip(np.vstack(allC), 0, 1) * 255).astype(np.uint8))
    g.export(f'{out}/arm.glb')                     # visual_mesh.py は GLB を Y-up として (x, -z, y) へ戻す
    print(f'✅ {out}: {", ".join(NAMES)} の STL・joints.json・arm.glb')


if __name__ == '__main__':
    main()
