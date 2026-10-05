"""⭐ Choreonoid の画面で、腕を**元のカラフルなメッシュ**で見せる（見た目専用。2026-10-05、デモ用）。

⚠️ **物理は変えない。**`.body` の腕のリンクで、カプセルを当たり判定専用（Collision）にし、
見た目（Visual）だけ元のメッシュに差し替える。質量・慣性は `.body` に数値で書いてあるので変わらない。
⭐ **環境変数 `CNOID_VISUAL_MESHES` を指定したときだけ**働く（学習では指定しない）。

| 環境変数 | 意味 |
|---|---|
| `CNOID_VISUAL_MESHES` | リンクごとの STL と `joints.json` のあるディレクトリ（例 `data/test/A1/meshes`） |
| `CNOID_VISUAL_GLB` | 元の GLB（色を拾う。無ければ灰色） |
| `CNOID_VISUAL_NAMES` | 根元から順のメッシュ名（既定 `base,upper_arm,forearm,hand`）。腕の body の先頭（取り付け用の球）は飛ばす |

変換は `scripts/render_arm_video.py`（mp4 のデモ）と同じ:
メッシュ座標のボーン方向を +Z へ回し、**いまのボーン長 ÷ 元の長さ**で Z だけ伸縮し、リンクのボーン方向へ回す。
⭐ メッシュは**頂点色付きの COLLADA（.dae）**で書く（Choreonoid の Assimp プラグインが読める拡張子は
dae / blend / x / dxf だけで、頂点色を `setColors` で取り込む）。
⚠️ ファイル上は `Y_UP` と宣言して**回転させずに**通す（Assimp は上向きの軸に応じて根の変換を掛け、Choreonoid はそれをそのまま使う）。
"""
import json
import os
import pathlib
import tempfile

import numpy as np

_CACHE = {}


def _align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def _decimate(V, F, C, pitch):
    key = np.floor(V / pitch).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.zeros(len(uniq)); np.add.at(cnt, inv, 1)
    nV = np.zeros((len(uniq), 3)); np.add.at(nV, inv, V); nV /= cnt[:, None]
    nC = np.zeros((len(uniq), 3)); np.add.at(nC, inv, C); nC /= cnt[:, None]
    nF = inv[F]
    ok = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    return nV, nF[ok], nC


def _spec():
    """環境変数からメッシュを読み、リンク名 → (頂点, 面, 頂点色, メッシュ座標のボーン) を返す（1 回だけ）。"""
    mdir = os.environ.get('CNOID_VISUAL_MESHES')
    if not mdir:
        return None
    if 'spec' in _CACHE:
        return _CACHE['spec']
    import trimesh
    mdir = pathlib.Path(mdir)
    names = os.environ.get('CNOID_VISUAL_NAMES', 'base,upper_arm,forearm,hand').split(',')
    origins = json.loads((mdir / 'joints.json').read_text())['frame_origins']
    glb = os.environ.get('CNOID_VISUAL_GLB')
    gV = gC = None
    if glb:
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        sc = trimesh.load(glb)
        g = sc.to_geometry() if hasattr(sc, 'to_geometry') else sc
        gC = np.asarray(g.visual.to_color().vertex_colors[:, :3], float) / 255.0
        V = np.asarray(g.vertices)
        gV = np.c_[V[:, 0], -V[:, 2], V[:, 1]]          # glb_to_links.py と同じ Y-up → Z-up
    from scipy.spatial import cKDTree
    out = {}
    for j, mn in enumerate(names):
        m = trimesh.load(mdir / f'{mn}.stl')
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
        C = np.full((len(V), 3), 0.6)
        if gV is not None and mn in origins:
            d, idx = cKDTree(gV - np.asarray(origins[mn])).query(V)
            if np.median(d) < 1e-6:
                C = gC[idx]
            else:
                print(f'[visual_mesh] ⚠️ {mn}: GLB と一致しない（距離の中央値 {np.median(d):.2e}）。灰色で描く', flush=True)
        V, F, C = _decimate(V, F, C, float(os.environ.get('CNOID_VISUAL_PITCH', '0.008')))
        if j + 1 < len(names) and names[j + 1] in origins:
            bl = np.asarray(origins[names[j + 1]]) - np.asarray(origins[mn])
        else:
            bl = V[np.argmax(np.linalg.norm(V, axis=1))]
        out[mn] = (V, F, C, bl)
    _CACHE['spec'] = (names, out)
    print(f'[visual_mesh] ⭐ 見た目を元のメッシュに差し替える: {names}（色 {"GLB" if gV is not None else "灰色"}）', flush=True)
    return _CACHE['spec']


def _write_dae(path, V, F, C):
    pos = ' '.join(f'{x:.6g}' for x in V.reshape(-1))
    col = ' '.join(f'{x:.4g}' for x in C.reshape(-1))
    idx = ' '.join(str(int(i)) for i in F.reshape(-1))
    n, m = len(V), len(F)
    path.write_text(f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Y_UP</up_axis></asset>
<library_geometries><geometry id="g" name="g"><mesh>
<source id="g-pos"><float_array id="g-pos-a" count="{3*n}">{pos}</float_array>
<technique_common><accessor source="#g-pos-a" count="{n}" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="g-col"><float_array id="g-col-a" count="{3*n}">{col}</float_array>
<technique_common><accessor source="#g-col-a" count="{n}" stride="3"><param name="R" type="float"/><param name="G" type="float"/><param name="B" type="float"/></accessor></technique_common></source>
<vertices id="g-vtx"><input semantic="POSITION" source="#g-pos"/></vertices>
<triangles count="{m}"><input semantic="VERTEX" source="#g-vtx" offset="0"/><input semantic="COLOR" source="#g-col" offset="0" set="0"/><p>{idx}</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
''')


def visual_dae_for(link_index, bone):
    """腕の body の `link_index` 番目（0 = 取り付け用の球）に付ける .dae のパス。無効なら None。

    `bone` は**いまの**ボーン（リンク座標）。長さと向きに合わせてメッシュを伸縮・回転する。
    """
    sp = _spec()
    if sp is None:
        return None
    names, meshes = sp
    j = link_index - 1                      # 先頭の取り付け用の球はメッシュを持たない
    if j < 0 or j >= len(names):
        return None
    V, F, C, bl = meshes[names[j]]
    bone = np.asarray(bone, float)
    Lb, Lo = float(np.linalg.norm(bl)), float(np.linalg.norm(bone))
    if Lb < 1e-9 or Lo < 1e-9:
        return None
    key = (names[j], tuple(np.round(bone, 5)))
    if key in _CACHE:
        return _CACHE[key]
    R1 = _align(bl, np.array([0., 0., 1.]))
    S = np.diag([1., 1., Lo / Lb])
    R2 = _align(np.array([0., 0., 1.]), bone)
    W = (R2 @ S @ R1 @ V.T).T
    d = pathlib.Path(tempfile.gettempdir()) / 'cnoid_visual_mesh'
    d.mkdir(exist_ok=True)
    path = d / f'{names[j]}_{abs(hash(key)) % 10**10}.dae'
    _write_dae(path, W, F, C)
    _CACHE[key] = str(path)
    return str(path)
