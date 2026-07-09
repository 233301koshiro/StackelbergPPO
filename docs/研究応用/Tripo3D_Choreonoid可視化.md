# Tripo3D → Choreonoid 可視化パイプライン

## 概要

Gemini で生成したロボットアーム画像を Tripo3D で 3D モデル化し、URDF + STL として Choreonoid で可視化するまでの手順と知見。

**このドキュメントの位置づけ:**
- **ここ** → デザイン確認・可視化（Gemini → Tripo3D → Choreonoid）
- `メッシュ分割.md` → セグメンテーション手法の比較・理論
- `メッシュXMLパイプライン.md` → MuJoCo/StackelbergPPO への統合（RL 学習用）

---

## パイプライン全体

```
Gemini プロンプト（メッシュ分割.md §Gemini プロンプト集 参照）
  ↓ 画像生成
color-coded ロボット画像（リンク=単色, 関節球=マゼンタ #FF00FF）
  ↓ Tripo3D image-to-3D
GLB ファイル（Y-up 座標系, color-textured）
  ↓ skeleton_extract.py --mode color
topology.json + per-link GLB（mine/robot_segments/）
  ↓ scripts/glb_to_links.py（↓参照。プロトタイプは mine/spatial_split_urdf.py）
URDF + per-link STL（data/tripo_arm_colorful/）
  ↓ Choreonoid で読み込み
3DOF アーム可視化（根本Z回転 + 2関節揺動）
```

---

## 実施済みモデル

| ファイル | 内容 |
|---|---|
| `data/tripo_arm_colorful/mechanical_joystick_3d_model.glb` | Tripo3D 生成 GLB（元データ）|
| `mine/tripo_arm_topology.json` | skeleton_extract.py が出力したトポロジー |
| `mine/tripo_arm_segments/link_{0,1,2}.glb` | 色クラスタ別セグメント（参考） |
| `data/tripo_arm_colorful/tripo_arm.urdf` | Choreonoid 読み込み用 URDF |
| `data/tripo_arm_colorful/meshes/link_{0,1,2}.stl` | リンクローカル座標 STL |

---

## ステップ詳解

### 1. Gemini で画像生成

`docs/研究応用/メッシュ分割.md` の **プロンプトA**（直接生成）または **プロンプトB**（スケッチ整形）を使う。

色の設計規則:
- リンク: 赤 `#FF2222` / 青 `#2222FF` / 緑 `#22CC22` / オレンジ `#FF8800`
- 関節球: **マゼンタ `#FF00FF`**（金色は使わない — アーム本体色と同化するため）

### 2. Tripo3D で GLB 生成

- [tripo3d.ai](https://www.tripo3d.ai) で image-to-3D
- 出力は **Y-up 座標系**（Choreonoid/MuJoCo は Z-up）→ 後段で変換
- 色はほぼ保持されるが色味が変化する場合あり（マゼンタは識別しやすいため誤検出しにくい）

### 3. skeleton_extract.py でセグメント＆トポロジー抽出

```bash
cd /userdir
python3 mine/skeleton_extract.py \
  StackelbergPPO/data/tripo_arm_colorful/mechanical_joystick_3d_model.glb \
  --mode color \
  --n-links 3 \
  --joint-color 255 0 255 \
  --joint-tol 40 \
  --segment-out mine/tripo_arm_segments \
  --out mine/tripo_arm_topology.json
```

出力される `topology.json` の `_position_world`（Y-up 座標）がジョイント位置として後段で使われる。

### 4. URDF + STL 生成（scripts/glb_to_links.py）

色クラスタの K-means によるセグメントは**全高さにまたがる**ため、視覚的に正しくない。
代わりに**ジョイント位置 Z 座標で空間分割**する。

```bash
python3 scripts/glb_to_links.py \
  --glb data/tripo_arm_colorful/mechanical_joystick_3d_model.glb \
  --out-dir data/tripo_arm_colorful/meshes \
  --urdf data/tripo_arm_colorful/tripo_arm.urdf \
  --joints -0.070 0.277   # マゼンタマーカー付き GLB なら省略可（自動検出）
```

スクリプトの処理内容:
1. ジョイント Z 座標を取得（マゼンタマーカー `--joint-color` 自動検出 or `--joints` 手動指定。Y-up → Z-up 変換）
2. 全体 GLB を Z 境界で面ごとに分割 → link_0 / link_1 / link_2
3. 各リンク STL を**リンクローカル座標**にシフト（フレーム原点が各リンクの親ジョイント位置）
4. URDF joint origins を正しく計算して出力

> プロトタイプ `mine/spatial_split_urdf.py`（パスをスクリプト冒頭の定数で指定）を CLI 化したものが `scripts/glb_to_links.py`。以降はリポジトリ版を使うこと。

### 5. Choreonoid で読み込み

```
File > Load > Body → data/tripo_arm_colorful/tripo_arm.urdf
```

Joint Displacement パネルで各関節を動かして確認。

---

## 生成された URDF の構造

```xml
world (固定)
  └─ world_to_base [continuous, Z軸回転, 360度]  ← 根本の向き制御
       └─ link_0 (赤: ベース)
            └─ joint_to_link_1 [revolute, Y軸]
                 └─ link_1 (青: 中間アーム)
                      └─ joint_to_link_2 [revolute, Y軸]
                           └─ link_2 (緑: 先端パドル)
```

| ジョイント | 種類 | 軸 | 可動域 |
|---|---|---|---|
| world_to_base | continuous | Z | 制限なし (360°) |
| joint_to_link_1 | revolute | Y | ±90° |
| joint_to_link_2 | revolute | Y | ±90° |

---

## 知見・落とし穴

### Y-up → Z-up 座標変換

Tripo3D の GLB は Y-up 出力。変換行列:

```python
R_yup2zup = np.array([[1,0,0],[0,0,-1],[0,1,0]])
verts_zup = (R_yup2zup @ verts_yup.T).T
# [x, y, z]_Yup → [x, -z, y]_Zup
```

### URDF の joint origin はフレームベースで計算する

**誤り**: `offset_from_parent = joint_pos - 親リンクの重心`  
**正解**: `offset_from_parent = joint_pos_Zup - 親リンクのフレーム原点_Zup`

根本リンク (link_0) のフレーム原点はアーム底面 (z_min) であり、重心とは異なる。
この誤りにより joint_to_link_1 の Z オフセットが 0.289 になるが、正解は 0.430。

### 各リンクの STL はローカル座標で書き出す

URDF の visual mesh はリンクのローカル座標系で配置される。
グローバル座標のまま STL を書き出すと `joint origin + mesh offset` の二重オフセットが発生してリンクが分離して見える。

正しい手順:
```python
# 各リンクの頂点を「フレーム原点を (0,0,0)」にシフト
verts_local = verts_global_zup - frame_pos_zup[link_name]
```

### ジョイント球の色は マゼンタ (#FF00FF) を使う

金色 (#FFD700) を指定すると Tripo3D がアーム本体の表面色と同化させる場合がある（今回: 全面の 19% が「金色」と検出されたが実際はアーム胴体）。マゼンタは自然物には現れないため誤検出しにくい。

### Tripo3D の色保持について

Tripo3D は色をほぼ保持するが、完全一致しない（2026-07-09 マゼンタ版 GLB で実測）:
- 赤 `#FF2222` → `[216, 61, 54]` 程度に変化
- 青 `#2222FF` → `[63, 46, 184]` 程度
- マゼンタ `#FF00FF` → **`[211, 75, 169]` 程度に暗色化・低彩度化**。ただし本体色とは明確に分離した独立クラスタとして残る（金色のような同化は起きない）。純 #FF00FF からの距離 ≈122 のため **tol=40 では拾えない** → `--joint-color 211 75 169 --joint-tol 60` を使うこと

色確認コマンド:
```python
import trimesh
from scipy.cluster.vq import kmeans2
scene = trimesh.load("model.glb")
geom = list(scene.geometry.values())[0]
vc = geom.visual.to_color().vertex_colors[:, :3].astype(float)
fc = vc[geom.faces].mean(axis=1)
centroids, labels = kmeans2(fc, k=5, seed=0, minit='points')
print(centroids.astype(int))
```

---

## 今後の TODO

- [x] `mine/spatial_split_urdf.py`（プロトタイプ）を `scripts/glb_to_links.py` として CLI 化・リポジトリに恒久化
- [x] マゼンタで再度 Tripo3D に入稿し、ジョイント自動検出が通るか検証 → **✅ 成功（2026-07-09）**。3関節すべて自動検出（`data/tripo_arm_colorful2/`）。ただし Tripo3D はマゼンタを ≈RGB[211,75,169] に暗色化するため `JOINT_COLOR="211 75 169" JOINT_TOL=60` の指定が必要（詳細: メッシュXMLパイプライン.md §8-2）
- [x] MuJoCo XML への変換 → RL 学習まで実証済み（2026-07-08、`メッシュXMLパイプライン.md` §8 参照）
