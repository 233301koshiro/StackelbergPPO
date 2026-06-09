# メッシュ分割手法の比較と StackelbergPPO への統合方針

## 概要

Tripo3D 等で生成した単一メッシュをロボットの「パーツ」に分割する方法の比較。
分割粒度（パーツ数）の制御しやすさと、StackelbergPPO への接続しやすさを主軸に評価する。

---

## 1. ヒューリスティック・幾何学的アプローチ

プログラム的なルールで分割するため、**計算が軽く、過分割の制御が容易**。

### 1-a. スケルトン抽出ベース（Reeb Graph / Medial Axis）

メッシュの内部に「骨格（カーブスケルトン）」を抽出し、骨格の分岐点（ノード）を境界として分割する。

```
メッシュ
  ↓ Medial Axis Transform / Reeb Graph
骨格グラフ（ノード = 分岐点、エッジ = リンク）
  ↓ 枝刈り（Pruning）で細かすぎる枝を削除
ノード数 = パーツ数 に対応した分割
```

**メリット:**
- 骨格のエッジが直接「リンク」、ノードが「関節」に対応 → キネマティックチェーンと直結
- 枝刈りの閾値 1 つでロボットの関節数を直接コントロールできる
- 抽出した骨格がそのままジョイント軸の初期推定にも使える

**デメリット:**
- 球体・箱形など骨格が不定形なメッシュでは骨格が不安定になる
- 実装ライブラリが少ない（`scikit-tda`, `vedo` の `skeletonize()` など）

**ツール例:**
```python
import vedo
mesh = vedo.load("robot.glb")
skel = mesh.tomesh().skeletonize(tol=0.01)  # tol で枝刈り閾値を調整
```

---

### 1-b. 曲率・凹面ベースの分割（Concavity-based）

メッシュ表面の「凹んでいる部分（谷）」を関節の継ぎ目とみなし、そこを境界として分割する。

```
メッシュ
  ↓ 各頂点の mean curvature / Gaussian curvature を計算
  ↓ 曲率が閾値を超える「谷」領域を検出
  ↓ Fast Marching / Geodesic Distance で領域を成長
パーツ境界 = くびれ部分
```

**メリット:**
- 物理演算で干渉しやすい「くびれ」で自然に分割される → 爆発問題回避に直結
- 曲率の閾値だけで分割の粗さを制御できる
- `trimesh` で曲率を計算できる

**デメリット:**
- 曲率が均一なメッシュ（円柱状の胴体など）では境界が曖昧になる
- 表面ノイズに弱い（スムージング前処理が必要）

**ツール例:**
```python
import trimesh
import numpy as np

mesh = trimesh.load("robot.glb")
# 頂点曲率の近似: 各頂点の法線変化量
# trimesh には直接の曲率 API はないので法線変動を代替指標として使う
vertex_normals = mesh.vertex_normals
adjacency = mesh.vertex_adjacency_graph
curvature = np.zeros(len(mesh.vertices))
for v, neighbors in adjacency.items():
    neighbor_normals = vertex_normals[list(neighbors)]
    curvature[v] = np.mean(1.0 - neighbor_normals @ vertex_normals[v])

# 高曲率領域（閾値以上）= 分割候補境界
threshold = 0.3
boundary_vertices = np.where(curvature > threshold)[0]
```

**より本格的な実装:** [VHACD（Approximate Convex Decomposition）](https://github.com/mikedh/trimesh/blob/main/trimesh/decomposition.py)を使うと、凹面分解として直接パーツに切り出せる。

```python
# trimesh の VHACD ラッパー
parts = trimesh.decomposition.convex_decomposition(mesh, maxhulls=8)
# maxhulls でパーツ数の上限を指定
```

---

## 2. ディープラーニング・データドリブンアプローチ

セマンティクス（意味）を理解して分割するため、**スケッチの意図を汲み取りやすい**。

### 2-a. 特徴量クラスタリング（PointNet++ + K-Means）

メッシュを点群に変換し、PointNet++ で局所特徴量を抽出した後、K-Means でクラスタリングする。

```
メッシュ → 点群サンプリング（FPS: Farthest Point Sampling）
  ↓ PointNet++ エンコーダ
各点の局所特徴量ベクトル（128〜512次元）
  ↓ K-Means（K = 目標パーツ数）
K 個のセグメント
```

**メリット:**
- `K`（分割数）を直接指定できる → 「大まかに 5 パーツ」のようなユーザーフレンドリーな指定が可能
- 形状の幾何学的特徴（面の向き・曲率・局所構造）を自動的に学習した特徴量で分割されるため、意味的にまとまりやすい
- 事前学習済みモデルを使えばファインチューニング不要

**デメリット:**
- PointNet++ の推論が必要（GPU or 数秒の CPU 推論）
- クラスタリング結果が毎回異なる（初期値依存）

**ツール例:**
```python
# torch-points3d や open3d の PointNet++ を使う場合
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

mesh = o3d.io.read_triangle_mesh("robot.glb")
pcd = mesh.sample_points_uniformly(number_of_points=4096)
points = np.asarray(pcd.points)

# PointNet++ の特徴量抽出（事前学習済みモデルを利用）
# ここでは簡略化として座標 + 法線を特徴量として代替
pcd.estimate_normals()
normals = np.asarray(pcd.normals)
features = np.hstack([points, normals])  # 6次元

K = 6  # 目標パーツ数
kmeans = KMeans(n_clusters=K, random_state=0).fit(features)
labels = kmeans.labels_
```

---

### 2-b. 3D-MLLM / VLM を用いた言語主導の分割（Open-vocabulary Segmentation）

画像または 3D モデルに対してテキストプロンプトを与え、ゼロショットで分割する。

```
メッシュ → マルチビューレンダリング（6〜12 方向からの 2D 画像）
  ↓ SAM2（Segment Anything Model 2）+ CLIP / DINO
「左腕」「右脚」「胴体」などのテキストプロンプト
  ↓ 各 2D マスクをメッシュ頂点に逆投影（バックプロジェクション）
3D セグメント
```

**メリット:**
- スケッチを描いた人が「ここは腕、ここは車輪」と指定した意図をそのままプロンプトとして流し込める
- 意味的なまとまりで剛体を切り出せる（胴体・脚など形状によらずセマンティクスで分割）
- ゼロショットのため追加学習不要

**デメリット:**
- マルチビューから 3D への逆投影で誤差が生じる（特に陰影になる部分）
- パーツ境界が 2D の視点依存になりやすい
- 推論コストが高い（SAM2 + CLIP の並列実行）

**ツール例（概念）:**
```python
# PartSLIP2 / PartDistill などを利用するアプローチ
# または SAM2 + テキストプロンプトで 2D セグメンテーション → 3D 逆投影
from segment_anything import SamPredictor
# ... マルチビューレンダリング → SAM2 でセグメント → 点群への逆投影
```

**参考実装:** [PartSLIP2](https://github.com/zyc00/PartSLIP2)（テキストプロンプトによる 3D パーツセグメンテーション）

---

## 3. 手法の比較

| 手法 | パーツ数の制御 | 意図の反映 | 計算コスト | StackelbergPPO との相性 |
|------|--------------|-----------|-----------|------------------------|
| スケルトン抽出 | 枝刈り閾値で○ | △（形状依存） | 軽い | ◎（骨格=キネマティックチェーン）|
| 凹面ベース（VHACD）| maxhulls で○ | △（物理的に自然） | 軽〜中 | ○（くびれ=関節で綺麗に切れる）|
| PointNet++ + K-Means | K を直指定 ◎ | △（意味は弱い） | 中 | ○（パーツ数指定が直感的）|
| 3D-MLLM / VLM | プロンプトで○ | ◎（意図を直接反映） | 重い | ○（意味的なパーツ定義）|

**推奨構成:** スケッチからのパイプラインではまず **スケルトン抽出**（骨格=関節の自動対応）を試し、意味的な調整が必要な場合に **VLM プロンプト**で補完する二段構えが現実的。

---

## 4. StackelbergPPO への統合方針

### 分割結果を初期状態 $s_0^L$ として設定する

分割済みメッシュを `topology.json`（[mesh_to_xml_pipeline.md](mesh_to_xml_pipeline.md) 参照）に変換した上で、Leader の初期状態を固定する。

```
分割済みメッシュ × N パーツ
  ↓ FK 解決・カプセルフィット（mesh_to_xml_pipeline.md）
topology.json + assets/mujoco_envs/my_robot.xml
  ↓
StackelbergPPO（fix_skeleton=true）
```

### Leader のアクション空間を制限する

| 制限事項 | 設定方法 | 意図 |
|---------|---------|------|
| トポロジーを固定（腕の追加・削除を禁止） | `fix_skeleton=true` or `max_nchild=0` | スケッチの意図を壊さない |
| リンク長の微調整幅を制限（±20%程度） | `body_params.offset` の `lb/ub` を現在値近傍に設定 | 「スケールの小さな補正」のみ許可 |
| モーターギア比の探索範囲を絞る | `actuator_params.gear` の `lb/ub` を実機仕様近傍に設定 | 物理的に非現実的なトルクを防ぐ |
| 関節可動域を解剖学的制限内に収める | `joint_params.range` の `lb/ub` を人体・動物の可動域に設定 | 自然な動作に制限 |

**設定例（スケッチ由来の 4 脚ロボット用 yml）:**

```yaml
# design_opt/cfg/my_robot.yml
env_name: ant  # または pusher
fix_skeleton: true

robot:
  param_mapping: sin
  no_root_offset: true

  body_params:
    offset:
      type: 'xy'
      lb: [-0.6, -0.6]   # デフォルト ±0.5 より少し広げる
      ub: [0.6, 0.6]

  geom_params:
    size:
      lb: 0.03
      ub: 0.12            # スケッチのリンク太さに合わせて調整

  actuator_params:
    gear:
      lb: 50              # 実機の最小トルク想定
      ub: 300             # 実機の最大トルク想定

  joint_params:
    range:
      lb: -1.04           # -60度（ラジアン）
      ub:  1.04           # +60度
```

### 分割手法ごとの統合特性

| 分割手法 | topology.json 生成の容易さ | ジョイント位置の自動推定 |
|---------|--------------------------|------------------------|
| スケルトン抽出 | ◎（骨格ノード → ジョイント座標が直接得られる）| ◎ |
| 凹面ベース | ○（パーツ境界の重心 → ジョイント候補）| △ |
| PointNet++ + K-Means | △（クラスタ境界からジョイント位置を推定）| △ |
| VLM プロンプト | ○（セマンティクスからジョイント名を推定可）| △ |

**スケルトン抽出が最も StackelbergPPO との相性が良い理由:**
骨格抽出の結果がそのまま `topology.json` の `offset_from_parent` に使えるため、パイプライン全体の人手介入が最小になる。

---

## 5. 実装ロードマップ

```
Phase 1: 手動 topology.json で動作確認
  └─ 手書きで topology.json を書き、パイプラインの後半（FK → XML → 学習）を先に完成させる

Phase 2: VHACD による自動凹面分割
  └─ trimesh.decomposition.convex_decomposition() で maxhulls を指定した半自動分割を実装
  └─ パーツ境界の重心からジョイント候補を自動推定

Phase 3: スケルトン抽出の自動化
  └─ vedo の skeletonize() で骨格を抽出
  └─ 骨格ノードを topology.json の offset_from_parent に自動変換

Phase 4: VLM プロンプト統合（オプション）
  └─ スケッチのテキスト説明を PartSLIP2 等に渡して意味的パーツ名を付与
  └─ topology.json の name フィールドに自動入力（可読性向上）
```
