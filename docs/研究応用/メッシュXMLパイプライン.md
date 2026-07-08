# メッシュ → MuJoCo XML 変換パイプライン設計方針

## 概要

Tripo3D 等で生成した非定型メッシュをそのまま RL 環境に渡すと、シミュレーション開始直後にロボットが吹き飛ぶ「爆発問題」がほぼ確実に発生する。本ドキュメントはこの問題を防ぐためのデータパイプライン設計方針をまとめる。

---

## 1. 爆発問題の原因と分類

| 原因 | 具体的な症状 | 深刻度 |
|------|-------------|--------|
| **初期干渉** | 隣接パーツのメッシュが重なった状態でシミュレーション開始 → 接触力が爆発的に発生 | ★★★ |
| **スケール不一致** | メッシュが mm 単位（MuJoCo は m）→ 重力・慣性が 10^6 倍ズレ | ★★★ |
| **ジョイント位置のズレ** | 関節錨点がパーツ内部に食い込んでいる → 拘束力が発散 | ★★★ |
| **不自然な重心** | COM がボディ外に出ている / 極端に偏る → 即転倒 | ★★ |
| **慣性テンソルの異常** | ゼロ近傍の慣性値 → 数値不安定 | ★★ |
| **関節軸の不整合** | 関節軸がパーツの実際の曲がり方向と垂直 → 制御不能 | ★ |

---

## 2. パイプライン全体像

```
[入力]
  mesh_parts/
    torso.glb
    FL_upper.glb
    FL_lower.glb
    ...
  topology.json          ← ユーザーが定義する接続関係・ジョイント位置

      ↓ Stage 1: 正規化
  スケール統一（→ m）・座標系統一（Z-up）

      ↓ Stage 2: FK による絶対座標解決
  T-pose で全ボディのワールド座標を計算（干渉しない姿勢に整列）

      ↓ Stage 3: 衝突形状の近似
  capsule / box フィッティング（各メッシュ → シンプルな衝突モデル）

      ↓ Stage 4: 検証
  干渉チェック・COM チェック・慣性チェック

      ↓ Stage 5: XML 生成
  MuJoCo XML（StackelbergPPO 命名規則に準拠）

[出力]
  assets/mujoco_envs/my_robot.xml
```

---

## 3. 各ステージの設計方針

### Stage 1: 正規化

**スケール検出と統一:**

```python
import trimesh
import numpy as np

def normalize_mesh(mesh: trimesh.Trimesh, target_unit='m') -> trimesh.Trimesh:
    extent = mesh.bounding_box.extents.max()
    # 経験則: 最大寸法が 0.01〜10m 範囲を超えていたらスケール補正
    if extent > 10.0:
        # おそらく mm 単位
        mesh.apply_scale(0.001)
    elif extent < 0.01:
        # おそらく cm 単位
        mesh.apply_scale(0.01)
    return mesh
```

**座標系統一（Y-up → Z-up）:**

Tripo3D の出力は Y-up の場合がある。MuJoCo は Z-up。

```python
def to_zup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Y-up → Z-up: X軸回りに -90度回転
    R = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    mesh.apply_transform(R)
    return mesh
```

---

### Stage 2: FK による絶対座標解決（干渉防止の核心）

**topology.json のフォーマット:**

```json
{
  "bodies": [
    {
      "name": "torso",
      "mujoco_id": "0",
      "mesh": "torso.glb",
      "parent": null,
      "joint": null
    },
    {
      "name": "FL_upper",
      "mujoco_id": "1",
      "mesh": "FL_upper.glb",
      "parent": "torso",
      "joint": {
        "type": "hinge",
        "axis": [0, 1, 0],
        "range": [-60, 60],
        "offset_from_parent": [0.2, 0.15, -0.1]
      }
    },
    {
      "name": "FL_lower",
      "mujoco_id": "11",
      "mesh": "FL_lower.glb",
      "parent": "FL_upper",
      "joint": {
        "type": "hinge",
        "axis": [0, 1, 0],
        "range": [-90, 0],
        "offset_from_parent": [0.0, 0.0, -0.25]
      }
    }
  ]
}
```

`offset_from_parent` = 親ボディのジョイント座標系における、次のジョイントまでのオフセット。

**FK 計算:**

```python
def solve_fk(bodies: list[dict]) -> dict[str, np.ndarray]:
    """各ボディのワールド座標 (4x4 変換行列) を返す"""
    world_transforms = {}
    body_map = {b['name']: b for b in bodies}

    def recurse(body_name, parent_T):
        body = body_map[body_name]
        if body['joint'] is None:
            T = parent_T.copy()
        else:
            offset = np.array(body['joint']['offset_from_parent'])
            T = parent_T.copy()
            T[:3, 3] += parent_T[:3, :3] @ offset  # 親座標系でのオフセットをワールドへ
        world_transforms[body_name] = T
        for child in bodies:
            if child['parent'] == body_name:
                recurse(child['name'], T)

    # root の初期変換: Z軸方向に適切な高さを与えてグラウンドから浮かせる
    root = next(b for b in bodies if b['parent'] is None)
    root_height = estimate_ground_clearance(root)  # 後述
    root_T = np.eye(4)
    root_T[2, 3] = root_height
    recurse(root['name'], root_T)
    return world_transforms
```

**地面クリアランスの自動推定:**

```python
def estimate_ground_clearance(root_body: dict) -> float:
    """ロボット全体の最下点が地面 (z=0) より少し上になる高さを返す"""
    mesh = trimesh.load(root_body['mesh'])
    lowest_z = mesh.bounds[0][2]   # メッシュの最小 Z
    return -lowest_z + 0.02        # 2cm のマージンを加える
```

FK を通すことで「T-pose（全関節 0 度）での各ボディのワールド座標」が確定し、**パーツ同士が空間上で重ならない** 初期配置が保証される。

---

### Stage 3: 衝突形状の近似

MuJoCo は capsule / sphere / box / cylinder を使う。StackelbergPPO は **capsule と sphere のみ** 対応（xml_robot.py の制約）。

**カプセルフィッティング（主軸ベース）:**

```python
def fit_capsule(mesh: trimesh.Trimesh) -> dict:
    """メッシュの主軸方向にカプセルをフィッティングする"""
    hull = mesh.convex_hull
    obb = hull.bounding_box_oriented

    extents = obb.primitive.extents          # [lx, ly, lz]
    transform = obb.primitive.transform      # OBB の変換行列

    # 最長軸 = カプセルの軸方向
    longest_axis_idx = np.argmax(extents)
    length = extents[longest_axis_idx]
    radius = np.mean(np.delete(extents, longest_axis_idx)) / 2.0

    # カプセルの両端点（ローカル座標）
    axis_dir = transform[:3, longest_axis_idx]
    center = transform[:3, 3]
    start = center - axis_dir * (length / 2)
    end   = center + axis_dir * (length / 2)

    return {
        'radius': float(np.clip(radius, 0.03, 0.10)),  # StackelbergPPO の範囲にクリップ
        'start': start,
        'end': end,
        'length': float(length),
    }
```

**フィッティング品質の確認:**

```python
def capsule_coverage_ratio(mesh, capsule) -> float:
    """カプセルに収まるメッシュ頂点の割合（0〜1）"""
    pts = mesh.vertices
    axis = capsule['end'] - capsule['start']
    axis_len = np.linalg.norm(axis)
    axis_unit = axis / axis_len
    t = np.dot(pts - capsule['start'], axis_unit)
    t_clamped = np.clip(t, 0, axis_len)
    closest = capsule['start'] + t_clamped[:, None] * axis_unit
    dist = np.linalg.norm(pts - closest, axis=1)
    return float(np.mean(dist <= capsule['radius']))
    # 0.8 以上なら良好; 0.6 未満なら別の形状を検討
```

---

### Stage 4: 検証チェック

**チェック 1: カプセル同士の干渉検出**

```python
def capsule_distance(c1, c2) -> float:
    """2つのカプセルの最近接距離（負なら干渉）"""
    # セグメント間最近接距離 - (r1 + r2)
    d = segment_segment_distance(c1['start'], c1['end'], c2['start'], c2['end'])
    return d - c1['radius'] - c2['radius']

def check_no_penetration(bodies, capsules, topology) -> list[str]:
    errors = []
    for b in topology:
        if b['parent'] is None:
            continue
        d = capsule_distance(capsules[b['name']], capsules[b['parent']])
        if d < -0.005:   # 5mm 以上の干渉は警告
            errors.append(f"干渉: {b['name']} ↔ {b['parent']}  overlap={-d:.3f}m")
    return errors
```

**チェック 2: 重心の妥当性**

```python
def check_com(mesh: trimesh.Trimesh, capsule: dict) -> bool:
    """重心がカプセル内に収まっているか"""
    com = mesh.center_mass
    axis = capsule['end'] - capsule['start']
    t = np.dot(com - capsule['start'], axis / np.linalg.norm(axis))
    closest = capsule['start'] + t * axis / np.linalg.norm(axis)
    dist = np.linalg.norm(com - closest)
    return dist <= capsule['radius'] * 1.5   # 多少の余裕を持つ
```

**チェック 3: 慣性テンソルの最小値保証**

```python
def clamp_inertia(inertia_tensor: np.ndarray, min_val=1e-4) -> np.ndarray:
    """対角成分をゼロ以上に保つ（MuJoCo が inertiafromgeom=true なら不要）"""
    diag = np.diag(inertia_tensor)
    diag = np.maximum(diag, min_val)
    return np.diag(diag)
```

MuJoCo の `<compiler inertiafromgeom="true"/>` を使えば、慣性は自動計算されるためこのチェックは不要になる。**基本的には `inertiafromgeom=true` を使うべき**。

---

### Stage 5: XML 生成

**StackelbergPPO の命名規則への変換:**

```python
def assign_mujoco_ids(bodies: list[dict]) -> dict[str, str]:
    """topology.json の name を StackelbergPPO の命名規則 (例: "11") に変換"""
    id_map = {}
    body_map = {b['name']: b for b in bodies}
    root = next(b for b in bodies if b['parent'] is None)

    def recurse(body_name, parent_id, sibling_idx):
        if parent_id is None:
            mid = "0"
        else:
            pname = "" if parent_id == "0" else parent_id
            mid = str(sibling_idx + 1) + pname
        id_map[body_name] = mid
        children = [b for b in bodies if b['parent'] == body_name]
        for i, child in enumerate(children):
            recurse(child['name'], mid, i)

    recurse(root['name'], None, 0)
    return id_map
```

**XML 出力:**

```python
from lxml.etree import Element, SubElement, ElementTree, tostring

def generate_mujoco_xml(bodies, world_transforms, capsules, id_map) -> str:
    root_elem = Element('mujoco', model='custom_robot')

    # compiler: inertiafromgeom=true で慣性を自動計算
    SubElement(root_elem, 'compiler',
               angle='degree', coordinate='global', inertiafromgeom='true')
    SubElement(root_elem, 'option', integrator='RK4', timestep='0.01')

    worldbody = SubElement(root_elem, 'worldbody')
    # floor
    SubElement(worldbody, 'geom', name='floor', type='plane',
               pos='0 0 0', size='200 200 .125', conaffinity='1', condim='3')

    actuator_elem = SubElement(root_elem, 'actuator')
    body_map = {b['name']: b for b in bodies}

    def add_body(body_name, parent_xml_elem):
        body = body_map[body_name]
        mid = id_map[body_name]
        T = world_transforms[body_name]
        pos = T[:3, 3]
        cap = capsules[body_name]

        body_elem = SubElement(parent_xml_elem, 'body',
                               name=mid,
                               pos=' '.join(f'{x:.6f}' for x in pos))

        if body['joint'] is None:
            # root: free joint
            SubElement(body_elem, 'joint', name=f'{mid}_joint',
                       type='free', armature='0', damping='0',
                       limited='false', pos=' '.join(f'{x:.6f}' for x in pos))
        else:
            j = body['joint']
            joint_pos = pos  # FK 解決済み
            axis_str = ' '.join(f'{x:.6f}' for x in j['axis'])
            rng = j.get('range', [-60, 60])
            SubElement(body_elem, 'joint',
                       name=f'{mid}_joint', type='hinge',
                       pos=' '.join(f'{x:.6f}' for x in joint_pos),
                       axis=axis_str,
                       range=f'{rng[0]} {rng[1]}')
            # アクチュエータ
            SubElement(actuator_elem, 'motor',
                       name=f'{mid}_joint', joint=f'{mid}_joint',
                       gear='150', ctrllimited='true', ctrlrange='-1 1')

        # カプセルジオメトリ
        fromto = ' '.join(f'{x:.6f}' for x in [*cap['start'], *cap['end']])
        SubElement(body_elem, 'geom', type='capsule',
                   fromto=fromto, size=f'{cap["radius"]:.6f}')

        # 子ボディを再帰的に追加
        children = [b for b in bodies if b['parent'] == body_name]
        for child in children:
            add_body(child['name'], body_elem)

    root_body = next(b for b in bodies if b['parent'] is None)
    add_body(root_body['name'], worldbody)

    return tostring(root_elem, pretty_print=True).decode()
```

---

## 4. 検証フロー（全体チェックリスト）

```
□ スケール: 最大寸法が 0.05〜2.0m の範囲に収まっている
□ 座標系: Z-up（地面が z=0 の平面）
□ ground clearance: 全ボディの最低点 z > 0
□ 干渉: 隣接カプセルの最近接距離 > -0.005m
□ COM: 各ボディの重心がカプセル内に収まっている
□ 慣性: inertiafromgeom=true を使用（または手動で > 1e-4）
□ ジョイント軸: FK で計算したワールド座標に準拠
□ 命名規則: str(ind+1)+parent_name 形式を使用
□ MuJoCo ロード: mujoco.MjModel.from_xml_string() でエラーなし
□ 10ステップシミュレーション: 速度・位置が有限値のまま
```

**最後の 2 項目を自動検証するスクリプト:**

```python
import mujoco
import numpy as np

def validate_xml(xml_path: str, n_steps=10) -> bool:
    with open(xml_path) as f:
        xml_str = f.read()

    try:
        model = mujoco.MjModel.from_xml_string(xml_str)
    except Exception as e:
        print(f"[FAIL] XML ロードエラー: {e}")
        return False

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        print(f"[FAIL] {n_steps} ステップ後に数値発散")
        return False

    max_vel = np.abs(data.qvel).max()
    if max_vel > 50.0:
        print(f"[WARN] 速度が大きすぎる: max_vel={max_vel:.1f} m/s（爆発の兆候）")
        return False

    print(f"[OK] {n_steps} ステップ正常完了")
    return True
```

---

## 5. 想定スクリプト構成

```
scripts/
  mesh_to_xml/
    __init__.py
    normalize.py       ← Stage 1: スケール・座標系統一
    fk_solver.py       ← Stage 2: FK による絶対座標解決
    capsule_fit.py     ← Stage 3: カプセルフィッティング
    validator.py       ← Stage 4: 干渉・COM・慣性チェック
    xml_generator.py   ← Stage 5: MuJoCo XML 生成
    pipeline.py        ← 上記を統合したエントリポイント
```

**`pipeline.py` の使い方（想定）:**

```bash
python scripts/mesh_to_xml/pipeline.py \
  --meshes-dir mesh_parts/ \
  --topology topology.json \
  --output assets/mujoco_envs/my_robot.xml \
  --validate
```

---

## 6. 既知の落とし穴と対策

| 落とし穴 | 対策 |
|---------|------|
| Tripo3D が複数パーツを 1 メッシュとして出力する | 手動セグメンテーション or SAM2 等でパーツ分割してから入力 |
| OBB がメッシュ全体の向きにフィットしない（凹凸が大きい） | 凸包（convex hull）に対して OBB を計算する |
| カプセルが StackelbergPPO の size 範囲（0.03〜0.10m）外になる | `clamp()` 後、元メッシュとの乖離を警告する |
| ジョイント位置がユーザー指定できない | `topology.json` の `offset_from_parent` で手動指定（自動推定は難しい） |
| 複数の子ボディが同じ親に繋がる（max_nchild=2 超過） | topology.json の段階で `max_nchild=2` を守るよう設計する |
| 地面と初期衝突（ロボットが地面に埋まっている） | `estimate_ground_clearance()` で自動的に初期高さを調整 |
| ジョイント軸が実際の動作と垂直 | FK で親ボディの向きに応じて軸を変換する |

---

## 7. 関節軸問題：初学者ユーザーへの対策方針

### 問題の整理

スケルトン抽出（`skeleton_extract.py`）は関節位置をおおむね正しく推定できるが、**関節軸の向きが90°ずれる**問題が残る。

- 現実装の軸推定: `cross(bone方向, Y軸)` → ボーン方向と直交するが、実際の回転軸とは一致しない
- 可動域: `[-60, 60]` または `[-90, 90]` のハードコード — メッシュ情報を一切使わない
- StackelbergPPO の `joint_params` は現状 `{}` (空) = **軸最適化を行わない**

ユーザーがロボットの形状を正しくスケッチしても、関節軸が間違っていると制御が意図通りにならない。初学者ユーザーが関節軸の種類（Y軸/Z軸など）を手動指定するのは現実的でない。

### Stackelberg PPO 側で軸を最適化できるか

コード（`xml_robot.py`）に軸最適化の仕組みは実装済み（theta/phi の極座標で表現）。`joint_params` に以下を追加すれば有効になる:

```yaml
joint_params:
  axis:
    lb: [0, -3.14]       # theta/phi の下限（全方向を探索する場合）
    ub: [3.14, 3.14]     # theta/phi の上限
```

ただし **元論文のどの設定でも有効化されていない未検証機能**。リスクは以下:
- リーダーが軸を変えるたびに「アクションの意味（トルク方向）」が変わる → フォロワー方策が不安定化する
- 全方向探索（lb=[0,-π], ub=[π,π]）は探索空間が広すぎる

**有効化するなら**: 探索範囲を初期推定値の近傍（±30〜45°）に絞った上で実験で確認する。

### 現実的な対策案（優先度順）

#### 案1: 平面推定による軸の自動決定（推奨）

シリアルチェーンアームの場合、スケルトンノード列に PCA をかけると「アームの張る平面」が求まる。その**平面の法線ベクトル**がすべての関節軸として最適な初期値になる。

```python
# スケルトンノード位置列に対して PCA
positions = np.array([body['_position_world'] for body in bodies])
_, _, Vt = np.linalg.svd(positions - positions.mean(0), full_matrices=False)
# 第1主成分 = アーム伸長方向、第2主成分 = アームが張る平面の横方向
# 法線 = 第1主成分 × 第2主成分
normal = np.cross(Vt[0], Vt[1])
normal /= np.linalg.norm(normal)
# → 全関節に normal を axis として設定
```

現在の「cross(bone, Y軸)」よりも形状を利用しており、自然な初期値が得られる。

#### 案2: タスク別デフォルト軸（最もシンプル）

タスクと設置姿勢がわかれば、軸はほぼ決まる:

| タスク | アームの主運動 | 推奨デフォルト軸 |
|---|---|---|
| Pusher（水平押し） | XZ平面内でスイング | Y軸 `[0,1,0]` |
| Reach（先端到達） | XZ平面内でリーチ | Y軸 `[0,1,0]` |
| 3D リーチ（将来） | 複数平面 | 交互軸（Y-Z-Y-Z） |

固定根本アームのスコープ内では「全関節 Y軸」をデフォルトとし、スケルトン抽出が何を返しても上書きする実装で十分機能する。

#### 案3: 限定的 PPO 軸最適化（補正ネット）

案1 or 案2 で良い初期値を与えた上で、`joint_params` に ±30° 程度の探索範囲を設定する。スケルトン抽出の誤差を PPO が吸収する。

```yaml
joint_params:
  axis:
    lb: [-0.5, -0.5]   # 初期値近傍 ±~30° に相当する極座標範囲
    ub: [0.5, 0.5]
```

フォロワー不安定化のリスクがあるため、**実験で安定性を確認してから採用**する。

### 実装ロードマップ

```
今すぐ: 案2（タスク別デフォルト軸）をスケルトン抽出の出力に上書きする処理を追加
  → skeleton_extract.py に --task-axis オプションを追加（pusher → Y軸固定など）

Tripo3D パイプライン統合時: 案1（PCA 平面法線）を実装
  → スケッチの形状からアームの張る平面を自動推定

安定性確認後: 案3（限定 PPO 軸最適化）を実験
  → joint_params に ±30° の軸最適化を追加し、G/H 系実験で比較
```
