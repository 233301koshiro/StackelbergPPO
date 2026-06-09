# トポロジー固定・属性値最適化モード

## 概要

通常の StackelbergPPO では Leader（形態最適化）が毎エポック以下の 2 フェーズを実行する。

| フェーズ | 内容 | 変更対象 |
|---------|------|---------|
| skeleton_transform | ボディの追加・削除 | **トポロジー（形状）** |
| attribute_transform | リンク長・太さ・ギア比を調整 | **属性値** |

このページでは「**トポロジーは固定し、属性値だけを Stackelberg 最適化する**」設定を説明する。

**ユースケース:**
- 人間が設計したロボット構造（スケッチ起源など）を出発点に、物理パラメータだけを自動チューニングしたい
- スケッチ → 3D生成 → セグメンテーション → このフレームワークで制御+パラメータを同時に学習したい

---

## 1. 今すぐできること（設定変更のみ）

### skeleton_transform を無効化する

`design_opt/cfg/pusher.yml`（または独自 yml）に以下を追加・変更する:

```yaml
skel_transform_nsteps: 1      # 最小 1 ステップで即終了（実質スキップ）
enable_remove: false           # ボディ削除を禁止

add_body_condition:
  max_nchild: 0                # ボディ追加も禁止（0 以上の子を持てない）
```

これで skeleton_transform フェーズで何も起きなくなり、毎エポック attribute_transform → execution のみ実行される。

### 現在最適化できる属性値

```yaml
robot:
  body_params:
    offset:           # リンクの向き・長さ（x,y 成分）
      type: 'xy'
      lb: [-0.5, -0.5]
      ub: [0.5, 0.5]

  geom_params:
    size:             # カプセルの半径
      lb: 0.03
      ub: 0.10
    ext_start:        # カプセル開始オフセット
      lb: 0.0
      ub: 0.2

  actuator_params:
    gear:             # モーターのギア比（出力トルクに比例）
      lb: 20
      ub: 400

  joint_params:
    axis:             # 関節軸方向（theta, phi で球面座標）
      lb: [0, -6.28]
      ub: [3.14, 6.28]
```

`joint_params` に `axis` を追加すると関節軸方向も最適化できる（`Joint.get_params/set_params` が対応済み）。

---

## 2. コード追加で対応すること（推奨）

### 2-a. `fix_skeleton` フラグ（小変更）

設定で無効化する方法は副作用（1 ステップの無駄な推論）があるため、明示的なフラグを追加する方が清潔。

**`design_opt/utils/config.py`** に追加:
```python
self.fix_skeleton = FLAG.get('fix_skeleton', False)
```

**`design_opt/envs/pusher.py`**（および `ant.py`）の `step()` 内:
```python
if self.stage == 'skeleton_transform':
    if self.cfg.fix_skeleton:
        self.transit_attribute_transform()
        ob = self._get_obs()
        return ob, 0.0, False, False, {'use_transform_action': True, 'stage': 'skeleton_transform', 'reward_ctrl': 0.0}
    # ... 既存の処理
```

設定:
```yaml
fix_skeleton: true
```

### 2-b. 関節可動域の最適化（中規模変更）

現状 `Joint` クラスは `gear`（アクチュエータ経由）と `axis` のみ最適化対象。
`range`（可動域）も最適化するには以下を追加する。

**`khrylib/robot/xml_robot.py`** の `Joint` クラス:

```python
# get_params() に追記
if 'range' in self.param_specs:
    if self.type == 'hinge':
        if get_name:
            param_list += ['range_lo', 'range_hi']
        else:
            r = normalize_range(
                self.range,
                np.array([self.param_specs['range']['lb']] * 2),
                np.array([self.param_specs['range']['ub']] * 2)
            )
            param_list.append(r)
    elif pad_zeros:
        param_list.append(np.zeros(2))

# set_params() に追記
if 'range' in self.param_specs:
    if self.type == 'hinge':
        self.range = denormalize_range(
            params[:2],
            np.array([self.param_specs['range']['lb']] * 2),
            np.array([self.param_specs['range']['ub']] * 2)
        )
        params = params[2:]
    elif pad_zeros:
        params = params[2:]

# sync_node() に追記
self.node.attrib['range'] = ' '.join([
    f'{np.rad2deg(x):.1f}' for x in self.range
])
```

設定（yml）:
```yaml
joint_params:
  range:
    lb: -1.57   # -90度（ラジアン）
    ub:  1.57   # +90度
```

---

## 3. 初期形状の指定方法

### コード上の仕組み

初期形状は `design_opt/envs/pusher.py`（および `ant.py`）の `__init__` で決まる（[pusher.py:27-30](../../design_opt/envs/pusher.py#L27-L30)）:

```python
if self.cfg.xml_name == "default":
    self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", "pusher.xml")
else:
    self.model_xml_file = os.path.join(cfg.project_path, "assets", "mujoco_envs", f"{self.cfg.xml_name}.xml")
```

`xml_name` は `design_opt/utils/config.py:77` で Hydra の設定から読み込まれる:

```python
self.xml_name = FLAG.get('xml_name', 'default')
```

### 初期形状を変えるには

1. `assets/mujoco_envs/` に任意の MuJoCo XML を置く
2. 学習起動時に `xml_name=ファイル名（拡張子なし）` を渡す

```bash
# assets/mujoco_envs/my_robot.xml を初期形状として使う
USE_CHOREONOID=1 choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  xml_name=my_robot \
  hydra.run.dir=single_run/my_robot
```

または `design_opt/conf/config.yaml` の `xml_name` フィールドを直接変更してもよい:

```yaml
xml_name: "my_robot"   # デフォルトは "default"（= pusher.xml）
```

### デフォルト形状（pusher.xml）

変更しない場合は `assets/mujoco_envs/pusher.xml` が使われる。これは root 球体 + 4 本腕の対称な十字形（各腕 1 関節・長さ 0.4m）。

```
"0"  root（球体 r=0.25）
 ├── "1"  腕（+x+y 方向）
 ├── "2"  腕（-x+y 方向）
 ├── "3"  腕（-x-y 方向）
 └── "4"  腕（+x-y 方向）
```

---

## 4. スケッチ → XML 変換パイプライン

### 想定フロー

```
手描きスケッチ
  ↓ Gemini / GPT-4V
構造記述 JSON（ノード・エッジ・初期寸法）
  ↓
Tripo3D / Meshy 等
各パーツの 3D メッシュ（GLB/OBJ）
  ↓ trimesh でカプセル近似
初期リンク長・半径を推定
  ↓ scripts/mesh_to_xml/pipeline.py
    （FK による座標解決・カプセルフィット・干渉チェックを実施）
    詳細設計 → docs/研究応用/mesh_to_xml_pipeline.md
MuJoCo XML（命名規則に準拠）
  ↓
xml_name=my_robot + fix_skeleton=true で学習
```

### MuJoCo XML の命名規則

このフレームワークは `reindex()` が呼ばれた時点でボディ名を以下のルールで上書きする。
初期 XML を手書きする場合は最初からこの規則に従う必要がある。

```
root body         = "0"
root の第1子      = "1"       (index 1, parent "0" → "1" + "" = "1")
root の第2子      = "2"
root の第3子      = "3"
"1" の第1子       = "11"      (index 1, parent "1" → "1" + "1" = "11")
"1" の第2子       = "21"      (index 2, parent "1" → "2" + "1" = "21")
"11" の第1子      = "111"
"11" の第2子      = "211"
"2" の第1子       = "12"      (index 1, parent "2" → "1" + "2" = "12")
```

例：4脚ロボット（各脚 2 セグメント）のボディツリー

```
"0"  torso              (depth 0)
 ├── "1"   FL_hip       (depth 1)
 │    └── "11" FL_knee  (depth 2)   ← 末端リンク（depth < max_body_depth=4）
 ├── "2"   FR_hip
 │    └── "12" FR_knee
 ├── "3"   BL_hip
 │    └── "13" BL_knee
 └── "4"   BR_hip
      └── "14" BR_knee
```

3 セグメント脚（max_body_depth=4 ギリギリ）:
```
"1" → "11" → "111"   (depth 1→2→3)
```

### XML テンプレート（4脚・1関節/脚の最小例）

```xml
<mujoco model="ant">
  <!-- assets, visual, compiler, option, default は pusher.xml を流用 -->
  <worldbody>
    <body name="0" pos="0 0 0.5">
      <joint name="0_joint" type="free" .../>
      <geom type="sphere" size="0.15"/>

      <body name="1" pos="0.2 0.2 0">          <!-- FL -->
        <joint name="1_joint" type="hinge" axis="0 1 0" range="-60 60"/>
        <geom type="capsule" fromto="0 0 0  0.3 0 -0.3" size="0.05"/>
        <body name="11" pos="0.3 0 -0.3">       <!-- FL lower -->
          <joint name="11_joint" type="hinge" axis="0 1 0" range="-60 0"/>
          <geom type="capsule" fromto="0 0 0  0 0 -0.3" size="0.04"/>
        </body>
      </body>
      <!-- "2","3","4" と "12","13","14" を同様に定義 -->
    </body>
    <!-- cube（pusher タスクのみ） -->
  </worldbody>
  <actuator>
    <motor joint="1_joint"  gear="150" ctrlrange="-1 1" ctrllimited="true" name="1_joint"/>
    <motor joint="11_joint" gear="150" ctrlrange="-1 1" ctrllimited="true" name="11_joint"/>
    <!-- ... -->
  </actuator>
</mujoco>
```

### capsule 近似スクリプトの概要

```python
import trimesh

mesh = trimesh.load("FL_thigh.glb")

# 主軸方向を取得（最長方向 = カプセルの軸）
obb = mesh.bounding_box_oriented
extents = obb.primitive.extents   # [lx, ly, lz]
longest_axis = np.argmax(extents)

length = extents[longest_axis]
radius = np.mean(np.delete(extents, longest_axis)) / 2

print(f"fromto 長さ: {length:.3f}m  半径: {radius:.3f}m")
```

出力値を XML の `fromto` と `size` に使う。

---

## 5. 学習起動コマンド

```bash
# 初期形状XMLを配置
cp my_robot.xml assets/mujoco_envs/my_robot.xml

# トポロジー固定・属性値のみ最適化
USE_CHOREONOID=1 choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  xml_name=my_robot \
  fix_skeleton=true \
  hydra.run.dir=single_run/my_robot_attr_only
```

`fix_skeleton=true` が実装済みなら上記で動く。
未実装の場合は yml に以下を追記して代替:

```yaml
skel_transform_nsteps: 1
enable_remove: false
add_body_condition:
  max_nchild: 0
```

---

## 6. 制約まとめ（フレームワーク側のハード上限）

| 項目 | 値 | 備考 |
|------|-----|------|
| 最大ボディ深さ | 4 | depth 0（root）〜3（末端）まで |
| 1ボディの最大子数 | 2 | `max_nchild=2`（デフォルト）|
| リンク半径 | 0.03〜0.10 m | `geom_params.size` |
| リンク長（offset） | ±0.5 m | `body_params.offset` |
| ギア比 | 20〜400 | `actuator_params.gear` |
| ジョイント型 | hinge / free のみ | prismatic 等は非対応 |
| ジオメトリ型 | capsule / sphere のみ | box/cylinder は無視される |

実際の四脚ロボット（Spot: 脚径 ~0.04m、リンク長 ~0.3m）はほぼ範囲内に収まる。
ただし arm の `max_nchild=2` 制約から、1つのボディから 3 本以上枝分かれする形状（胴体から 4 本脚を直接生やすなど）はできない。
その場合は root の子として 4 つのヒップボディを並べ、そこから脚を生やすツリー構造にする。

---

## 7. 今後の拡張案

| 機能 | 難易度 | 内容 |
|------|--------|------|
| `fix_skeleton` フラグ | 小 | pusher.py / ant.py に 5 行追加 |
| joint range 最適化 | 中 | Joint クラスに range get/set/sync を追加（上記 2-b）|
| 3D メッシュ → XML 変換スクリプト | 中 | trimesh で capsule fitting、XML テンプレート生成 |
| 関節位置の自動推定 | 大 | セグメント間の接触点 or ユーザー指定 |
| スケッチ → JSON → XML | 大 | Gemini API でノード・エッジを構造化し XML 生成 |
