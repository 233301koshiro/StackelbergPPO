# Choreonoid シミュレーション層の解説

ネットワーク（ポリシー・価値関数）をブラックボックスとしたとき、Choreonoid がどのように機械学習の「環境」として機能しているかを説明する。

---

## なぜ Choreonoid を使うのか

物理シミュレーターとして MuJoCo を使っていた部分を Choreonoid に置き換えている。
Choreonoid の物理エンジン（`AISTSimulatorItem`）は C++ オブジェクトで、Qt アプリケーションコンテキストが存在している間だけ動く。そのため学習スクリプトは Python から直接起動できず、**必ず Choreonoid 経由で起動する**。

```bash
choreonoid --no-window --python scripts/choreonoid_train.py
```

`--no-window` を付けると GUI ウィンドウなしで起動し、Python スクリプトを PythonPlugin スレッドで実行する。Qt イベントループはバックグラウンドで動き続ける。

---

## 全体の構成

```
choreonoid プロセス
├── Qt メインスレッド（イベントループ）
└── PythonPlugin スレッド
      └── choreonoid_train.py
            └── train.py  →  BodyGenAgent
                               └── PusherEnv（タスクルール）
                                    └── ChoreonoidEnv（MuJoCo 互換 API）
                                         └── ChoreonoidSimWorld（Choreonoid 操作）
                                              └── AISTSimulatorItem（C++ 物理）
```

各層の責務：

| クラス | 責務 |
|--------|------|
| `AISTSimulatorItem` | C++ 物理計算本体 |
| `ChoreonoidSimWorld` | Choreonoid アイテムツリーの操作、物理ステップ呼び出し |
| `ChoreonoidEnv` | MuJoCo の `env.step()` / `env.reset()` 互換 API を提供 |
| `PusherEnv` | pusher タスクのルール（報酬計算・終了判定） |

---

## クラス・オブジェクト説明

### `WorldItem`（Choreonoid 組み込み）

1 つの「物理シミュレーション世界」を表すコンテナ。床・ロボット・シミュレーター設定など、その世界に属するすべての物体をこのノードの子として管理する。複数の `WorldItem` を持つことで複数の独立したシミュレーション世界を共存させることもできる。

### `BodyItem`（Choreonoid 組み込み）

1 つのロボットまたは物体モデルを保持するノード。`.urdf` / `.body` ファイルをロードしてアイテムツリーに追加することで、そのモデルがシミュレーション対象になる。本プロジェクトではロボット（毎エピソード形態が変わる）と床（固定）の 2 種類の `BodyItem` が使われる。

### `AISTSimulatorItem`（Choreonoid 組み込み）

AIST が開発した剛体物理エンジンの設定と制御を担うノード。タイムステップ・リアルタイム同期モード・接触判定パラメータなどを保持する。`startSimulation()` を呼ぶと配下の `BodyItem` たちを物理演算の対象として認識し、`tickRequest()` で 1 サブステップずつ時間を進める。

### `SimulationBody`（Choreonoid 組み込み）

`startSimulation()` 後に `findSimulationBody(name)` で取得できる実行時オブジェクト。`BodyItem` の「シミュレーション中の分身」であり、関節角度・角速度・リンク位置などをリアルタイムに読み書きできる。シミュレーションを止めると無効になるため、`startSimulation()` のたびに再取得が必要。

### `ChoreonoidSimWorld`（本プロジェクト独自）

[khrylib/rl/envs/common/mujoco_env_choreonoid.py](../../khrylib/rl/envs/common/mujoco_env_choreonoid.py) に定義。上記の Choreonoid 組み込みクラスを操作するラッパー。アイテムツリーの組み立て・URDF ロード・物理ステップ・状態読み取り・リセットなど、Choreonoid 固有の操作をすべてここに集約している。

### `ChoreonoidEnv`（本プロジェクト独自）

同ファイルに定義。`ChoreonoidSimWorld` の上に MuJoCo 互換の公開 API（`env.step()` / `env.reset()` / `env.get_body_com()` など）を被せたクラス。元々 MuJoCo の `MujocoEnv` を継承していた `PusherEnv` などのタスク環境が、コードをほぼ変えずに Choreonoid 上で動くようになっている。

---

## 起動時の初期化（`ChoreonoidSimWorld._setup_world()`）

Choreonoid には「アイテムツリー」という概念がある。シーン内のすべての物体（ロボット・床・シミュレーター設定）がツリー上のノードとして管理される。

```
RootItem（Choreonoid のルート）
└── WorldItem（1 シミュレーション世界）
      ├── BodyItem（floor.body）  ← 床
      └── AISTSimulatorItem       ← 物理エンジン設定
```

初期化時にこのツリーを Python から組み立てる。

```python
self.world_item = WorldItem()
RootItem.instance.addChildItem(self.world_item)

floor_item = BodyItem()
floor_item.load('floor.body')
self.world_item.addChildItem(floor_item)

self.sim_item = AISTSimulatorItem()
self.sim_item.setTimeStep(0.01)          # 物理の 1 サブステップ = 0.01 秒
self.sim_item.setRealtimeSyncMode(3)     # リアルタイム同期しない（全速で走る）
self.world_item.addChildItem(self.sim_item)
```

---

## ロボットモデルの読み込み（`load_model()`）

設計フェーズでロボットの形態が変わるたびに呼ばれる。

```
MuJoCo XML（xml_robot.py が生成）
   ↓  mujoco_xml_to_urdf()
URDF 文字列（/tmp/xxxxx.urdf に一時書き出し）
   ↓  BodyItem.load()
Choreonoid のアイテムツリーに追加
   ↓  sim_item.startSimulation(doReset=True)
物理シミュレーション開始
   ↓  sim_item.findSimulationBody()
SimulationBody（状態読み書き用ハンドル）を取得
```

URDF は Choreonoid が直接読める形式なので、MuJoCo XML → URDF 変換が必要になる。変換処理（`mujoco_xml_to_urdf()`）は `mujoco_env_choreonoid.py` 内に実装されている。

---

## 1 ステップの物理進行（`ChoreonoidSimWorld.step()`）

ネットワークが出力したトルク（`ctrl`）を受け取り、`frame_skip` 回（デフォルト 4 回）物理を進める。

```python
# 1. 各関節にトルクを書き込む
for i, (jname, ainfo) in enumerate(self.actuators_map.items()):
    j = b.joint(jname)
    j.u = ctrl[i] * ainfo['gear']   # u = 制御入力（トルク）

# 2. 物理を frame_skip 回進める
for _ in range(n_frames):            # n_frames = frame_skip = 4
    self.sim_item.tickRequest(True)  # 0.01 秒分の物理計算（C++）
    IU.processEvent()                # Qt イベントを処理（GUI 更新など）

# 3. 状態を読み取って返す
return _get_state_dict(sim_body)
```

1 RL ステップ = 4 サブステップ × 0.01 秒 = **0.04 秒**の物理時間が進む。

`tickRequest(True)` が C++ 物理計算の本体。`IU.processEvent()` は Qt イベントループを一回転させるためのもので、GUI モード時に 3D ビューを再描画させる役割もある。

---

## 状態の読み取り（`_get_state_dict()`）

物理ステップ後に `SimulationBody` オブジェクトから関節状態と各リンクの位置・姿勢を読み取る。

```python
for i in range(b.numJoints):
    j = b.joint(i)
    qpos.append(j.q)    # 関節角度 [rad]
    qvel.append(j.dq)   # 関節角速度 [rad/s]

for i in range(b.numLinks):
    lk = b.link(i)
    body_xpos[lk.name] = list(lk.translation)   # ワールド座標系での位置 [m]
    body_xmat[lk.name] = np.asarray(lk.rotation) # ワールド座標系での回転行列
```

この `body_xpos` が `env.get_body_com('cube')` などで使われ、報酬計算に使われる。

---

## リセット（`ChoreonoidSimWorld.reset()`）

```python
self.sim_item.stopSimulation()
for item in self.body_items.values():
    item.restoreInitialState(True)    # storeInitialState() 時のスナップショットに戻す
self.sim_item.startSimulation(doReset=True)
```

`storeInitialState()` はモデルロード直後に一度だけ呼ばれており、その時の姿勢・速度ゼロ状態がスナップショットとして保持される。リセット時はそこに戻す。

---

## 学習ループ全体の流れ

```
train.py: for epoch in range(max_epoch):
    agent.optimize(epoch)
        ↓
    agent.sample()  ← 環境からデータを集める
        ↓
    env.reset()  ← Choreonoid: シミュレーション再起動
        ↓
    for step in episode:
        ネットワーク → action（トルク）
        env.step(action)
            ├── ChoreonoidSimWorld.step()
            │     ├── 関節にトルク書き込み
            │     └── tickRequest × 4 回（物理計算）
            ├── 状態読み取り（qpos, qvel, body_xpos）
            └── 報酬計算・終了判定（PusherEnv）
        ↓
    サンプルデータ蓄積
        ↓
    ネットワーク更新（PPO）← Choreonoid は無関係（純粋な PyTorch 計算）
```

ネットワーク更新フェーズは純粋な PyTorch 計算なので Choreonoid は関与しない。Choreonoid が使われるのは `env.step()` / `env.reset()` / `reload_sim_model()` の呼び出し時のみ。

---

## 形態変更時（設計フェーズ）

設計フェーズの各ステップで `reload_sim_model(xml_str)` が呼ばれる。

```
新しい形態の MuJoCo XML
   ↓ reload_sim_model()
既存 BodyItem を detach（ツリーから除去）
   ↓
新しい XML → URDF 変換 → BodyItem ロード → シミュレーション再起動
   ↓
新しい形態でシミュレーション続行
```

設計フェーズは 6 回（skeleton×5 + attribute×1）この再ロードが走るため、Choreonoid のロードログが一気に流れる。
