# Choreonoid シミュレーション層の解説

> **補足（2026-08-06）**: 本ファイルは Choreonoid 層の**仕組み**を説明したもので、内容は現在も有効。
> ただし**この層で見つかった罠は書かれていない**。以下は [デバッグ戦記.md](../デバッグ戦記.md) を参照:
> Bug 13（`_check_initial_contact()` の2関節ハードコード）、Bug 16（平面アーム専用の `[:2]` 切り詰めで
> 縦型4関節の先端位置が常にゼロになる）、Bug 17（`AISTSimulatorItem` の録画モード既定値で
> Scene 表示が更新されない）。
> また **学習時の `.body` 生成は `khrylib/rl/envs/common/mujoco_env_choreonoid.py` が行う**。
> `scripts/dynamic_body_updater.py` は実質デッドコードなので、描画・形態を直すときに触っても効かない。

ネットワーク（ポリシー・価値関数）をブラックボックスとしたとき、Choreonoid がどのように機械学習の「環境」として機能しているかを説明する。

---

## ⭐ Choreonoid / AIST / Bullet の関係（初学者向け・2026-10-02、系譜 9-198）

> ⭐ **「Choreonoid」「AISTSimulator」「Bullet」が何者で、どう違うのかを先に押さえる。**
> ⚠️ **ここを取り違えると「設定したのに効かない」で何日も溶ける**（実際に溶かした）。

### ⭐ まず結論を 1 行で

> ⭐ **Choreonoid は「入れ物」で、AIST と Bullet は「中に入れる物理エンジン」。
> 本研究は AIST を使っていて、AIST は反発係数（跳ね返り）を扱えない。**

### ⭐ 3 つの関係 — 「アプリ」と「差し替え可能なエンジン」

```
┌─────────────────────────────────────────────┐
│  Choreonoid（アプリ本体）                      │
│   ロボットモデルの読み込み・GUI・Python 実行     │
│   ⭐ 物理計算そのものはしない                    │
│                                             │
│   ┌──────────────┐  ┌──────────────┐        │
│   │ AISTSimulator │  │ BulletSim... │  …     │
│   │ ⭐ 本研究が使用 │  │ 使っていない   │        │
│   └──────────────┘  └──────────────┘        │
│      物理エンジン（差し替え可能）                │
└─────────────────────────────────────────────┘
```

⭐ **自動車に例えると**: Choreonoid が**車体**、AIST / Bullet が**エンジン**。
⭐ **車体を替えずにエンジンだけ載せ替えられる**。⚠️ **ただし載せ替えると走り方が全部変わる。**

### ⭐ それぞれの正体

| 名前 | 何者か | 作った人 | ⭐ 本研究での立ち位置 |
|---|---|---|---|
| **Choreonoid** | ロボットの**統合アプリ**（シミュレータ・動作エディタ等）。⭐ **物理計算は担当しない** | 産総研 → 現 Choreonoid Inc. | ⭐ **土台**。学習スクリプトはこの上で動く |
| **AISTSimulator** | ⭐ **Choreonoid 標準の物理エンジン。**中身は `ConstraintForceSolver`（接触力を反復計算で解く） | 産総研 | ⭐⭐ **本研究が使っている唯一のエンジン** |
| **Bullet** | ⭐ **オープンソースの有名な物理エンジン**（ゲーム・映像でも広く使われる）。Choreonoid にプラグインとして同梱 | Erwin Coumans ら（外部） | ⚠️ **同梱されビルド済みだが使っていない** |
| （AGX Dynamics） | 商用の高精度物理エンジン | Algoryx（スウェーデン） | ⛔ **ライセンス購入が要るので対象外** |

⚠️ **「AIST」は組織名（産業技術総合研究所）であり、同時にエンジン名でもある。**
⭐ **この文書で「AIST」と書いたら `AISTSimulatorItem` というエンジンのこと。**

### ⛔⛔ 一番大事な違い — **AIST は跳ね返りを扱えない**

| | ⭐ 摩擦 | ⛔ 反発係数（跳ね返り） |
|---|---|---|
| **AISTSimulator** | ⭐ **材質ごとに設定できる** | ⛔⛔ **できない** |
| **Bullet** | できる | ⭐ **できる** |
| **AGX Dynamics** | できる | ⭐ できる |

⭐ **産総研（開発元）の公式見解**（Choreonoid Discourse）:

> **「産総研エンジンでは、マテリアルの摩擦係数だけ対応しています。」**

#### ⛔ 「設定する口はあるのに効かない」という罠

⚠️ **AIST にも反発係数を設定する API は存在する**（`setEpsilon()`、Python にも公開済み）。
⛔⛔ **だが設定した値は物理計算に一度も使われない。**

```
ConstraintForceSolver.cpp で defaultCoefficientOfRestitution が出るのは 4 箇所だけ
   199: 宣言
   379: 初期化 = 0.0        ← ⭐ 既定は完全非弾性（跳ね返らない）
  2266: setter（setEpsilon が書き込む）
  2272: getter
   ⛔⛔ 求解コードからは一度も読まれない ＝ デッドパラメータ
```

⭐ **実測の反発係数 0.055 が「ほぼ 0」なのは、既定値 0.0 がそのまま使われているからである。**

### ⭐ 本研究が実際にやっていること

```
  MuJoCo 互換 XML（形態を記述）
        │  ⛔ solref / solimp / 摩擦 は変換器が読まない
        ▼
  Choreonoid の .body 形式
        │
        ▼
  AISTSimulator で物理を計算  ← ⛔ 跳ね返りは 0 のまま
        │
        ▼
  ⭐ env の step() の中で「跳ね返り」を手計算で足す
     （関節ダンピング・速度クランプと同じ場所。流儀は一貫している）
```

⭐⭐ **これは回避策ではなく、唯一の手段だった**（系譜 9-198 でソースまで確認）。
⚠️ **修論には「壁との接触は物理エンジンではなく速度反射としてモデル化した」と明記する。**
⭐ エアホッケーの壁は実際ほぼ弾性（e=0.6〜0.8）なので、モデル化としても正当である。

### ⚠️ では Bullet に替えればいいのでは？ — **替えない**

⭐ **技術的には可能**（ビルド済み・反発係数を実装・型も登録済み）。
⛔ **Python から生成する `py::init<>()` が無いだけで、差はバインディングの 1 行。**

⛔ **それでも替えない理由:**

| | |
|---|---|
| ⛔⛔ **比較可能性** | **エンジンを替えると全接触が変わる。**これまでの全実験と物理が違ってしまい、比べられなくなる |
| ⚠️ **二重適用** | この env は「**AIST が無視する量を `step()` で明示的に適用する**」設計。⛔ Bullet では同じ量を 2 回かけることになる |
| ⚠️ **時間** | 修論の残り期間で全実験を取り直すのは見合わない |

### ⭐ 覚えておくこと 3 つ

| | |
|---|---|
| ① | ⭐ **Choreonoid は入れ物。物理を計算しているのは AISTSimulator** |
| ② | ⛔ **AIST は摩擦は扱えるが跳ね返りは扱えない。**設定 API はあるが**値が使われない** |
| ③ | ⭐ **跳ね返りが要る場面は env の `step()` で手計算している。**物理エンジンの中では起きていない |

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
   ↓  mujoco_xml_to_body()
Choreonoid .body YAML（/tmp/xxxxx.body に一時書き出し）
   ↓  BodyItem.load()
Choreonoid のアイテムツリーに追加
   ↓  sim_item.startSimulation(doReset=True)
物理シミュレーション開始
   ↓  sim_item.findSimulationBody()
SimulationBody（状態読み書き用ハンドル）を取得
```

Choreonoid ネイティブの `.body` YAML フォーマットに変換してロードする。変換処理（`mujoco_xml_to_body()`）は `mujoco_env_choreonoid.py` 内に実装されている。URDF 経由ではなくネイティブ形式を使う理由は、Capsule ジオメトリを精度よく表現できることと、URDF パーサーの `<dynamics>` タグ未対応警告を回避できるため。

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

---

## コードナビゲーション

### エントリーポイント

| 用途 | ファイル |
|------|---------|
| 学習 | [`scripts/choreonoid_train.py`](../../scripts/choreonoid_train.py) → [`design_opt/train.py`](../../design_opt/train.py) |
| 評価（数値） | [`scripts/eval_cnoid_numerical.py`](../../scripts/eval_cnoid_numerical.py) |
| 評価（動画） | [`scripts/eval_cnoid_visual.py`](../../scripts/eval_cnoid_visual.py) |
| 評価（GUI） | [`scripts/eval_cnoid_viewer.py`](../../scripts/eval_cnoid_viewer.py) |

### メソッド定義場所

⚠️ **行番号は貼らない。** 2026-09-25 に照合したところ **23 件中 18 件がずれていた**
（`transit_execution` は L205 と書いてあったが実際は L472）。
コードが動くたびに腐るうえ、誰も再検査しない。**探し方だけを書く。**

```bash
grep -n "def transit_execution\|class ChoreonoidEnv" \
    design_opt/envs/pusher.py khrylib/rl/envs/common/mujoco_env_choreonoid.py
```

| ファイル | 何があるか |
|---|---|
| `design_opt/train.py` | `main_loop()`（epoch ループ）・`main()`（Hydra 入口） |
| `design_opt/agents/genesis_agent.py` | `BodyGenAgent`・`sample()`（エピソード収集）・`optimize()`（1 epoch 分の学習） |
| `design_opt/envs/pusher.py` | `PusherEnv`・`step()`（設計/実行の分岐と報酬）・`transit_execution()`（設計→実行。`reset_state` のみで reload 不要）・`reset_state()`（初期姿勢。`add_noise` で ±0.1）・`reset_robot()`・`reset_model()`・`is_fixed_base` |
| `khrylib/rl/envs/common/mujoco_env_choreonoid.py` | `mujoco_xml_to_body()`（XML → .body YAML 変換）・`ChoreonoidSimWorld`（Choreonoid 操作ラッパー）・`_setup_world()`・`load_model()`・`reset()`・`step()`（トルク書き込み＋`tickRequest()` × frame_skip）・`set_state_cmd()`・`ChoreonoidEnv`（MuJoCo 互換 API）・`do_simulation()`・`reload_sim_model()`・`get_body_com()` |

---

## ⛔ 変換器が読まないもの（**ここが最大の落とし穴**）

**MuJoCo XML に書いても Choreonoid の実走に届かない記述がある。**
⚠️ **書いてあるので「効いている」と誤解しやすく、実際に 3 回誤った。**

| 書いても届かないもの | 何が起きたか |
|---|---|
| ⛔ **`worldbody` 直下の `<geom>`**（静的な壁・床・障害物） | 変換器は **`worldbody/body` しか読まない**。⛔ **ホッケーの壁がこれで、8 回の修正すべてが空振りだった**（9-98）。⚠️ **プローブは全部 MuJoCo で回していたので正しく動き、実走と無関係だった。**分水嶺は `USE_CHOREONOID=1` の有無 |
| ⛔ **接触パラメータ**（`solref`・`solimp`・geom の `friction`・`condim`・`margin`） | **読む箇所が 1 つも無い。**⛔ 9-128 の接触の根拠づけは **MuJoCo の中だけの話だった**（9-143）。実測でも反発係数は `solref` 違いの 2 条件で **0.0545 対 0.0552** と変わらない。⭐ **学習側の実効反発係数は約 0.055** で、実在の金属・樹脂間の 0.2〜0.5 を大きく下回る |
| ⛔ **`cylinder` の geom** | 扱えるのは **`capsule` / `sphere` / `box` の 3 つだけ**。⛔ **cylinder は黙って落ちるので、形だけ存在して物理に無い**（Bug 44）。⚠️ 障害物 Reach の 2 件がこれで、回避として読めない |

```bash
# 自分で確かめる（どちらも 1 行で出る）
grep -n "gtype ==" khrylib/rl/envs/common/mujoco_env_choreonoid.py   # 扱える型
grep -c "get('solref'\|get('friction'" khrylib/rl/envs/common/mujoco_env_choreonoid.py   # → 0
```

⭐ **是正の手段は特定済み**（Choreonoid の `ContactMaterial` で反発・摩擦を直接設定できる）**が、適用していない。**
接触条件を変えると既存の全実験と物理が異なり、**比較可能性が失われる**ため（修論 6.4.2 (7)）。
