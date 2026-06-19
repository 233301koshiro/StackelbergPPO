# StackelbergPPO を Choreonoid で動かす作業まとめ

## 目的

StackelbergPPO（形態・制御の共設計を行う強化学習リポジトリ）は MuJoCo + conda で動作している。
これを研究室のDockerイメージ（Choreonoid入り）上で動かし、最終的にChoreonoidをシミュレータとして使えるようにする。

**設計方針（重要）**:
- MuJoCo XML によるロボット形態生成（`xml_robot.py`）はそのまま流用する
- MuJoCo をシミュレータとして使っている部分だけを Choreonoid に置き換える
- XML→URDF 変換など中間フォーマットは引き続き使用

---

## フェーズ1: Docker環境の構築

### 問題

- 研究室提供の Choreonoid Docker イメージ (`irsl_system:noetic`) は Python 3.8 / Ubuntu 20.04 ベース
- StackelbergPPO が要求する PyTorch 2.x は Python 3.8 非対応
- RTX 5060 Ti（sm_120, Blackwell）に対応するには PyTorch 2.7 以降が必要

### 解決策: `akita_sp` イメージ

`Dockerfile.add_akita_sp`（研究室イメージを継承する個人用Dockerfile）で以下を構築:

1. **Python 3.12** + Choreonoidバインディングを **Python 3.12 用に再ビルド**
2. **PyTorch 2.7.0+cu128** をインストール（RTX 5060 Ti 対応）
3. **StackelbergPPO 依存ライブラリ**（gym, torch-geometric, hydra-core 等）を Python 3.12 でインストール
4. **MuJoCo バイナリは不要**（Choreonoid に完全移行するため）

**重要な変化**: Choreonoidバインディングが Python 3.12 で動くようになった。
これにより ZMQ による2プロセス構成が不要になった。

---

## フェーズ2: Choreonoidバックエンドの実装

### アーキテクチャの変遷

#### 旧アーキテクチャ（Python 3.8 制約時代）

```
PPO プロセス (Python 3.9)          Choreonoid プロセス (Python 3.8)
┌─────────────────────────┐        ┌──────────────────────────────┐
│  mujoco_env_choreonoid  │←─ZMQ──│  cnoid_sim_server.py         │
│  (gym.Env クライアント)   │        │  (AISTSimulator + ZMQ サーバー)│
└─────────────────────────┘        └──────────────────────────────┘
```

ZMQ が必要だった理由: Choreonoidバインディングが Python 3.8 専用だったため、
学習コード（Python 3.9+）から直接呼び出せなかった。

#### 現アーキテクチャ（Python 3.12 統一後）

```
choreonoid --no-window --python scripts/choreonoid_train.py
┌──────────────────────────────────────────────────────────┐
│  Choreonoidプロセス (Python 3.12)                         │
│   ├─ Qt イベントループ（メインスレッド）                    │
│   └─ 学習コード（PythonPlugin スレッド）                   │
│         ├─ BodyGenAgent / PusherEnv                       │
│         └─ ChoreonoidEnv → ChoreonoidSimWorld             │
│               └─ WorldItem + AISTSimulatorItem（直接呼出）│
└──────────────────────────────────────────────────────────┘
```
```
train.py: main_loop()
│
├─ agent = BodyGenAgent(cfg, ...)          ← 初期化フェーズ
│     │
│     ├─ self.env = PusherEnv(cfg)         ← タスク環境を生成
│     │     │
│     │     └─ super().__init__()          ← ChoreonoidEnv.__init__()
│     │           │
│     │           └─ self._world = ChoreonoidSimWorld()
│     │                 │
│     │                 ├─ WorldItem()          ← Choreonoidシーングラフに物理空間を追加
│     │                 └─ AISTSimulatorItem()  ← 物理エンジンをWorldItemに接続
│     │
│     ├─ self.policy_net = BodyGenPolicy(...)
│     └─ self.value_net  = BodyGenValue(...)
│
└─ for epoch in range(...):
      agent.optimize(epoch)               ← 1エポック全体
            │
            ├─ agent.sample()             ← サンプル収集
            │     │
            │     └─ ループ:
            │           env.reset()
            │           env.step(action)       ← PusherEnv.step()
            │                 │
            │                 ├─ apply_skel_action() → reload_sim_model()
            │                 │     └─ self._world.load_model()
            │                 │           └─ AISTSimulatorItem.startSimulation()
            │                 │
            │                 └─ do_simulation() → self._world.step()
            │                       └─ AISTSimulatorItem.stepSimulation()
            │
            ├─ agent.update_params()      ← PPO勾配更新（GPU）
            └─ agent.log_eval()           ← ログ出力
```

| クラス | 責任範囲 |
|--------|---------|
| `WorldItem + AISTSimulatorItem` | Choreonoid本体のC++オブジェクト。物理計算のみ。RL・MuJoCo・タスクを知らない |
| `ChoreonoidSimWorld` | Choreonoid固有のセットアップ（床ロード・シーングラフ追加）を1か所に閉じ込める |
| `ChoreonoidEnv` | MuJoCo版（`mujoco_env_gym.py`）と同じAPIを提供。`PusherEnv`はMuJoCoかChoreonoidかを意識しない |
| `PusherEnv` | タスク固有ロジック。`USE_CHOREONOID=0`にすればMuJoCo版と共存できる |

`ChoreonoidSimWorld`は「Choreonoidの操作方法」、`ChoreonoidEnv`は「MuJoCoのふりをする」、`PusherEnv`は「pusherタスクのルール」という役割分担。`WorldItem + AISTSimulatorItem`はChoreonoidのC++クラスをPythonバインディングで呼んでいるだけ。

**起動方法**:
```bash
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1
```

### 実装ファイル

#### `khrylib/rl/envs/common/mujoco_env_choreonoid.py`

`mujoco_env_gym.py` と**同じ API** を持つ drop-in replacement。
以前は ZMQ クライアントだったが、現在は `cnoid` バインディングを直接呼ぶ。

主要コンポーネント:

- **`mujoco_xml_to_body()`**: MuJoCo XML → URDF 変換器（旧 `cnoid_sim_server.py` から移設）
- **`ChoreonoidSimWorld`**: WorldItem + AISTSimulatorItem の管理クラス（同上）
- **`ChoreonoidEnv`**: `MujocoEnv` 互換クラス。`ChoreonoidSimWorld` を内部に持ち直接呼ぶ
- **`_ModelProxy`, `_DataProxy`**: 既存 env コードが `self.model.nq` や `self.data.qpos` にアクセスする部分を透過補完

#### `scripts/choreonoid_train.py`

`choreonoid --no-window --python` に渡すエントリポイント。
`sys.argv` を組み立てて `design_opt.train.main()` を呼ぶだけ。

#### `scripts/cnoid_transfer.py`（更新済み）

MuJoCo → Choreonoid 移行の自動化スクリプト。
旧版は内部で ZMQ サーバーを起動していたが、現在は `choreonoid --no-window --python` を subprocess で呼ぶ。

---

## フェーズ3: バグ修正と調査（シミュレーション精度）

### 修正1: 角速度の誤取得 (`dv` → `w`)

**問題**: 根リンクの角速度を `root.dv`（線形加速度）で取得していた。

```python
# 誤: root.dv = 線形加速度（重力 -9.81 が入る）
qvel += list(root.v) + list(root.dv)

# 正: root.w = 角速度
qvel += list(root.v) + list(root.w)
```

**影響**: `qvel[5] = -9.807`（重力加速度）が角速度として入り、報酬計算が崩壊。

### 修正2: Armature（関節慣性）の欠落

**問題**: MuJoCo XML の `<joint armature="1">` は URDF に対応フィールドがない。

armature がないと慣性が ~200 倍小さくなり、関節角が爆発した（11.18 rad → 本来 0.057 rad）。

**解決**: URDF ロード後に Choreonoid API で設定:
```python
for i in range(b.numJoints):
    j = b.joint(i)
    arm = joint_armatures.get(j.jointName, 0.0)
    if arm > 0:
        j.setEquivalentRotorInertia(arm)
```

**効果**: 11.18 rad → 0.099 rad（MuJoCo の 0.057 rad に近づく）。

### 修正3: cube の2本目スライド関節の欠落

**問題**: pusher.xml の cube ボディは2本のスライド関節（x/y方向）を持つが、
変換器が1本しか取っていなかった。

```
MuJoCo: nq=13, nv=12
Choreonoid (修正前): nq=12, nv=11  ← 1DOF少ない
Choreonoid (修正後): nq=13, nv=12  ✅
```

**解決**: 複数関節ボディを仮想リンクで連結する処理を追加。

### 修正4: collision 要素の欠落

**問題**: URDF に `<visual>` しかなく `<collision>` がないと Choreonoid は接触計算しない。

**解決**: 全ジオメトリタイプ（capsule/sphere/box）に `<collision>` 要素を追加。

### 修正5: capsule 慣性公式の誤り

```python
# 誤: m_cap * (2r²/5 + l²/2 + 3lr/8)
# 正: m_cap * (2r²/5 + (l/2 - 3r/8)²)
#              ↑ 半球の重心は平面から 3r/8 内側にある
```

**修正後**: MuJoCo の慣性値との誤差 1.6% 以内。

---

## MuJoCo XML → URDF の情報損失と補完

```
MuJoCo XML の情報
  │
  ├── URDF で表現できるもの（シミュレータ非依存）
  │     質量・慣性テンソル・ジオメトリ・joint damping・関節タイプ・軸・可動域
  │     → 誤差 1.6% 以内で渡せる
  │
  └── URDF に存在しないもの（シミュレータ固有）
        armature=1       → URDF規格なし → Choreonoid APIで補完（近似）
        gear=150         → URDF規格なし → トルク適用時に手動スケール
        integrator=RK4   → URDF規格なし → Choreonoidは別の積分法（差1.5%）
        solimp/solref    → URDF規格なし → 接触剛性・ソルバが消える ← 残差の主因
        condim/friction  → URDF規格なし → 接触摩擦モデルが消える
```

**結論**: 接触ソルバと積分法の差がある（残差 ~32%）ため、
MuJoCo 学習済み重みをそのまま転用するより Choreonoid でゼロから再学習する方が確実。

---

## フェーズ4: Python 3.12 移行に伴う追加修正

### 修正6: NumPy 2.0 非互換 (`body_a.item()`)

**問題**: `ctrl[aind] = body_a` が NumPy 2.0 で `ValueError` になる。

`body_a` が shape `(1,)` の配列のとき、スカラー位置への代入は明示的な変換が必要。

```python
# 修正前
ctrl[aind] = body_a
# 修正後
ctrl[aind] = body_a.item()
```

全 env ファイル（pusher/gap/hopper/ant/walker/swimmer/stair/stairhard）に適用。

### 修正7: wandb の条件付き import

**問題**: `wandb` が NumPy 2.0 非対応でインポート時にクラッシュする。
`enable_wandb=false` でも import 文は実行されるため影響が出る。

```python
# 修正後
try:
    import wandb
except Exception:
    wandb = None
```

### 修正8: Hydra 1.3 対応

**問題**: Hydra 1.2.0 が Python 3.12 の dataclasses の仕様変更で動かない。

- `hydra-core` を 1.2.0 → 1.3.2 に更新
- `design_opt/conf/__init__.py` を新規作成（Hydra 1.3 はモジュールとして認識させる必要がある）

### 修正9: `sim.setRealtimeSyncMode(3)` → `sim.NonRealtimeSync`

**問題**: Choreonoid 2.3 では数値リテラルではなく enum を使う必要がある。

```python
# 修正前
sim.setRealtimeSyncMode(3)
# 修正後
sim.setRealtimeSyncMode(sim.NonRealtimeSync)
```

### 修正10: サンプリング後にシミュレーションを停止しない問題

**問題**: eval サンプリングで必要数を収集した後、実行中のエピソードを止めずに Python が抜けると
Choreonoid がシミュレーションを走らせ続け、プロセスが数分間終了しない。

**解決**: 学習ループ終了後に `env.close()` を呼ぶ。

```python
# design_opt/train.py の学習ループ末尾
agent.logger.info('training done!')
if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
    agent.env.close()
```

```python
# mujoco_env_choreonoid.py
def close(self):
    if self._world.is_running:
        self._world.sim_item.stopSimulation()
        self._world.is_running = False
```

**注意**: `close()` をサンプリングループ内（各エポック後）で呼ぶと
Qt のシグナル連鎖を壊してクラッシュする。学習全体の終了時のみ呼ぶこと。

---

## 動作確認・スレッド数ベンチマーク結果

### シングルスレッド基礎確認

`min_batch_size=500`, `eval_batch_size=200`, 1スレッドでのプロセス全体計測:

```
[WALL] total=12.9s   （Python 計測: T_sample=4.83s  T_update=5.43s  T_eval=3.30s）
```

T_sample のボトルネックは URDF ロード（骨格変換ステップごとに発生、約 7回/エピソード）。

### スレッド数ベンチマーク

`min_batch_size=5000`, `eval_batch_size=1000`, RTX 5060 Ti 16GB, CPU 20コア

| スレッド | T_sample | T_update | T_eval | wall合計 | GPU avg | GPU peak | sample 倍速 |
|---------|----------|----------|--------|---------|---------|----------|------------|
| 1 | 33.5s | 40.3s | 3.7s | 80s | 65% | 100% | 1.0x |
| 2 | 11.7s | 46.7s | 3.1s | 75s | 85% | 100% | **2.9x** |
| 4 | 6.7s | 35.2s | 3.2s | **69s** | 61% | 100% | **5.0x** |
| 8 | 6.4s | 38.5s | 2.5s | 91s | 37% | 100% | 5.2x |

**考察**:
- T_sample は 4スレッドで 5.0倍速（ほぼ線形スケール）
- 4→8スレッドでは頭打ち（6.7s→6.4s）。ワーカー起動コスト・通信オーバーヘッドが支配的になる
- T_update はスレッド数に依存しない（GPU 上での勾配計算）が GPU peak 100% に達する
- 8スレッドは wall 合計が逆に増加（91s）。4スレッドが最適バランス
- GPU avg 利用率: 2スレッドで 85%（最高効率）。8スレッドで 37%（サンプリング時間が短くなり update 割合が下がる）

**推奨: `num_threads=4`**（wall 69s/epoch）

```bash
# 4スレッドで学習（推奨設定）
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid \
  enable_wandb=false

# 評価（チェックポイントから）
USE_CHOREONOID=1 choreonoid --no-window --python \
  scripts/eval_cnoid_numerical.py -- \
  --restore_dir single_run/pusher_cnoid
```

### マルチスレッド実装の仕組み

`num_threads > 1` かつ `USE_CHOREONOID=1` のとき、`fork()` ではなく
spawn + 永続ワーカー方式を使う（`design_opt/utils/worker_pool.py`）。

```
メインプロセス（choreonoid --no-window --python）
  └─ GPU でネットワーク更新（Stackelberg 勾配）

Worker 0（choreonoid --no-window --python scripts/worker_sampler.py）
Worker 1（同上）   ← 全ワーカーが並列でサンプリング
Worker N（同上）

通信: multiprocessing.Pipe + send_bytes/recv_bytes（numpy 変換で torch pickle 認証回避）
```

---

## Choreonoid 起動・エピソードリセット時のログ解説

`choreonoid --python` でヘッドレス起動すると、エピソードリセット（`env.reset()`）のたびに以下のメッセージが出力される。1かたまりが1エピソードリセット分。

```
Loading Body "/tmp/tmp1d57gp00.body"
 -> ok!
Loading Body "/tmp/tmpa8b2cx1f.body"
 -> ok!
Simulation by AISTSimulator has finished at 0.02 [s].
Computation time is 0.026 [s], computation time / simulation time = 1.3.
Simulation by AISTSimulator has started.
```

**注**: URDF 経由（旧実装）では `Warning: 'dynamics' tag is currently not supported.` が関節数分繰り返されていたが、`.body` ネイティブ形式への移行後はこの警告はなくなった。

### 各メッセージの意味

| メッセージ | 意味 | 問題か？ |
|-----------|------|---------|
| `Loading Body "/tmp/tmpXXXX.body"` | エピソードごとに MuJoCo XML → `.body` YAML 変換した一時ファイルをロード | 正常 |
| `-> ok!` | .body ロード完了 | 正常 |
| `Simulation by AISTSimulator has finished at 0.02 [s].` | 初期化用の極短いシミュレーション完了 | 正常 |
| `Simulation by AISTSimulator has started.` | 本番エピソード開始 | 正常 |

### ボディ名の命名規則

ログに出る `bodies=['0', '1', '11', '12', '111', ...]` はロボットのツリー構造をエンコードしたもの（[xml_robot.py の `reindex()` メソッド](../khrylib/robot/xml_robot.py)）。

```
ルール: 子の名前 = str(自分が親の何番目の子か) + 親の名前（ルートは '0' で除く）
```

```
'0'              ← ルート（胴体）
├── '1'          ← ルートの1番目の子
│   ├── '11'     ← '1' の1番目の子（"1" + "1"）
│   │   ├── '111'   ← '11' の1番目の子
│   │   └── '211'   ← '11' の2番目の子（"2" + "11"）
│   ├── '12', '13', '14'
├── '2'
│   ├── '22', '23', '24'
├── '3', '4'
```

pusher の `index_base = 5`（最大4子まで許容）はこの名前を5進数として `int(name, base=5)` で観測ベクトルのインデックスに変換するために使用される。

---

## 修正11: `mujoco_xml_to_body()` グローバル座標変換バグ（2026-06-09）

### 問題

`mujoco_env_choreonoid.py` の `mujoco_xml_to_body()` が、`coordinate="global"` の MuJoCo XML を URDF に変換するとき、ボディの `pos` と geom の `fromto`/`pos` をワールド座標のまま URDF の親相対オフセットとして使っていた。

URDF では `<joint>` の `<origin>` と `<link>` の geometry `<origin>` はすべて**親リンクローカル座標**で記述する必要がある。

**影響範囲**:
- depth-1 ボディ（`"1"`,`"2"`,…）: 親は常にワールド原点 → 誤差ゼロ（偶然正しい）
- **depth-2 以降のボディ（`"11"`,`"21"`,`"111"`,…）: 全て影響を受ける**

誤差の大きさ = depth-1 親のワールド座標値（通常 0.2〜0.5 m 程度）

### 影響

- **物理シミュレーション**（AISTSimulator）: 関節位置・衝突形状・慣性テンソルが間違った位置に配置される → depth-2 以降の脚・腕の物理が不正確
- **可視化**（Choreonoid GUI）: depth-2 以降のリンクが視覚的にも誤った位置に表示される
- **学習への影響**: 初期形態は全ボディが原点にあるため序盤の誤差はゼロ。depth-2 ボディが出現し親ボディが原点から離れるにつれ誤差が拡大する

### 修正内容

`process_body()` に親のグローバル位置を引き回し、子ボディの位置を親相対に変換するように変更。  
`add_link()` にボディのグローバル位置を渡し、`fromto`/`pos` をボディローカル座標に変換するように変更。

```python
# process_body: 親グローバル位置を引き回して親相対オフセットを計算
def process_body(body_el, parent_link_name, parent_global_pos=None):
    global_pos = np.array(parse_vec(body_el.get('pos', '0 0 0')))
    bpos = (global_pos - parent_global_pos).tolist()   # ← 修正点
    add_link(bname, geom_el, default_density, body_global_pos=global_pos)
    ...
    process_body(child_body, bname, global_pos)         # ← 親 pos を伝播

# add_link: fromto をボディローカルに変換
def add_link(name, geom_el, density, body_global_pos=None):
    p0 = np.array(fromto[:3]) - body_global_pos        # ← 修正点
    p1 = np.array(fromto[3:]) - body_global_pos        # ← 修正点
```

`coordinate="local"` の XML では変換は不要なため、コンパイラ属性を読んで処理を分岐する。

### 経緯

この修正以前に実施した **pusher_cnoid** と **crawler_cnoid (epoch 134 まで)** の学習結果は、depth-2 以降のボディが存在する場合に不正確な物理下での結果となっている。修正後は **crawler_cnoid_v2** として再学習を行い、正しい物理での結果と比較する。

---

---

## 修正12: cube ボディのフロア接触による `tickRequest` デッドロック（2026-06-19）

### 問題

`rrbot_arm.xml` で cube の body を `pos="1.0 0 0.15"` に設定した結果、cube の底面（z=0.15 - half_size=0.15 = **z=0**）がChoreonoidフロア（surface at z=0）と完全接触した状態になっていた。

`env.reset()` → `ChoreonoidSimWorld.reset()` 内の `stopSimulation() + startSimulation(doReset=True) + tickRequest(True)` のサイクルで AIST ソルバーが接触拘束を解こうとしてデッドロック。初回の `load_model()`（シミュレーション未起動状態）では `tickRequest(True)` が正常終了するため、env 初期化時（ワーカーが "ready" を出力するまで）は問題が顕在化せず、最初のエピソードリセット時にのみハングした。

モバイルロボット（`pusher.xml`）の cube は z=1.02 で空中に浮いているため同じコードでハングしなかった。

### 修正内容

- `rrbot_arm.xml`: cube を `pos="1.0 0 0.20"` に変更 → 底面 z=0.05（フロア非接触）
- `mujoco_env_choreonoid.py`: `stopSimulation()` の直後に `IU.processEvent()` を追加 → 停止シグナルを確実に処理してから再起動
- `scripts/smoke_test_cnoid.py` 追加: 3エピソード完走・NaN/Inf なし・exec_reward > 0 を検証するヘルスチェックスクリプト（学習前の事前確認用）

### 影響・補足

- arm は z=0.15 のまま。cube（z=0.05〜0.35 の高さ範囲）の側面にアーム先端（z=0.15）が当たるため水平プッシュのジオメトリは維持。
- cube は x/y スライド関節のみ（z自由度なし）なので、z=0.20 で初期化すれば重力で落下することはない。

---

## 既知の制限

| 項目 | 状態 |
|------|------|
| 1スレッド学習 | ✅ 動作確認済み |
| マルチスレッド学習 (`num_threads=4` 推奨) | ✅ 動作確認済み（5.0倍速） |
| `fork()` ベースのマルチプロセス | ❌ Qt / CUDA が fork 後に壊れる |
| 正常終了 (`B.App.exit()`) | ⚠️ ハング → `os._exit(0)` で強制終了 |
| gym → gymnasium 移行 | ⚠️ NumPy 2.0 警告あり（動作に支障なし）|
| URDF キャッシュ | 🔜 骨格変換ごとの再ロードを削減できる余地あり |
