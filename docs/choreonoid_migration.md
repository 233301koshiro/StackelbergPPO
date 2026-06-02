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

ZMQ・別プロセス・別スレッド構成が一切不要。
学習コードと Choreonoid が同一プロセスで動く。

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

- **`mujoco_xml_to_urdf()`**: MuJoCo XML → URDF 変換器（旧 `cnoid_sim_server.py` から移設）
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

## 動作確認結果

`min_batch_size=500`, `eval_batch_size=200`, 1スレッドでの計測:

```
0  T_sample 4.83  T_update 5.43  T_eval 3.30  exec_R 0.80  exec_R_eps 803.02  pusher
[WALL] total=12.9s
```

| 項目 | 時間 |
|------|------|
| T_sample（サンプリング） | 4.83秒 |
| T_update（Stackelberg 勾配更新） | 5.43秒 |
| T_eval（評価サンプリング） | 3.30秒 |
| プロセス全体 wall-clock | **12.9秒** |

`min_batch_size=5000` では約 **130秒/エポック** の見込み。

T_sample のボトルネックは URDF ロード（骨格変換ステップごとに発生、約 7回/エピソード）。
GPU 利用率はネットワーク更新中に 5〜11%（小規模 Transformer のため低め）。
シミュレーション自体は CPU 専用（Choreonoid）。

```bash
# 1スレッドで学習（現在推奨）
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1 \
  hydra.run.dir=single_run/pusher_cnoid \
  enable_wandb=false

# 評価（チェックポイントから）
USE_CHOREONOID=1 choreonoid --no-window --python \
  scripts/eval_cnoid_numerical.py -- \
  --restore_dir single_run/pusher_cnoid
```

---

## 既知の制限と今後の課題

| 項目 | 状態 |
|------|------|
| シングルスレッド学習 | ✅ 動作確認済み（12.9s/epoch で 500steps）|
| マルチスレッド学習 (`num_threads>1`) | 🔜 spawn + 永続ワーカー方式で実装予定 |
| `fork()` ベースのマルチプロセス | ❌ Qt / CUDA が fork 後に壊れる |
| 正常終了 (`B.App.exit()`) | ⚠️ ハング気味、`Ctrl+C` 推奨 |
| gym → gymnasium 移行 | ⚠️ NumPy 2.0 警告あり（動作に支障なし）|
| URDF キャッシュ | 🔜 骨格変換ごとの再ロードを削減できる余地あり |
