# StackelbergPPO × Choreonoid 移行作業レポート v2

> v1 (`report.md`) は ZMQ 構成時代の記録。本書は Python 3.12 直接呼び出し構成への移行後の最終状態をまとめる。

---

## 1. 研究テーマと目的

### Stackelberg PPO とは

ロボットの**形態（骨格・リンク長・関節）**と**制御（動き方）**を同時に最適化する強化学習システム。

通常の共設計は「形を決めてから動きを学ぶ」の繰り返しだが、**Stackelberg PPO**（ICLR 2026）はゲーム理論の「シュタッケルベルク戦略」を導入し、**設計時点でコントローラーの反応を先読みして形を選ぶ**。

```
リーダー（形態設計）: 「この形にするとコントローラーはどう動くか」を予測して設計
フォロワー（制御）  : 与えられた形で最適に動く
```

ネットワークは Transformer、学習アルゴリズムは PPO。元リポジトリは Transform2Act をベースとする。

### 移行の動機

| 項目 | 元リポジトリ | 目標（垣内研標準） |
|------|------------|-----------------|
| パッケージ管理 | conda | Docker |
| シミュレータ | MuJoCo 2.1.0 | Choreonoid |
| 実行環境 | ローカル Python | `akita_sp` Docker コンテナ |

---

## 2. 開発環境（`akita_sp` Docker イメージ）

### 構成

研究室標準の `irsl_system` イメージを継承する個人用 Dockerfile（`Dockerfile.add_akita_sp`）。

| 項目 | 内容 |
|------|------|
| ベースイメージ | `repo.irsl.eiiris.tut.ac.jp/irsl_system:noetic` |
| Python | **3.12**（Choreonoid バインディングを 3.12 用に再ビルド）|
| PyTorch | 2.7.0+cu128（RTX 5060 Ti / sm_120 対応）|
| Choreonoid | 2.3（Ubuntu 24.04 対応版）|
| MuJoCo | **不要**（Choreonoid に完全移行）|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| CPU | 20コア |

### 重要な変化：Python 3.12 で cnoid が直接使える

従来の研究室イメージは Python 3.8 ベースで Choreonoid バインディングも 3.8 専用だった。`akita_sp` では **Choreonoid を Python 3.12 用に再ビルド**したことで、ZMQ による 2 プロセス構成が不要になった。

```python
# Python 3.12 から直接使える
from cnoid.BodyPlugin import WorldItem, AISTSimulatorItem
import cnoid.IRSLUtil as IU
```

### 開発環境

VS Code Dev Container 拡張機能で `akita_sp` コンテナに接続し Claude Code を使用。

---

## 3. MuJoCo → Choreonoid バックエンドの設計

### アーキテクチャの変遷

#### 旧構成（ZMQ、v1 時代）

```
PPO プロセス (Python 3.9)     ZMQ     Choreonoid (Python 3.8)
┌───────────────────────┐   ←────→  ┌──────────────────────┐
│ mujoco_env_choreonoid │            │ cnoid_sim_server.py  │
│ (ZMQ クライアント)     │            │ (ZMQ サーバー)        │
└───────────────────────┘            └──────────────────────┘
```

ZMQ が必要だった理由：Choreonoid バインディングが Python 3.8 専用 → PPO の Python 3.9 と直接通信できなかった。

#### 現構成（直接呼び出し）

```
choreonoid --no-window --python scripts/choreonoid_train.py
┌──────────────────────────────────────────────────────────┐
│ Choreonoid プロセス (Python 3.12)                         │
│  ├─ Qt イベントループ（メインスレッド）                    │
│  ├─ 学習コード（ネットワーク更新・GPU）                    │
│  └─ WorkerPool 管理                                       │
│                                                           │
│ Worker 0 (別 choreonoid プロセス)  ←─ Pipe ─→ メイン    │
│ Worker 1 (別 choreonoid プロセス)  ←─ Pipe ─→           │
│ Worker N ...                                              │
└──────────────────────────────────────────────────────────┘
```

ZMQ・別言語プロセス・Jupyter カーネルが全て不要になった。

### MuJoCo XML → URDF 変換

元リポジトリはロボット形態を **MuJoCo XML** で生成する。Choreonoid は URDF を使うため、`mujoco_env_choreonoid.py` 内の `mujoco_xml_to_urdf()` で変換する。

URDF で表現できない MuJoCo 固有パラメータの補完方法:

| パラメータ | 対処法 |
|-----------|--------|
| armature（関節慣性）| ロード後に `joint.setEquivalentRotorInertia()` で設定 |
| gear ratio | `step()` 時に `ctrl × gear` で手動スケール |
| integrator=RK4 | Choreonoid は semi-implicit Euler 固定（差 ~1.5%）|
| solimp/solref（接触柔性）| 渡せない（残差 ~32% の主因）|
| condim/friction（摩擦）| 渡せない |

### 環境変数による切り替え

`USE_CHOREONOID=1` でバックエンドを切り替える。

```python
# design_opt/envs/pusher.py（他の env も同様）
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
```

---

## 4. バグ修正

### Choreonoid 固有バグ（5件）

報酬値が 10^30 オーダーになる問題の原因調査で発見。

1. **角速度の誤取得**: `root.dv`（線形加速度）→ `root.w`（角速度）に修正。`qvel[5] = -9.807` が混入し報酬崩壊。
2. **Armature（関節慣性）欠落**: URDF に対応フィールドなし。ロード後に API で補完。関節角が 11.18 rad → 0.099 rad に改善（MuJoCo の 0.057 rad に近似）。
3. **cube の 2 本目スライド関節欠落**: `body_el.find('joint')` が 1 本しか取らないバグ → `findall` + 仮想リンク連結で解決。nq=13 に一致。
4. **collision 要素欠落**: `<visual>` しかなく接触計算されなかった → 全ジオメトリに `<collision>` 追加。
5. **capsule 慣性公式の誤り**: 半球重心の位置（平面から `3r/8` 内側）を正しく計算。修正後 MuJoCo との誤差 1.6% 以内。

### Python 3.12 / NumPy 2.0 移行バグ（5件）

6. **`ctrl[aind] = body_a` → `.item()`**: shape `(1,)` 配列のスカラー代入が NumPy 2.0 で `ValueError`。全 env ファイル（8個）に適用。
7. **`wandb` 条件付き import**: NumPy 2.0 非対応 → `try/except` で包む。
8. **Hydra 1.2.0 → 1.3.2**: Python 3.12 の dataclasses 仕様変更で動かない → アップグレード。`design_opt/conf/__init__.py` を新規追加。
9. **`sim.setRealtimeSyncMode(3)` → `.NonRealtimeSync`**: Choreonoid 2.3 で enum が必要。
10. **サンプリング後のシミュレーション放置**: Python 終了後も Choreonoid がシミュを走らせ続けプロセスが数分終了しない。学習ループ末尾で `env.close()` → `os._exit(0)` に変更。

### 残差について

全修正後も MuJoCo と Choreonoid の関節速度に **~32% の差**が残る。
原因は接触モデルの根本的な違い（solimp/solref が URDF で表現不可）。
**Choreonoid 上でゼロから再学習することで回避可能**。

---

## 5. 並列サンプリング実装

### 問題

`fork()` でワーカープロセスを作ると Qt オブジェクトと CUDA コンテキストが破損する。

### 解決策: spawn + 永続ワーカー

```
メインプロセス（choreonoid --no-window）
  └─ GPU でネットワーク更新（Stackelberg 勾配）

Worker 0（choreonoid --no-window --python scripts/worker_sampler.py）
Worker 1（同上）  ← 学習開始時に起動、全エポックを通じて常駐
Worker 2（同上）
Worker 3（同上）
```

**通信プロトコル（Pipe + bytes）**

```
起動時:   main → worker: JSON bytes {cmd:init, cfg_yaml, project_path}
          worker → main: b'ready'

各エポック: main → worker: pickle bytes {cmd:sample, policy_state(numpy), batch_size}
           worker → main: pickle bytes {memory, logger}

終了時:   main → worker: pickle bytes {cmd:quit}
```

torch テンソルを numpy に変換してから pickle することで、subprocess 間の認証エラーを回避。

### スレッド数ベンチマーク結果

`min_batch_size=5000`, `eval_batch_size=1000`, RTX 5060 Ti 16GB, CPU 20コア

| スレッド | T_sample | T_update | T_eval | wall合計 | GPU avg | sample 倍速 |
|---------|----------|----------|--------|---------|---------|------------|
| 1 | 33.5s | 40.3s | 3.7s | 80s | 65% | 1.0x |
| 2 | 11.7s | 46.7s | 3.1s | 75s | 85% | 2.9x |
| **4** | **6.7s** | **35.2s** | **3.2s** | **69s** | **61%** | **5.0x** |
| 8 | 6.4s | 38.5s | 2.5s | 91s | 37% | 5.2x |

**結論: `num_threads=4` が最適**（wall 69s/epoch）。8スレッドはオーバーヘッドで逆に遅くなる。

---

## 6. 追加した設定フラグ

```yaml
# design_opt/conf/config.yaml
reset_epoch: false    # true でエポックカウンタを 0 にリセット（MuJoCo重み引き継ぎ再学習用）
reset_obs_norm: false # true で obs_norm を引き継がず再学習
save_model_interval: 10  # 10エポックごとにチェックポイント保存
```

---

## 7. ファイル構成（追加・変更・削除）

### 追加

```
scripts/
  choreonoid_train.py      ← choreonoid --no-window --python 用エントリポイント
  worker_sampler.py        ← 永続ワーカープロセスのサンプリングループ
design_opt/
  utils/worker_pool.py     ← ChoreonoidWorkerPool クラス
  conf/__init__.py         ← Hydra 1.3 のモジュール解決に必要
```

### 書き直し

```
khrylib/rl/envs/common/mujoco_env_choreonoid.py
  旧: ZMQ クライアント
  新: ChoreonoidSimWorld（シミュ世界）を内包して直接呼び出し
      mujoco_xml_to_urdf()、状態取得ヘルパーも統合
```

### 削除

```
khrylib/rl/envs/common/cnoid_sim_server.py  ← ZMQ サーバー（不要）
scripts/start_cnoid_server.py               ← Jupyter カーネル起動スクリプト（不要）
```

### 主な変更

```
design_opt/envs/pusher.py（他 7 env も同様）
  - ctrl[aind] = body_a.item()  ← NumPy 2.0 対応
design_opt/agents/genesis_agent.py
  - WorkerPool 統合、fork ベースのマルチプロセスを廃止
  - wandb 条件付き import
design_opt/train.py
  - Hydra version_base="1.2"、env.close()、os._exit(0)
design_opt/utils/config.py
  - _flags 保存（WorkerPool での OmegaConf 直列化用）
scripts/cnoid_transfer.py、eval_cnoid_numerical.py、eval_cnoid_visual.py
  - ZMQ サーバー管理コードを全削除
  - choreonoid --no-window --python 方式に変更
```

---

## 8. 学習の起動方法

```bash
# 4スレッドで学習（推奨設定）
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid \
  enable_wandb=false

# チェックポイントから再開
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=4 \
  hydra.run.dir=single_run/pusher_cnoid \
  +restore_dir=single_run/pusher_cnoid \
  enable_wandb=false

# 数値評価
USE_CHOREONOID=1 choreonoid --no-window --python \
  scripts/eval_cnoid_numerical.py -- \
  --restore_dir single_run/pusher_cnoid --num_episodes 10

# MuJoCo → Choreonoid 移行
python3 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher
```

チェックポイントは `single_run/pusher_cnoid/models/` に `epoch_XXXX.p`（10エポックごと）と `best.p`（最高報酬更新時）で保存される。

---

## 9. 現在の学習状態（2026-06-02）

```
設定: cfg=pusher, num_threads=4, min_batch_size=50000（デフォルト）
      eval_batch_size=10000
稼働中: PID 1213201（メイン）+ ワーカー 4個

1エポックあたりの所要時間:
  T_sample  ~49s   （4ワーカー並列シミュレーション）
  T_update  ~275s  （Stackelberg 勾配計算・GPU）
  T_eval    ~10s
  合計      ~330s ≈ 6分/エポック

報酬推移（exec_R_eps）:
  Epoch 0: 101.9
  Epoch 2: 254.4  ← 最高（best.p 保存）
  Epoch 8: 137.1  （現在も上下しながら序盤の探索中）

ETA: 残り ~3日20時間（1000エポックまで）
```

---

## 10. 既知の制限

| 項目 | 状態 |
|------|------|
| シングル/マルチスレッド学習 | ✅ 動作確認済み（4スレッド推奨）|
| `fork()` ベースのマルチプロセス | ❌ Qt / CUDA が壊れる → spawn 方式で代替 |
| 正常終了 | ⚠️ `B.App.exit()` がハング → `os._exit(0)` で強制終了 |
| gym → gymnasium 移行 | ⚠️ NumPy 2.0 警告あり（動作に支障なし）|
| URDF キャッシュ | 🔜 骨格変換ごとの再ロードを削減できる余地あり |
| MuJoCo との残差 | ⚠️ ~32%（接触モデルの違い）→ Choreonoid でゼロから学習で回避 |
