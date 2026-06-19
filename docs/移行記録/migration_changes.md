# MuJoCo+conda → Choreonoid+Docker 移行: 変更箇所まとめ

元の StackelbergPPO リポジトリに対して何をどう変えたかを、ファイル単位でまとめる。

---


## 新規作成ファイル

### `khrylib/rl/envs/common/mujoco_env_choreonoid.py`（全面書き直し）

`mujoco_env_gym.py` と**完全に同じ API** を持つ drop-in replacement。
旧版は ZMQ クライアントだったが、現在は `cnoid` バインディングを直接使う。

```
mujoco_xml_to_body()       ← MuJoCo XML → Choreonoid .body YAML 変換（ネイティブ形式）
                              ※当初は mujoco_xml_to_urdf() を使っていたが .body 形式に移行
ChoreonoidSimWorld          ← 旧 cnoid_sim_server.py から移設
ChoreonoidEnv               ← ZMQ 削除、ChoreonoidSimWorld を直接呼ぶように変更
_ModelProxy, _DataProxy     ← 変更なし（透過補完レイヤー）
```

`ChoreonoidEnv` の主要メソッドの対応:

| メソッド | 旧（ZMQ） | 新（直接） |
|---------|----------|-----------|
| `reload_sim_model()` | `_send({'cmd': 'reload_model'})` | `self._world.load_model()` |
| `reset()` | `_send({'cmd': 'reset'})` | `self._world.reset()` |
| `do_simulation()` | `_send({'cmd': 'step'})` | `self._world.step()` |
| `set_state()` | `_send({'cmd': 'set_state'})` | `self._world.set_state_cmd()` |

削除したもの: `ZMQ`, `reconnect()`, `sys.path` 操作, `irsl_choreonoid` 依存

### `scripts/choreonoid_train.py`（新規）

`choreonoid --no-window --python` に渡すエントリポイント。

```python
# 使い方
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher num_threads=1
```

### `design_opt/conf/__init__.py`（新規）

Hydra 1.3 が `config_path` をPythonモジュールとして解決するために必要。

### `scripts/worker_pool.py`（新規）

マルチスレッド学習用の永続ワーカープール。`choreonoid --no-window --python scripts/worker_sampler.py` をスポーンし、Pipe 経由でポリシー重みとサンプル結果をやり取りする。

### `scripts/smoke_test_cnoid.py`（新規）

Choreonoid 環境のヘルスチェックスクリプト。学習前の事前確認用。3エピソード完走・NaN/Inf なし・exec_reward > 0 をチェック。

```bash
USE_CHOREONOID=1 /choreonoid_ws/install/bin/choreonoid --no-window \
  --python scripts/smoke_test_cnoid.py cfg=pusher xml_name=rrbot_arm
```

---

## 既存ファイルへの変更

### `design_opt/envs/pusher.py` ほか全 env ファイル

**変更1**: `USE_CHOREONOID` 環境変数スイッチ（旧来から）

```python
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
```

**変更2**: NumPy 2.0 非互換修正（全 env ファイル共通）

```python
# 修正前（NumPy 2.0 で ValueError）
ctrl[aind] = body_a

# 修正後
ctrl[aind] = body_a.item()
```

対象: pusher / gap / hopper / ant / walker / swimmer / stair / stairhard

**変更3 (pusher.py 固有)**: 固定根本アーム（`rrbot_arm`）対応

- `is_fixed_base` プロパティ追加（ルートに free joint がなければ True）
- `get_sim_obs()` i==0 分岐: 固定根本時は `zeros(11)` でパディング（free joint state なし）
- 接触報酬・観測の基準点: 固定根本では `robot.bodies[-1].name`（末端ボディ）を使用
- `model.jnt_dofadr` フォールバック: Choreonoid `_ModelProxy` は `jnt_dofadr` 非対応のため `jnt_qposadr` で代替
- `reset_model()` を `reset_robot()` + `reset_state()` に分離（reload 責務の明確化）

---

### `design_opt/agents/genesis_agent.py`

**変更1**: `wandb` を条件付き import に変更（NumPy 2.0 非互換対応）

```python
try:
    import wandb
except Exception:
    wandb = None
```

**変更2**: `reconnect()` 呼び出し削除（ZMQ ワーカー接続コードを削除）

```python
# 削除したコード
if os.environ.get('USE_CHOREONOID', '0') == '1' \
        and hasattr(self.env, 'reconnect'):
    self.env.reconnect(5556 + pid)
```

---

### `design_opt/train.py`

**変更1**: `wandb` を条件付き import に変更

```python
try:
    import wandb
except Exception:
    wandb = None
```

**変更2**: Hydra `version_base` を `"1.2"` に固定

```python
@hydra.main(version_base="1.2", config_path="conf", config_name="config")
```

Hydra 1.3.2 は `version_base="1.2"` を受け付け、1.2 互換の挙動を維持する。

**変更3**: 学習ループ終了後に `env.close()` を追加

```python
agent.logger.info('training done!')
if hasattr(agent, 'env') and hasattr(agent.env, 'close'):
    agent.env.close()
```

サンプリング後にシミュレーションが放置されたままプロセスが終了しない問題の修正。
`close()` はサンプリングループ内では呼ばない（Qt シグナル連鎖を壊すため）。

---

### `scripts/cnoid_transfer.py`

ZMQ サーバー管理コードを全削除。`run_train()` が `choreonoid --no-window --python` を呼ぶように変更。

```python
# 変更前
proc, cf_path = start_choreonoid(args.server_script)
run_train(cfg, overrides, use_choreonoid=True)
stop_choreonoid(proc, cf_path)

# 変更後
run_train(cfg, overrides)  # 内部で choreonoid --no-window --python を subprocess 実行
```

---

### `scripts/eval_cnoid_numerical.py` / `eval_cnoid_visual.py`

ZMQ サーバー起動コード（`start_cnoid_server` import、`_eval_proc` 管理）を全削除。
`choreonoid --no-window --python` で直接実行するスクリプトに変更。

---

## 変更しなかったもの

- `khrylib/robot/xml_robot.py` — MuJoCo XML 生成ロジック。Choreonoid でもそのまま使う
- `design_opt/envs/pusher.py` の `PusherEnv` クラス本体 — `ctrl[aind]` 修正のみ
- `design_opt/agents/genesis_agent.py` の学習ロジック本体
- `design_opt/models/` — ネットワーク定義
- `design_opt/train.py` の学習ループ本体
- `khrylib/rl/envs/common/mujoco_env_gym.py` — MuJoCo パスとして残存

---

## 依存パッケージの変更

| パッケージ | 旧 | 新 |
|-----------|----|----|
| Python | 3.9 | 3.12 |
| Choreonoidバインディング | Python 3.8 専用 | Python 3.12 用に再ビルド |
| PyTorch | 2.0.1 | 2.7.0+cu128 |
| hydra-core | 1.2.0 | 1.3.2 |
| MuJoCo バイナリ | 2.1.0（必要） | 不要 |
| ZMQ | 必要 | 不要 |

---

## 学習の起動方法

```bash
# Choreonoid バックエンドで学習（現在の方法）
USE_CHOREONOID=1 OMP_NUM_THREADS=1 \
  choreonoid --no-window --python scripts/choreonoid_train.py \
  cfg=pusher \
  num_threads=1 \
  hydra.run.dir=single_run/pusher_cnoid \
  enable_wandb=false

# MuJoCo バックエンドで学習（元の方法、MuJoCo が入っている環境のみ）
OMP_NUM_THREADS=1 python -m design_opt.train cfg=pusher
```

**num_threads=1 の理由**: Choreonoid の `AISTSimulatorItem` は Qt オブジェクトを含むため、
Python の `fork()` で子プロセスを作ると Qt の内部状態が壊れる。
現状はシングルスレッド運用のみ安定動作。
