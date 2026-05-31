# MuJoCo+conda → Choreonoid+Docker 移行: 変更箇所まとめ

元の StackelbergPPO リポジトリに対して何をどう変えたかを、ファイル単位でまとめる。

---

## 新規作成ファイル

### `khrylib/rl/envs/common/mujoco_env_choreonoid.py`

MuJoCo の `MujocoEnv` と**完全に同じ API** を持つ差し替えクラス `ChoreonoidEnv`。
既存の env コード（pusher.py 等）は一切変更しなくてよい。

```
ChoreonoidEnv.__init__()
  → ZMQ REQ ソケットで localhost:5556 に接続
  → load_model コマンドを送って MuJoCo XML をサーバーに渡す
  → 応答から nq/nv/ctrlrange 等のモデル情報を取得

_ModelProxy, _DataProxy
  → 既存コードが self.model.nq や self.data.qpos にアクセスする部分を透過補完

step() / reset() / set_state()
  → ZMQ で対応コマンドを送り、物理演算はサーバー側（Choreonoid）が担う
```

### `khrylib/rl/envs/common/cnoid_sim_server.py`

Choreonoid の Python 3.8 カーネル内で動く ZMQ REP サーバー。

```python
# 起動方法（scripts/start_cnoid_server.py が自動処理）
jupyter_process.sh choreonoid {connection_file}
# → Choreonoid が起動し、このスクリプトがカーネル内で実行される

# 提供コマンド
load_model  → MuJoCo XML を URDF に変換して Choreonoid にロード
reset       → シミュレーションをリセット、初期状態を返す
step        → 1ステップ進める（ctrl × gear でトルク適用）
set_state   → qpos/qvel を直接セット
ping        → 疎通確認
```

**MuJoCo XML → URDF 変換**（`mujoco_xml_to_urdf()`）で行っていること:
- capsule/sphere/box ジオメトリの `<visual>` + `<collision>` 要素生成
- hinge/slide/free 関節の変換
- 複数関節ボディ（cube の x/y スライド等）→ 仮想リンクで連結
- capsule の慣性テンソルを正しい公式で計算

URDF では表現できない MuJoCo 固有パラメータの補完:
- `armature` → ロード後に `joint.setEquivalentRotorInertia()` で設定
- `gear` → step 時に `ctrl × gear` をトルクとして手動スケール

### `scripts/start_cnoid_server.py`

Choreonoid ZMQ サーバーを起動するランチャー。
`xvfb-run choreonoid --python` の代わりに、研究室標準の**Jupyter カーネル方式**で起動する。

```bash
# 使い方
python3.9 scripts/start_cnoid_server.py

# 内部でやっていること
1. UUID4 キーを持つ Jupyter 接続ファイルを手動生成
   （jupyter_client.write_connection_file() は空キーを返すため使えない）
2. jupyter_process.sh choreonoid {接続ファイル} を起動
   （irsl_entryrc を source → choreonoid --jupyter-connection）
3. ZMQ DEALER ソケットでハートビートをポーリング（60s タイムアウト）
4. cnoid_sim_server.py のコードを Jupyter カーネル内で execute
5. ZMQ ping/pong でサーバー疎通確認
6. Choreonoid の stdout/stderr を /tmp/cnoid_console.log に記録
```

### `scripts/cnoid_transfer.py`

MuJoCo で学習済みのチェックポイントを Choreonoid に移行するスクリプト。

```bash
# 使い方
python3.9 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher
python3.9 scripts/cnoid_transfer.py --mujoco-dir single_run/pusher --auto-scratch

# 処理フロー
Step 1: MuJoCo チェックポイントから形態重みだけ引き継いで Choreonoid で再学習
        (morph_prior=true, reset_epoch=true, reset_obs_norm=true)
Step 2: Choreonoid報酬 / MuJoCo報酬の比率を確認（閾値デフォルト 0.5）
Step 3: 閾値未満なら --auto-scratch でスクラッチ再学習に移行
```

---

## 既存ファイルへの変更

### `design_opt/envs/pusher.py` — 2行追加（+環境変数スイッチ）

```python
# 変更前
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv

# 変更後（追加 4行）
import os
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
```

これだけで `USE_CHOREONOID=1` の環境変数で切り替わる。クラス名は `MujocoEnv` のまま残すので、`PusherEnv` の中身は一切変更不要。

---

### `design_opt/train.py` — `load_epoch` と `start_epoch` の分離

```python
# 変更前（1行）
start_epoch = int(FLAGS.epoch) if ... else FLAGS.epoch

# 変更後（7行）
load_epoch = int(FLAGS.epoch) if ... else FLAGS.epoch   # どのファイルを読むか

if getattr(FLAGS, 'reset_epoch', False):
    start_epoch = 0          # チェックポイントを読むが、エポック0から再学習
elif isinstance(load_epoch, int):
    start_epoch = load_epoch # 通常の継続学習
else:
    start_epoch = 0

agent = BodyGenAgent(..., checkpoint=load_epoch)  # ← load_epoch で読む
for epoch in range(start_epoch, ...):             # ← start_epoch から回す
```

MuJoCo 学習済みモデルを読み込みつつ、Choreonoid でエポック0から再学習するために必要。

---

### `design_opt/agents/genesis_agent.py` — obs_norm リセット条件を1箇所修正

```python
# 変更前
if model_cp['obs_norm'] is not None and cfg.uni_obs_norm and not cfg.morph_prior:

# 変更後（条件を1つ追加）
if model_cp['obs_norm'] is not None and cfg.uni_obs_norm \
        and not cfg.morph_prior and not cfg.reset_obs_norm:
```

MuJoCo と Choreonoid では観測値のスケールが異なる（接触ダイナミクスの違い）ため、移行時に観測正規化の統計を捨てる。

---

### `design_opt/conf/config.yaml` — フラグ2行追加

```yaml
# 追加
reset_epoch: false    # true にするとチェックポイントをロードしてもエポック0から再学習
reset_obs_norm: false # true にすると obs_norm を引き継がず再学習
```

---

### `design_opt/utils/config.py` — フラグ2行追加

```python
# 追加（Config クラスの __init__ 内）
self.reset_epoch = FLAG.get('reset_epoch', False)
self.reset_obs_norm = FLAG.get('reset_obs_norm', False)
```

---

### `khrylib/rl/envs/common/mujoco_env_gym.py` — mujoco_py を遅延 import

```python
# 変更前
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("...")  # ← import 時点でクラッシュ

from khrylib.rl.envs.common.mjviewer import MjViewer

# 変更後
try:
    import mujoco_py
except Exception:
    mujoco_py = None  # ← import 時はクラッシュせず、使う時だけ失敗

try:
    from khrylib.rl.envs.common.mjviewer import MjViewer
except Exception:
    MjViewer = None
```

**なぜ必要か**: `design_opt/envs/__init__.py` が hopper/swimmer/walker/ant 等を全て一括 import し、それらが `mujoco_env_gym` を import する。`USE_CHOREONOID=1` でも pusher 以外の env ファイルが読まれるため、mujoco_py が見つからないと起動時点でクラッシュしていた。

---

### `khrylib/rl/envs/common/mjviewer.py` — mujoco_py 系を遅延 import

```python
# 変更前
import glfw
from mujoco_py.builder import cymj
from mujoco_py.generated import const
from mujoco_py.utils import rec_copy, rec_assign

# 変更後
try:
    import glfw
    from mujoco_py.builder import cymj
    from mujoco_py.generated import const
    from mujoco_py.utils import rec_copy, rec_assign
    _mujoco_available = True
except Exception:
    glfw = cymj = const = rec_copy = rec_assign = None
    _mujoco_available = False
```

---

### `design_opt/envs/{hopper,swimmer,walker,ant,stair,stairhard}.py` — mujoco_py 遅延 import

各ファイルで同じ修正（実際には mujoco_py を使っていないが、import 文だけがあった）:

```python
# 変更前
import mujoco_py

# 変更後
try:
    import mujoco_py
except Exception:
    mujoco_py = None
```

---

## 学習の起動方法

```bash
# 1. Choreonoid ZMQ サーバーを起動（初回・再起動時）
python3.9 scripts/start_cnoid_server.py &

# 2. 学習開始
USE_CHOREONOID=1 OMP_NUM_THREADS=1 python3.9 -m design_opt.train \
    cfg=pusher \
    hydra.run.dir="single_run/pusher_cnoid" \
    enable_wandb=false \
    num_threads=1 \
    min_batch_size=5000 \
    eval_batch_size=2000
```

**num_threads=1 の理由**: ZMQ REP サーバーは1シミュレーション世界を持ち、複数ワーカーから同時アクセスすると状態が混在する。フォーク後のREQソケット共有はデッドロックを引き起こすため、現状はシングルスレッド運用。マルチスレッド化にはワーカーごとにChoreonoidサーバーを1つ用意する構成変更が必要。

**min_batch_size=5000 の理由**: ZMQ 経由の Choreonoid は 1 env.step() あたり約 50ms（neural network 推論込み）。`50000 × 50ms ≈ 42分/エポック` で非現実的なため、`5000 × 50ms ≈ 4分/エポック`（1000エポックで約3日）に変更。

---

## 変更しなかった（変える必要がなかった）もの

- `khrylib/robot/xml_robot.py` — MuJoCo XML 生成ロジック。Choreonoid でもそのまま使う
- `design_opt/envs/pusher.py` の本体（`PusherEnv` クラス）— 一切変更不要
- `design_opt/agents/genesis_agent.py` の学習ロジック本体 — 一切変更不要
- `design_opt/models/` — ネットワーク定義 — 一切変更不要
- 他の env ファイル（hopper 等）のクラス本体 — mujoco_py import 部分のみ変更
