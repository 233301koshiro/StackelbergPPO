# MuJoCo 3.x 移行記録

## 背景

元のコードは `mujoco-py`（OpenAI, 2022年開発終了）を使用していた。  
Python 3.12 / NumPy 2.0 環境では以下の理由でビルド不可:

- `longintrepr.h` が `cpython/` サブディレクトリに移動（Python 3.12）
- `PyArray_Descr.subarray` メンバー削除（NumPy 2.0）
- `PyCode_New` シグネチャ変更（Python 3.12）

新しい `mujoco` パッケージ（DeepMind, Python 3.12 対応, v3.9.0）に移行する。

---

## 変更ファイル一覧

### 1. `khrylib/rl/envs/common/mujoco_env_gym.py` — 全面書き換え

`mujoco_py.MjSim` API を互換ラッパーで再現。呼び出し側コードは無変更。

```python
# 互換ラッパー構成
class _ModelWrapper:   # mujoco.MjModel をラップ
    body_names         # tuple: body_id → name
    actuator_names     # tuple: actuator_id → name
    _body_name2id      # dict: name → body_id

class _DataWrapper:    # mujoco.MjData をラップ
    body_xpos          # プロパティ: data.xpos へのエイリアス
    get_body_xpos(name)
    get_body_xmat(name)

class _SimWrapper:     # mujoco_py.MjSim を再現
    reset()            # mujoco.mj_resetData
    forward()          # mujoco.mj_forward
    step()             # mujoco.mj_step
```

### 2. `assets/mujoco_envs/*.xml` — 座標系変換（9ファイル）

MuJoCo 3.0 で `coordinate="global"` が廃止されたため、全XMLをローカル座標系に変換。

**変換スクリプト**: `assets/mujoco_envs/convert_global_to_local.py`

変換内容:
- `compiler` から `coordinate="global"` と `inertiafromgeom="true"` を削除
- `body.pos`: ワールド座標 → 親ボディ相対座標（`local = global - parent_global`）
- `geom/joint/site.pos`: ワールド座標 → ボディ相対座標
- `geom.fromto` 両端点: ワールド座標 → ボディ相対座標

> **注意**: 全ボディが回転なし（euler/quat なし）のため、座標変換は平行移動のみ。

動作確認（mujoco 3.9.0）:
- ✅ ant, hopper, walker, climber, swimmer, pusher
- ❌ gap, stair, stair-hard（地形PNG ファイル欠如 — XML の問題ではない）

### 3. `khrylib/robot/xml_robot.py` — KeyError 修正

```python
# 修正前（変換後の XML に coordinate 属性がなく KeyError）
self.local_coord = compiler.attrib['coordinate'] == 'local'

# 修正後（属性なし = MuJoCo 3.x デフォルトの local 座標と解釈）
coord = compiler.attrib.get('coordinate', 'local') if compiler is not None else 'local'
self.local_coord = coord != 'global'
```

---

## インストール手順

```bash
pip install mujoco --break-system-packages
# → 3.9.0 がインストールされる（2026-06 時点）
```

> `mujoco 2.3.x` は Python 3.12 向けのビルド済みホイールが存在しないため使用不可。

---

## 並列サンプリング最適化

### 問題

従来の `multiprocessing.Process`（毎バッチ新規生成）では以下のオーバーヘッドが生じる:

1. **サブプロセス起動コスト**: 毎バッチ n_workers 個のプロセスを起動・Python 再インポート
2. **OpenMP スレッド競合**: MuJoCo 3.x は内部で OpenMP を多用。  
   `n_workers × OpenMP_threads` 個のスレッドが競合し CPU を圧迫
3. **CPU Policy 推論**: サブプロセスは GPU を使えず Transformer を CPU で計算

### 解決策: 永続ワーカープール

Choreonoid 版（`ChoreonoidWorkerPool`）と同じ設計を MuJoCo 用に実装。

| ファイル | 役割 |
|---|---|
| `scripts/mujoco_worker_sampler.py` | ワーカープロセス本体（Pipe でメインと通信） |
| `design_opt/utils/mujoco_worker_pool.py` | プール管理クラス |
| `design_opt/agents/genesis_agent.py` | USE_CHOREONOID に関係なく pool を使用 |

```bash
# 起動コマンド（OMP_NUM_THREADS は環境変数で強制）
OMP_NUM_THREADS=1 python3 -m design_opt.train cfg=pusher num_threads=20 enable_wandb=false
```

### 効果（pusher, 20 workers, min_batch_size=50000）

| 指標 | 改善前 | 改善後 |
|---|---|---|
| T_sample（初期） | ∞（詰まり） | 35s |
| T_sample（定常） | — | **7s** |
| T_update | 675s（epoch 0）| **133s**（定常） |
| 1 epoch 合計 | — | **~2.5 分** |
| Choreonoid 版比較 | — | **約 10 倍速** |

> T_update が epoch 0 の 675s から定常 133s に下がる理由:  
> Stackelberg 探索でロボット形態が収束し、ボディ数（Transformer のノード数）が減少するため。

---

## 注意事項

### `coordinate="global"` XML の動的生成

`Robot.export_xml_string()` は `local_coord=True` のとき、  
body の位置をローカル座標で出力する（`xml_robot.py` の `local_coord` フラグ参照）。  
XML 変換後の実行環境では常に `local_coord=True` となるため、  
動的生成 XML も MuJoCo 3.x に正しくロードされる。

### ドッカーファイルへの追記

次回コンテナビルド時に追加すること:

```dockerfile
# MuJoCo 3.x (Python 3.12 対応; mujoco-py は廃止)
RUN pip install mujoco
```
