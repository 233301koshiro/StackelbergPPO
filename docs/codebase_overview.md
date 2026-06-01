# StackelbergPPO コードベース概要

元のリポジトリがどのファイルで構成され、どう連携して動くかをまとめる。

---

## 全体の流れ（1エポック）

```
train.py
  └─ BodyGenAgent.optimize(epoch)
        ├─ [1] サンプリング: sample() → sample_worker()
        │       └─ PusherEnv.step() を繰り返す（シミュレーション）
        │
        └─ [2] ネットワーク更新: optimize_policy()
                ├─ リーダー更新（形態ポリシー） ← Stackelberg 勾配
                ├─ フォロワー更新（制御ポリシー） ← 通常 PPO
                └─ バリュー更新
```

---

## ディレクトリ構成と役割

```
StackelbergPPO/
├── design_opt/          ← このプロジェクト固有のコード
│   ├── train.py         ← エントリーポイント
│   ├── agents/          ← 学習ループ全体を管理
│   ├── envs/            ← 各タスク環境（pusher, walker 等）
│   ├── models/          ← ポリシー・バリューネットワーク
│   └── utils/           ← 設定・Stackelberg勾配計算・ツール
│
├── khrylib/             ← 汎用 RL ライブラリ（Transform2Act から継承）
│   ├── rl/
│   │   ├── agents/      ← AgentPPO 基底クラス
│   │   ├── core/        ← Policy, Critic, TrajBatch, RunningNorm 等
│   │   └── envs/common/ ← MuJoCo/Choreonoid 環境ラッパー
│   ├── robot/
│   │   └── xml_robot.py ← ロボット形態の表現・操作・XML生成
│   └── utils/           ← 汎用ユーティリティ
│
└── assets/
    └── mujoco_envs/     ← 初期ロボット XML ファイル（pusher.xml 等）
```

---

## 1エポックの詳細フロー

### [1] サンプリングフェーズ

```
BodyGenAgent.sample()
  │
  ├─ 複数ワーカーを fork して並列サンプリング
  │
  └─ sample_worker() ×N
        │
        └─ エピソードループ:
              reset() → 骨格変換フェーズ開始
              │
              ├─ [骨格変換ステージ] skel_transform_nsteps=5 ステップ
              │    ポリシーが「ボディを追加/削除/維持」を選択
              │    Robot.add_body() / remove_body() でグラフを変更
              │    → reload_sim_model(xml_str) でシミュレータに反映
              │
              ├─ [属性変換ステージ] 1 ステップ
              │    ポリシーがリンク長・関節パラメータを調整
              │    → reload_sim_model(xml_str) で反映
              │
              └─ [実行ステージ] 最後まで
                   ポリシーがジョイントトルクを出力
                   env.step(ctrl) → シミュレーション1ステップ進む
                   報酬（タスク達成度）を取得
```

### [2] ネットワーク更新フェーズ（Stackelberg PPO）

```
optimize_policy()
  │
  ├─ サンプルを リーダーステップ / フォロワーステップ に分類
  │    リーダー = 骨格変換・属性変換の行動（形態設計）
  │    フォロワー = 実行フェーズの行動（コントローラー）
  │
  ├─ [フォロワー更新] 通常の PPO
  │    surrogate loss + clip → ∇θ₂ で更新
  │
  └─ [リーダー更新] Stackelberg 勾配
       surrogate loss から ∇θ₁J_surr を計算
       + フィッシャー情報行列を使った補正項 J_delta を追加
         （「このリーダー行動にフォロワーが反応したらどうなるか」を予測）
       → 共役勾配法（CG）で逆ヘッセアン×勾配を近似計算
       → θ₁ を更新
```

---

## 主要ファイルの役割

### `design_opt/train.py`
エントリーポイント。Hydra で設定を読み、`BodyGenAgent` を作って学習ループを回す。

### `design_opt/agents/genesis_agent.py`
学習全体を管理する `BodyGenAgent` クラス。`AgentPPO`（khrylib）を継承。
- `setup_env()` — 環境を作る
- `setup_policy()` / `setup_value()` — ネットワークを作る
- `sample()` — 並列サンプリング
- `optimize_policy()` — Stackelberg PPO 更新
- `log_optimize_policy()` — TensorBoard / wandb へのログ出力

### `design_opt/envs/pusher.py`
タスク環境。`MujocoEnv`（またはChoreonoidEnv）を継承。
- エピソードを3ステージ（骨格変換 → 属性変換 → 実行）に分けて管理
- ステージに応じてポリシーに渡す観測を切り替える
- `reload_sim_model()` — 形態変更をシミュレータに反映

### `khrylib/robot/xml_robot.py`
ロボット形態を**グラフ**として表現するクラス群。
- `Robot` — 全ボディの親クラス。bodies リストでツリー構造を保持
- `Body` — 1つのリンク（質量、慣性、ジオメトリ、子ボディ、関節）
- `Joint` — 関節パラメータ（軸方向、可動域、減衰）
- `export_xml_string()` — 現在の形態を MuJoCo XML として出力
- `add_body()` / `remove_body()` — ツリーに枝を追加/削除

### `design_opt/models/bodygen_policy.py`
ポリシーネットワーク `BodyGenPolicy`。
- **ボディグラフに Transformer を適用**（各ボディをトークンとして扱う）
- 3つの独立したヘッド:
  - `skel_transformer` + `skel_action_logits` → 骨格変換行動（add/remove/keep）
  - `attr_transformer` + `attr_action_head` → 属性変換行動（パラメータ調整量）
  - `control_transformer` + `control_action_head` → 実行行動（関節トルク）

### `design_opt/models/bodygen_critic.py`
バリューネットワーク `BodyGenValue`。同じく Transformer ベース。

### `design_opt/utils/stackelberg.py`
Stackelberg 勾配計算の核心部分。
- `fisher_vector_product_selfkl()` — フィッシャー情報行列とベクトルの積（F·v）を計算
- `conjugate_gradient()` — CG 法で F⁻¹b を近似
- `bilevel_leader_grad_correct()` — リーダーの勾配補正項 J_delta を計算

### `design_opt/utils/config.py`
Hydra の設定を Python オブジェクトに変換する `Config` クラス。
YAMLの値をすべてフィールドに格納し、各クラスがここから設定を読む。

### `khrylib/rl/agents/agent_ppo.py`
標準的な PPO の基底クラス。`BodyGenAgent` の親。
- `update_params()` — surrogate loss + clip の基本更新

### `khrylib/rl/core/trajbatch.py`
サンプリングしたトラジェクトリをバッチ化するクラス群。

---

## 観測ベクトルの構造

各ステップで env が返す観測は3種類の情報を連結したもの:

```
state = [attr_fixed | sim_obs | attr_design]
         ↑ボディの固定情報  ↑シミュレーション状態  ↑現在の設計パラメータ
         （親子関係、深さ等）（qpos, qvel, xpos等）  （リンク長、関節軸等）
```

**ポイント**: ボディ数が変わると `sim_obs` の次元が変わる。
そのため Transformer でボディを「可変長のトークン列」として処理する。

---

## 設定ファイルの流れ

```
design_opt/conf/config.yaml       ← デフォルト設定
design_opt/cfg/pusher.yml         ← 環境固有設定（報酬・ロボット構造等）
        ↓ Hydra でマージ
design_opt/utils/config.py        ← Config クラスに格納
        ↓
BodyGenAgent / PusherEnv / BodyGenPolicy が Config から読む
```

---

## 学習に関わるハイパーパラメータ（主要）

| パラメータ | 意味 | デフォルト |
|-----------|------|----------|
| `skel_transform_nsteps` | 1エピソードの骨格変換ステップ数 | 5 |
| `stack_follower_steps` | Stackelberg 更新で使うフォロワーステップ数 | 6 |
| `lamda` | フィッシャー行列の減衰係数（ダンピング）| 5 |
| `min_batch_size` | 1エポックのサンプル数 | 50000 |
| `num_optim_epoch` | 1エポックあたりのネットワーク更新回数 | 10 |
| `max_epoch_num` | 総エポック数 | 1000 |

---

## MuJoCo との接点（Choreonoid 移行前）

```
PusherEnv → MujocoEnv（mujoco_env_gym.py）
  └─ self.sim = mujoco_py.MjSim(model)   ← MuJoCo シミュレータ
  └─ self.data.qpos / self.model.nq 等   ← 直接 C API にアクセス

reload_sim_model(xml_str)
  └─ self.sim = mujoco_py.MjSim(load_model_from_xml(xml_str))
                ↑ 形態変更のたびに MuJoCo にリロード
```
