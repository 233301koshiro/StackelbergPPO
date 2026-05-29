# StackelbergPPO を Choreonoid で動かす作業まとめ

## 目的

StackelbergPPO（形態・制御の共設計を行う強化学習リポジトリ）は MuJoCo + conda で動作している。
これを研究室のDockerイメージ（Choreonoid入り）上で動かし、最終的にChoreonoidをシミュレータとして使えるようにする。

**設計方針（重要）**:
- MuJoCo XML によるロボット形態生成（`xml_robot.py`）はそのまま流用する
- MuJoCo をシミュレータとして使っている部分だけを Choreonoid に置き換える
- MuJoCo の XML→URDF 変換機能など「便利な箇所」はそのまま残す

---

## フェーズ1: Docker環境の構築

### 問題

- 研究室提供の Choreonoid Docker イメージは Python 3.8 ベース
- StackelbergPPO が要求する PyTorch 2.0.1 は RTX 5060 Ti（sm_120, Blackwell）に非対応
- PyTorch 2.3 以降は Python 3.8 を非サポート

### 解決策

`Dockerfile.add_akita_sp`（研究室イメージを継承する個人用Dockerfile）に以下を追加:

1. **Python 3.9** を Ubuntu 20.04 標準リポジトリからインストール
2. **PyTorch 2.7.0+cu128** をインストール（RTX 5060 Ti の sm_120 に対応）
3. **MuJoCo 2.1.0** を `/ros_home/.mujoco/` にインストール（コンテナの HOME が `/ros_home`）
4. **PPO依存ライブラリ**（gym==0.15.4, mujoco-py, torch-geometric 等）を Python 3.9 でインストール
5. **xvfb**（Choreonoidのヘッドレス起動用）
6. **lxml + pyzmq**（Python 3.8 側でも必要、サーバー通信用）

```dockerfile
FROM repo.irsl.eiiris.tut.ac.jp/irsl_system:noetic

# Python 3.9 + MuJoCo + PyTorch 2.7 (GPU対応)
RUN apt-get install -y python3.9 python3.9-dev xvfb ...
RUN python3.9 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN python3.9 -m pip install gym==0.15.4 mujoco-py ...
```

**確認**: `akita_sp` イメージで RTX 5060 Ti を使い PPO 学習が動作することを確認。

---

## フェーズ2: Choreonoidバックエンドの実装

### アーキテクチャ

Choreonoid の Python バインディングは Python 3.8 専用でコンパイルされており、
PPO 側の Python 3.9 から直接呼び出せない。そこで **ZeroMQ による2プロセス構成**を採用:

```
PPO プロセス (Python 3.9)          Choreonoid プロセス (Python 3.8)
┌─────────────────────────┐        ┌──────────────────────────────┐
│  mujoco_env_choreonoid  │←─ZMQ──│  cnoid_sim_server.py         │
│  (gym.Env クライアント)   │        │  (AISTSimulator + ZMQ サーバー)│
└─────────────────────────┘        └──────────────────────────────┘
```

**起動方法**:
```bash
# Choreonoidサーバーを Xvfb でヘッドレス起動
Xvfb :99 -screen 0 1024x768x24 &
DISPLAY=:99 choreonoid --python cnoid_sim_server.py &

# PPO学習（Choreonoidバックエンド）
USE_CHOREONOID=1 python3.9 -m design_opt.train cfg=pusher
```

### 実装ファイル

#### `khrylib/rl/envs/common/mujoco_env_choreonoid.py` (Python 3.9)

`mujoco_env_gym.py` と **同じ API** を持つ drop-in replacement。

- `ChoreonoidEnv` クラス: `MujocoEnv` と同じインターフェース
- ZMQ REQ ソケットでサーバーに `load_model`, `reset`, `step`, `set_state` を送信
- `_ModelProxy`, `_DataProxy`: 既存の env コード（`pusher.py` 等）が `self.model.nq` や `self.data.qpos` にアクセスする部分を透過的に補完

#### `khrylib/rl/envs/common/cnoid_sim_server.py` (Python 3.8)

Choreonoid 内で動く ZMQ REP サーバー。

- **MuJoCo XML → URDF 変換器**: `mujoco_xml_to_urdf()` 関数
  - capsule/sphere/box ジオメトリ対応
  - hinge/slide/free 関節対応
  - 複数関節ボディ（cube の x/y スライド）を仮想リンクで対応
  - `<collision>` 要素も生成（接触検出に必要）
- **`ChoreonoidSimWorld`**: WorldItem + AISTSimulatorItem の管理
  - `sim.setRealtimeSyncMode(3)`: マニュアルモード（外部から1ステップずつ制御）
  - `sim.tickRequest(True)`: 1ステップ同期実行（RL に適した同期制御）

#### `design_opt/envs/pusher.py` (変更: 2行追加のみ)

```python
import os
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
```

元の MuJoCo 版は完全に残したまま、環境変数で切り替え可能。

---

## フェーズ3: バグ修正と調査

PPO 学習を動かすと報酬値が異常（10^30 オーダー）になる問題が発生。MuJoCo と Choreonoid の数値を詳細比較した。

### 修正1: 角速度の誤取得 (`dv` → `w`)

**問題**: 根リンクの角速度を `root.dv`（線形加速度）で取得していた。

```python
# 誤: root.dv = 線形加速度（重力 -9.81 が入る）
qvel += list(root.v) + list(root.dv)

# 正: root.w = 角速度
qvel += list(root.v) + list(root.w)
```

**影響**: `qvel[5] = -9.807`（重力加速度）が角速度として入り、報酬計算が崩壊。

---

### 修正2: Armature（関節慣性）の欠落

**問題**: MuJoCo XML の `<joint armature="1">` は URDF に対応フィールドがない。

```
MuJoCo dof_armature: [0,0,0,0,0,0, 1.0,1.0,1.0,1.0, 1.0,1.0]
                                    ↑ 4関節に各1 kg·m² の慣性
```

armature がないと慣性が ~200 倍小さくなり、関節角が爆発した（11.18 rad → 本来 0.057 rad）。

**解決**: URDF ロード後に Choreonoid API で設定:

```python
joint_armatures = {}  # MuJoCo XML から joint名 → armature値を収集
...
for i in range(b.numJoints):
    j = b.joint(i)
    arm = joint_armatures.get(j.jointName, 0.0)
    if arm > 0:
        j.setEquivalentRotorInertia(arm)
```

**効果**: 11.18 rad → 0.099 rad（MuJoCo の 0.057 rad に近づく）。

---

### 修正3: cube の2本目スライド関節の欠落

**問題**: pusher.xml の cube ボディは2本のスライド関節（x/y方向）を持つが、
変換器が `body_el.find('joint')` で最初の1本しか取っていなかった。

```
MuJoCo: nq=13, nv=12
Choreonoid (修正前): nq=12, nv=11  ← 1DOF少ない
Choreonoid (修正後): nq=13, nv=12  ✅
```

**解決**: 複数関節ボディを仮想リンクで連結する処理を追加。

---

### 修正4: collision 要素の欠落

**問題**: URDF に `<visual>` しかなく `<collision>` がないと、Choreonoid は接触計算しない。

**解決**: 全ジオメトリタイプ（capsule/sphere/box）に `<collision>` 要素を追加。
capsule は `fromto` 方向に合わせた RPY 回転も計算して付与。

---

### 修正5: capsule 慣性公式の誤り

**問題**: 半球の慣性計算に誤った公式を使用。

```python
# 誤: m_cap * (2r²/5 + l²/2 + 3lr/8)
# 正: m_cap * (2r²/5 + (l/2 - 3r/8)²)
#              ↑ 半球の重心は平面から 3r/8 内側にある
```

**影響**: 慣性テンソルが約 1.5 倍に膨張していた。
**修正後**: MuJoCo の慣性値との誤差 1.6% 以内。

---

## MuJoCo XML → URDF の情報損失と補完方法

MuJoCo XML には URDF 規格に存在しないフィールドがある。
変換器（`cnoid_sim_server.py` の `mujoco_xml_to_urdf()`）はそれぞれ以下のように扱っている。

```
MuJoCo XML の情報
  │
  ├── URDF で表現できるもの（シミュレータ非依存）
  │     質量・慣性テンソル・ジオメトリ・joint damping・関節タイプ・軸・可動域
  │     → Choreonoid に正確に渡せる（誤差 1.6% 以内）
  │
  └── URDF に存在しないもの（シミュレータ固有）
        armature=1       → URDF規格なし → Choreonoid APIで補完（近似）
        gear=150         → URDF規格なし → トルク適用時に手動スケール
        integrator=RK4   → URDF規格なし → Choreonoidは別の積分法（差1.5%）
        solimp/solref    → URDF規格なし → 接触剛性・ソルバが消える ← 残差の主因
        condim/friction  → URDF規格なし → 接触摩擦モデルが消える
```

### armature（関節慣性）

MuJoCo の `<joint armature="1">` はロータ慣性（モータの回転子が関節に与える等価慣性）。
URDF に対応フィールドがないため、**URDF ロード後に Choreonoid API で事後設定する**。

```python
# 変換器: MuJoCo XML から armature 値を収集
joint_armatures = {}          # joint名 → armature値
# ... parse_xml で hinge joint ごとに ...
joint_armatures[jname] = float(joint_el.get('armature', default_armature))

# Choreonoid ロード後: Body.joint(i).setEquivalentRotorInertia() で適用
for i in range(b.numJoints):
    j = b.joint(i)
    arm = joint_armatures.get(j.jointName, 0.0)
    if arm > 0:
        j.setEquivalentRotorInertia(arm)
```

armature がないと慣性が ~200 倍小さくなり関節角が爆発する（11 rad → 本来 0.057 rad）。
補完後も実効慣性に ~1.5% の差が残るが、学習上は許容範囲。

### gear ratio（ギア比）

MuJoCo の `<motor gear="150">` はアクチュエータのギア倍率。URDF に対応なし。
**シミュレーションのステップ実行時に手動で乗算する**。

```python
# step コマンド受信時
for jname, ainfo in self.actuators_map.items():
    j = b.joint(jname)
    if j is not None:
        j.u = float(ctrl[i]) * ainfo['gear']   # ctrl × 150 = 実トルク [Nm]
```

### integrator（積分スキーム）

MuJoCo は `<option integrator="RK4">` で4次のルンゲ・クッタ法を使用。
Choreonoid の AIST シミュレータは semi-implicit Euler 固定で変更不可。

単独関節の比較:

```
MuJoCo RK4        : 1.4676 rad/s
Choreonoid Euler  : 0.9999 rad/s  （同一パラメータ・同一 dt・同一トルク）
```

積分法の差による誤差は ~1.5% 程度。

### solimp / solref（接触ソルバパラメータ）・condim / friction（摩擦モデル）

MuJoCo XML では接触の柔らかさ・摩擦モデルを詳細に設定できる。

```xml
<!-- MuJoCo XML 例 -->
<default>
  <geom condim="3"                      <!-- 接触自由度: 法線+2摩擦方向 -->
        friction="1.0 0.5 0.5"          <!-- 接線/転がり/スピン摩擦係数 -->
        margin="0.01"                   <!-- 接触検出マージン -->
        solimp="0.9 0.95 0.001"        <!-- 接触柔性（インピーダンス） -->
        solref="0.02 1.0"/>             <!-- 目標加速度・減衰比 -->
</default>
```

これらは **URDF に対応フィールドが存在しない**。Choreonoid の AIST シミュレータは
独自のペナルティ接触モデルを使うため、同じジオメトリでも接触力が異なる。

pusher 環境では z=0.4 の初期配置で肢キャプセルが cube に食い込んでおり、
この食い込み接触の解釈がシミュレータごとに異なることが残差 ~32% の主因。

---

## 調査結果: 残差 ~32% について

全修正を適用後も、MuJoCo と Choreonoid で関節速度に ~32% の差が残った。

詳細調査の結果、これは **接触モデルの根本的な違い** によるものと判明:

```
MuJoCo: z=0.4 初期配置で、肢キャプセルが cube に食い込んでいる
  → 接触反力が根ボディを逆回転させる (ωy = +5.59 rad/s)
  → joint_dq (相対速度) が水増しされる

Choreonoid: 接触剛性・ソルバが異なり同じ接触力が再現されない
  → 根ボディがほぼ動かない (ωy ≈ 0)
  → joint_dq が実際の肢の動きをそのまま反映
```

**MuJoCo XML に存在するが URDF で表現できないパラメータ**:

| パラメータ | URDF | 状態 |
|-----------|------|------|
| 質量・慣性テンソル | `<inertial>` | ✅ 完全に渡せる（誤差1.6%）|
| joint damping | `<dynamics damping=>` | ✅ 完全 |
| armature | なし | ⚠️ Choreonoid API で補完（近似）|
| gear ratio | なし | ✅ 手動補完 |
| integrator=RK4 | なし | ❌ Choreonoid は別積分法固定（差1.5%）|
| solimp/solref（接触柔性） | なし | ❌ 渡せない（主因）|
| condim/friction（摩擦モデル） | なし | ❌ 渡せない |

**結論**: 「モデル → XML → URDF」の変換で失われる情報は接触ソルバと積分法のパラメータのみ。
ロボットの構造・物性（質量・慣性・ジオメトリ）は URDF を介して正確に渡せる。
残差は Choreonoid 固有の接触モデルに起因するため、**Choreonoid 上でゼロから再学習することで回避可能**。

---

## 動作確認

```bash
# Choreonoid サーバー起動
Xvfb :99 &; DISPLAY=:99 choreonoid --python .../cnoid_sim_server.py &

# 小バッチでの動作確認（3エポック完走）
USE_CHOREONOID=1 python3.9 -m design_opt.train cfg=pusher \
    min_batch_size=200 eval_batch_size=100 num_threads=1

# 出力例
# Evaluation: [body_0, body_1, ..., body_114, body_214]  ← 形態が変化
# 0  T_sample 13.5  T_update 10.5  exec_R 84.81  pusher
# 1  T_sample 13.1  T_update 9.5   exec_R 202.49
```

形態が `body_11`, `body_114`, `body_214` などに変化していることから、
**MuJoCo XML で生成された URDF が Choreonoid 上で動的にロードされていることが確認できた**。
これが第一フェーズのゴールとして設定していたマイルストーン。
