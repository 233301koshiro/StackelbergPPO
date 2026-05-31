# StackelbergPPO × Choreonoid 移行作業レポート

## 1. 研究テーマと目的

設計を進化させていくロボット最適化システムのフェーズに、**Stackelberg PPO** を採用した。

### なぜ Stackelberg PPO か

通常のロボット形態・制御共設計（morphology–control co-design）は、以下の2段階で構成される。

1. **形態探索フェーズ**: ロボットの骨格・リンク形状を変化させる
2. **制御最適化フェーズ**: その形態に最適なコントローラーを学習する

当初参考にしようとした **Transform2Act**（ICRL 2022）も同じ構成を持つ。しかし **Stackelberg PPO**（ICLR 2026）はここにゲーム理論の「シュタッケルベルク戦略」を組み込んでいる点が新しい。

**シュタッケルベルク戦略とは**: リーダーが先に行動し、その行動に対するフォロワーの反応を「ある程度予測したうえで」行動を決める非対称ゲーム。形態設計をリーダー、制御をフォロワーに対応させることで、**「この設計にするとコントローラーがどのくらいのパフォーマンスを出すか」を報酬設計に織り込む**ことができる。これが従来手法との最大の差異。

ネットワークには Transformer、学習手法には強化学習の PPO（Proximal Policy Optimization）を使用。元リポジトリは Transform2Act のコードを強く参照しており、比較検討の結果こちらを採用した。

### 移行の動機

元リポジトリは **conda + MuJoCo**（シミュレーション）で動作する。  
垣内研究室の思想は **Docker + Choreonoid** であるため、システムをそちらに移行する必要がある。

---

## 2. 開発環境の構築

### 問題の構図

| 項目 | 元リポジトリ | 研究室標準 |
|------|------------|----------|
| パッケージ管理 | conda | Docker |
| シミュレータ | MuJoCo 2.1.0 | Choreonoid |
| Python | 3.9 | 3.8（Choreonoid バインディング固定）|
| GPU | 非 Blackwell 想定 | RTX 5060 Ti（sm_120） |

研究室提供の Docker イメージ `irsl_system` は Choreonoid 入りだが Python 3.8 ベース。  
PPO が要求する PyTorch 2.0.1 は RTX 5060 Ti（sm_120, Blackwell アーキテクチャ）に非対応であり、PyTorch 2.3 以降は Python 3.8 を非サポートという二重の制約があった。

### 解決策: `akita_sp` Docker イメージ

`irsl_system` を継承する個人用 Dockerfile（`Dockerfile.add_akita_sp`）を作成し以下を追加:

1. **Python 3.9** を追加インストール（Ubuntu 20.04 標準リポジトリ）
2. **PyTorch 2.7.0+cu128**（RTX 5060 Ti の sm_120 に対応）
3. **MuJoCo 2.1.0** を `/ros_home/.mujoco/` にインストール
4. **PPO 依存ライブラリ**（gym==0.15.4, mujoco-py, torch-geometric 等）を Python 3.9 で導入
5. **xvfb**（ヘッドレス起動用）
6. **lxml + pyzmq**（Python 3.8 側でもサーバー通信用）

### VS Code Dev Container の活用

研究室 Choreonoid は Jupyter 上で動作するが、Jupyter のターミナルでは **GitHub Copilot / Claude Code が使えない**という問題があった。  
これに対して **VS Code の Dev Container 拡張機能**を使用することで解決。Dev Container は「Docker コンテナを VS Code のフル機能が使える開発環境として利用するための拡張機能」であり、`akita_sp` コンテナに VS Code でアクセスし Claude Code を動かすことに成功した。

---

## 3. MuJoCo → Choreonoid バックエンドの設計

### Python バージョン分離の問題

Choreonoid の Python バインディング（`cnoid.Base`, `cnoid.BodyPlugin` 等）は **Python 3.8 専用**でコンパイルされており、PPO 側の Python 3.9 から直接呼び出せない。

### 解決策: ZeroMQ による 2 プロセス構成

```
PPO プロセス (Python 3.9)           Choreonoid プロセス (Python 3.8)
┌──────────────────────────┐        ┌──────────────────────────────┐
│  mujoco_env_choreonoid   │←─ZMQ──│  cnoid_sim_server.py         │
│  (gym.Env クライアント)    │        │  (AISTSimulator + ZMQ サーバー)│
└──────────────────────────┘        └──────────────────────────────┘
```

- `khrylib/rl/envs/common/mujoco_env_choreonoid.py`（Python 3.9）: `MujocoEnv` と**同じ API**を持つ差し替えクラス `ChoreonoidEnv`。ZMQ REQ ソケットでサーバーに `load_model / reset / step / set_state` を送信。`_ModelProxy`, `_DataProxy` で既存の env コード（`self.model.nq`, `self.data.qpos` 等）を透過的に補完。
- `khrylib/rl/envs/common/cnoid_sim_server.py`（Python 3.8）: Choreonoid 内で動く ZMQ REP サーバー。
- `design_opt/envs/pusher.py`に 2 行追加するだけで切り替え可能:

```python
import os
if os.environ.get('USE_CHOREONOID', '0') == '1':
    from khrylib.rl.envs.common.mujoco_env_choreonoid import ChoreonoidEnv as MujocoEnv
else:
    from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
```

### MuJoCo XML → URDF 変換

元リポジトリでは、モデルが出力したパラメータを **MuJoCo XML** 形式に変換してロボットを生成している。Choreonoid は MuJoCo XML を直接読めないため、`cnoid_sim_server.py` 内の `mujoco_xml_to_urdf()` 関数で URDF に変換してからロードする。

URDF に対応していない MuJoCo 固有パラメータと補完方法:

| パラメータ | URDF | 対処法 |
|-----------|------|--------|
| armature（関節慣性）| なし | Choreonoid API `setEquivalentRotorInertia()` で事後設定 |
| gear ratio | なし | step 時に `ctrl × gear` をトルクとして手動計算 |
| integrator=RK4 | なし | Choreonoid は semi-implicit Euler 固定（差 ~1.5%）|
| solimp/solref（接触柔性）| なし | 渡せない（残差の主因）|
| condim/friction（摩擦）| なし | 渡せない |

---

## 4. バグ修正（計 5 件）

PPO 学習を動かすと報酬値が異常（10^30 オーダー）になる問題が発生。MuJoCo と Choreonoid の数値を詳細比較して以下を修正。

1. **角速度の誤取得**: 根リンク角速度を `root.dv`（線形加速度）で取得していた → `root.w`（角速度）に修正。`qvel[5] = -9.807`（重力加速度）が角速度として混入し報酬が崩壊していた。
2. **Armature（関節慣性）の欠落**: armature がないと慣性が約200倍小さくなり関節角が爆発（11.18 rad → 本来 0.057 rad）。URDF ロード後に `setEquivalentRotorInertia()` で補完することで 0.099 rad まで改善。
3. **cube の 2 本目スライド関節の欠落**: `body_el.find('joint')` が最初の 1 本しか取得しないバグ。`findall` に変更して仮想リンクで連結する処理を追加。MuJoCo の nq=13 に合致。
4. **collision 要素の欠落**: URDF に `<visual>` しかなく `<collision>` がないと Choreonoid が接触計算しない。全ジオメトリタイプ（capsule/sphere/box）に `<collision>` を追加。
5. **capsule 慣性公式の誤り**: 半球の重心は平面から `3r/8` 内側にあるが、誤った公式を使用していた。修正後 MuJoCo との誤差 1.6% 以内。

### 残差について

全修正後も MuJoCo と Choreonoid の関節速度に **~32% の差**が残った。原因は接触モデルの根本的な違い（solimp/solref, condim/friction が URDF で表現できない）であり、pusher の初期配置で肢が cube に食い込んでいる部分の処理がシミュレータ間で異なる。これはシミュレータ固有の問題であり、**Choreonoid 上で再学習することで回避可能**。

---

## 5. Choreonoid 上での再学習

MuJoCo と Choreonoid で学習済み重みを直接転用するのは困難なため、以下の方針を採る。

### 追加した設定フラグ

```yaml
reset_epoch: false    # true でエポックカウンタを 0 にリセット（MuJoCo 重みを使いつつ再学習）
reset_obs_norm: false # true で obs_norm を引き継がず再学習（観測スケールが異なるため）
```

### 移行自動化スクリプト: `scripts/cnoid_transfer.py`

```
[Step 1] MuJoCo チェックポイントから形態重みだけ引き継ぎ、Choreonoid 上で再学習
[Step 2] Choreonoid 報酬 / MuJoCo 報酬の比率が閾値（デフォルト 0.5）を超えるか確認
[Step 3] 閾値未満なら --auto-scratch でスクラッチ再学習に自動移行
```

---

## 6. Choreonoid 起動方式の改善（本日の作業）

### 問題

元の起動方法（`xvfb-run choreonoid --python cnoid_sim_server.py`）は研究室の標準的な Choreonoid 利用方法ではなかった。研究室では Choreonoid を **Jupyter カーネルとして起動**（`choreonoid --jupyter-connection {connection_file}`）して使用する。

### 調査と実装

Jupyter カーネル一覧を確認したところ、`choreonoid` / `choreonoid_ros` カーネルが存在。`/user_scripts/jupyter_process.sh` というラッパースクリプトが `irsl_entryrc` を source してから Choreonoid を起動する仕組みになっていた。

この仕組みを使って `scripts/start_cnoid_server.py` を実装:

```
jupyter_process.sh choreonoid {接続ファイル}
    ↓（Python 3.8 カーネルとして起動）
cnoid_sim_server.py のコードをカーネル内で実行
    ↓
ZMQ サーバーがポート 5556 でリッスン開始
```

### デバッグの経緯（本日）

1. `jupyter_client.write_connection_file()` が **空の HMAC キーを生成**していた → xeus-python（Choreonoid が内部で使うカーネル実装）は空キーでは動かない。UUID を手動生成することで解決。
2. `kc.wait_for_ready(timeout=30)` がタイムアウト → ハートビートを **ZMQ DEALER ソケット**で直接ポーリングする方式に変更（REQ ソケットはタイムアウト後に再送不可になるため）。
3. Choreonoid は `xeus-python 0.17.4`（C++ 実装の Jupyter カーネル）を内部で使用していることが判明。プロトコルは 5.3 で互換性あり。
4. `run_server()` の無限ループがカーネルを占有するため、ZMQ サーバーが起動したかどうかを直接ポート疎通で確認するように実装。

現在: **Choreonoid ZMQ サーバーの起動は成功**（ハートビート確認、ZMQ ping/pong 確認済み）。

### 学習実行の問題（未解決）

学習コマンドに `USE_CHOREONOID=1` を付け忘れたため mujoco_py のパス不一致でクラッシュ → 修正済み。

---

## 7. 学習実行のデバッグ（2026-05-31）

Choreonoidサーバーの起動・接続・学習実行を通じて複数の問題が発生し、以下の順番で修正した。

### 問題1: mujoco_py のトップレベル import

`USE_CHOREONOID=1` にもかかわらず、`design_opt/envs/__init__.py` が全 env ファイルを一括 import し、そこから `mujoco_env_gym.py` が `import mujoco_py` を実行してクラッシュ。

**修正**: `mujoco_env_gym.py`、`mjviewer.py`、全 env ファイル（hopper, swimmer, walker, ant, pusher 等）の mujoco_py import を `try/except Exception` でラップし、インスタンス化時のみ失敗する遅延 import に変更。

### 問題2: xeus-python が空 HMAC キーで動かない

`jupyter_client.write_connection_file()` が `"key": ""` の接続ファイルを生成し、Choreonoid 内の xeus-python カーネルがメッセージを無視していた。

**修正**: 接続ファイルを手動生成（UUID4 キー）する `_write_connection_file()` 関数を実装。

### 問題3: 旧 Choreonoid プロセスがポート 5556 を占有

テスト中に起動した古い Choreonoid プロセス（PID 36864）が SIGTERM を無視して生き続け、ポート 5556 を保持していた。新しいサーバーが起動してもポートが取れず、学習プロセスが旧サーバーに接続したまま。

**修正**: SIGKILL で強制終了。start_cnoid_server.py 起動後は `ss -tlnp` でポート確認を推奨。

### 問題4: ZMQ REQ ソケットのフォーク安全性

`multiprocessing.Process` でワーカーをフォークすると、親の ZMQ REQ ソケットが子プロセスに引き継がれる。複数ワーカーが同一の REQ ソケットを使うと：
- REQ の状態機械（send→recv→send...）が競合してデッドロック
- `multiprocessing.Lock()` で直列化しようとしたが、今度は親の Lock が壊れた状態で引き継がれる場合がありハング

**現在の対処**: `num_threads=1`（シングルワーカー）で学習を実行。マルチスレッド化には Choreonoid サーバーを各ワーカーに 1 つずつ用意するアーキテクチャ変更が必要。

### 問題5: min_batch_size=50000 が単スレッドでは非現実的

ZMQ 経由の Choreonoid は 1 ステップ約 0.8ms（純通信）だが、neural network 推論込みの env.step() は約 50ms。
```
50000 ステップ × 50ms = 2500秒 ≈ 42分/エポック
42分 × 1000エポック = 29日
```

**対処**: `min_batch_size=5000` に変更（約 4分/エポック × 1000エポック ≈ 3日）。

### 問題6: Choreonoid コンソールログが /dev/null に捨てられていた

`start_cnoid_server.py` で `stdout=DEVNULL, stderr=DEVNULL` としていたため、Choreonoid 側の print・エラーが全て消えていた。

**修正**: `/tmp/cnoid_console.log` に追記するよう変更（次回起動から有効）。

---

### 現在の学習状態（2026-05-31）

```
学習プロセス: PID 2213407（実行中）
設定: cfg=pusher, num_threads=1, min_batch_size=5000, eval_batch_size=2000
Choreonoidサーバー: scripts/start_cnoid_server.py 経由で起動（Jupyter カーネル方式）
ログ: single_run/pusher_cnoid/log/log_train.txt（追記）
     /tmp/cnoid_console.log（Choreonoid 側、次回起動から）
```

---

## 8. ファイル構成（追加・変更ファイル）

```
StackelbergPPO/
├── khrylib/rl/envs/common/
│   ├── mujoco_env_choreonoid.py   # 新規: ChoreonoidEnv (Python 3.9 クライアント)
│   ├── mujoco_env_gym.py          # 変更: mujoco_py/MjViewer を遅延 import に変更
│   ├── mjviewer.py                # 変更: mujoco_py 系 import を遅延化
│   └── cnoid_sim_server.py        # 新規: ZMQ サーバー (Python 3.8, Choreonoid 内)
├── design_opt/
│   ├── envs/pusher.py             # 変更: USE_CHOREONOID 切り替え + mujoco_py 遅延化
│   ├── envs/{hopper,swimmer,walker,ant,stair,stairhard}.py  # 変更: mujoco_py 遅延化
│   ├── train.py                   # 変更: reset_epoch 対応
│   ├── agents/genesis_agent.py    # 変更: reset_obs_norm 対応
│   └── conf/config.yaml           # 変更: reset_epoch, reset_obs_norm フラグ追加
├── scripts/
│   ├── cnoid_transfer.py          # 新規: MuJoCo→Choreonoid 移行自動化（Jupyter カーネル方式に更新）
│   └── start_cnoid_server.py      # 新規: Jupyter カーネル経由でサーバー起動
└── docs/
    └── choreonoid_migration.md    # 新規: 移行作業詳細ドキュメント
```
