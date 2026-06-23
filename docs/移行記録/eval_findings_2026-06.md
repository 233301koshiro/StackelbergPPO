# 評価・物理エンジン移行 — 調査所見（2026-06）

---

## バグ修正1: `detect_engine` の誤判定

### 問題

`eval_cross_env.py` の `detect_engine()` は `.hydra/config.yaml` の `cfg` 名で判定していた。

```python
cfg_name = str(cfg.get('cfg', ''))
return 'choreonoid' if 'cnoid' in cfg_name.lower() else 'mujoco'
```

`rrbot_arm_cnoid` は `cfg: pusher`（`cnoid` なし）なので **mujoco と誤判定**され、Choreonoid ポリシーが MuJoCo 環境で評価されていた。

### 修正

`stdout.log` の冒頭 1024 バイトに Choreonoid 固有ログが含まれるかで補足判定するよう変更。

```python
for log_name in ('stdout.log', 'train.log'):
    log_path = os.path.join(restore_dir, log_name)
    if os.path.exists(log_path):
        with open(log_path, 'r', errors='ignore') as f:
            header = f.read(1024)
        if 'cnoid' in header.lower() or 'choreonoid' in header.lower():
            return 'choreonoid'
```

---

## バグ修正2: Choreonoid サブプロセス起動

### 問題

`USE_CHOREONOID=1 python3 eval_cross_env.py` では `cnoid` モジュールの import 時に **SIGSEGV** でクラッシュ。

さらに `choreonoid --no-window --python script.py --arg1 val1` とすると、Choreonoid 自身が `--arg1` を引数として解析してエラーになる。

### 修正

引数を環境変数 `_EVAL_*` で渡し、スクリプト内モジュールレベルで読み込む方式に変更。

```python
# 起動側
env['_EVAL_RESTORE_DIR'] = restore_dir
env['_EVAL_TMPOUT']      = tmp_out
cmd = ['choreonoid', '--no-window', '--python', __file__]

# スクリプト内（モジュールレベル）
if os.environ.get('_EVAL_RESTORE_DIR'):
    run_worker(os.environ['_EVAL_RESTORE_DIR'], ...)
    os._exit(0)  # sys.exit は Choreonoid で TypeError になる
```

---

## バグ修正3: eval 成功判定（Choreonoid の exit code 問題）

Choreonoid サブプロセスは正常終了でも exit code が非ゼロになる場合がある。  
returncode チェックを廃止し、**JSON 出力ファイルの存在**で成功/失敗を判定するよう変更。

```python
if engine == 'choreonoid':
    if not os.path.exists(tmp_out):
        raise RuntimeError(f"Worker failed (no output) for {restore_dir}")
```

---

## バグ修正4: Choreonoid cube slide joint の damping 欠落 ⚠️ 最重要

### 発見の経緯

Choreonoid GUI ビューア（`eval_cnoid_viewer.py`）で可視化したところ、cube が **137m** 飛んでいることを確認。アーム全長 0.55m に対して 250 倍の移動距離は明らかに異常。

### 根本原因

`mujoco_env_choreonoid.py` の `mujoco_xml_to_body()` が MuJoCo XML の slide joint の `damping` 属性を読んでいたが、`.body` YAML に書き出していなかった。

```python
# 修正前（damping を捨てていた）
elif jtype_mj in ('slide', 'prismatic'):
    axis = parse_vec(j_el.get('axis', '1 0 0'))
    rng_raw = [float(x) for x in j_el.get('range', '-10 10').split()]
    add_link(child_name, parent_name, 'prismatic', jname,
             axis, rng_raw, translation, mass, com, inr, shape)

# 修正後
elif jtype_mj in ('slide', 'prismatic'):
    axis    = parse_vec(j_el.get('axis', '1 0 0'))
    rng_raw = [float(x) for x in j_el.get('range', '-10 10').split()]
    damping = float(j_el.get('damping', default_damping))
    add_link(child_name, parent_name, 'prismatic', jname,
             axis, rng_raw, translation, mass, com, inr, shape,
             damping=damping)
```

`serialize_links_to_yaml` にも `joint_damping` 出力を追加。

### 影響範囲

| | MuJoCo 学習 | Choreonoid 学習（v1） | Choreonoid 学習（v2）|
|---|---|---|---|
| cube damping | 2.0（XML がそのまま適用） | **0（欠落）** | 2.0（修正済み） |
| 学んだ戦略 | 押し続ける必要あり | **一発弾けばいい** | 正しく押す学習へ |

### なぜ「80% 成功」が出たか

学習・eval ともに damping なしで実行されており内部整合性はあった。  
cube が摩擦ゼロで滑るため、アームが接触した瞬間の衝撃力だけで +x 方向に高速滑走。  
`reward_fwd_cube = Δx/dt` なので飛ぶほど報酬大 → 「弾き飛ばし」が最適戦略になった。

**結論**: v1 の 80% 成功は物理バグの産物。`rrbot_arm_cnoid_v2`（damping 修正版）で再学習・再評価が必要。

### 同様の修正を `dynamic_body_updater.py` にも適用

`dynamic_body_updater.py` 経由で生成する cube.body にも `joint_damping: 2.0` を追加。  
（eval スクリプト系がこちらのパスを使う場合に対応。）

---

## 調査結果1: MuJoCo でキューブが −x 方向に移動

### 症状

全 10 エピソードで `cube_disp_m` が負（mean = −0.625m）。

### 根本原因

`fix_skeleton=false` で Leader が 3 リンクに骨格変換 + bone_offset 延長。  
execution 開始時に **アームの先端がキューブ内部に侵入**した状態になり、接触衝突の反発力でキューブが −x に弾け飛ぶ。

```
arm_tip:    x = 0.849 m
cube 左面:  x = 0.751 m   ← arm_tip が cube 内部
cube center: x = 0.901 m  → 開始直後に 0.426 m へ急落 (−0.475m)
```

`best.p` が epoch 93（学習初期）というのも Follower 未収束の証拠。

### 対応方針

現フェーズは MuJoCo を一旦捨て Choreonoid に注力する方針のため未対処。  
（`fix_skeleton=true` で骨格固定すれば再現しない）

---

## 調査結果2: Choreonoid exec_R スケール差（v1）

Choreonoid 学習中の exec_R_eps が MuJoCo（最高 813）に対して 26979 と約 33 倍高い。  
主因は **cube damping 欠落**（上記バグ）。damping ありで再学習すれば MuJoCo と同程度のスケールになるはず。

---

## 調査結果3: ep 8 数値発散 (NaN)

Choreonoid の特定エピソードで `exec_R=NaN`、中途終了する事象が発生。  
Choreonoid 物理ソルバーが特定の状態で数値発散する。

### 対策（実施済み）

`run_worker` の統計計算で NaN エピソードを除外。

```python
rewards_finite = [r for r in rewards if np.isfinite(r)]
mean_exec_reward = float(np.mean(rewards_finite)) if rewards_finite else float('nan')
```

---

## 現在地と方針

| フェーズ | 状態 |
|---|---|
| eval インフラ整備 | ✅ 完了（detect_engine・サブプロセス・NaN除外） |
| v1 Choreonoid 学習 | ❌ 無効（damping バグ） |
| damping バグ修正 | ✅ 完了（mujoco_env_choreonoid.py + dynamic_body_updater.py） |
| v2 Choreonoid 再学習 | 🔄 進行中（`single_run/rrbot_arm_cnoid_v2`） |
| v2 eval | ⏳ v2 完走後に実施 |
