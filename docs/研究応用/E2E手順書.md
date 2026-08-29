# 手描きスケッチ E2E 実行手順書

> 第6章の未達 (g)「手描き線画を入力とする end-to-end 実走」を閉じるための手順。
> **新規実装は不要。** M1・M2 が Web UI 経由である点は限界として残す（未達 (h) は別問題）。
> 所要: 描く 10分 + Web 2回 5分 + 自動処理 1分 + 判定 200ep。

## 描くもの（3パターン推奨）

診断機能の実演を兼ねるため、**通る例と通らない例を混ぜる**こと。

| # | 形 | 狙い | 期待される第1層の出力 |
|---|---|---|---|
| A | 普通の3関節アーム | 正常系 | ✅ 可動範囲の内側 |
| B | **極端に短い**アーム | 不適合の検出 | ❌ 「腕全体を N 倍に伸ばしてください」 |
| C | 先端だけ極端に長い | 配分の影響 | ✅ だが co-design のスコアが落ちる（9-9 と同傾向） |

紙でもタブレットでもよい。**白背景・輪郭がはっきりしていれば画質は問わない**（後段で Gemini が描き直すため）。

## 手順

### 1. Gemini で整形（Web UI）

[付録A_プロンプト全文.md](修論ドラフト/付録A_プロンプト全文.md) のプロンプトを**そのまま貼り**、スケッチ画像を添付する。
関節数が3以外なら、プロンプト内の「3関節」とマーカー位置の列挙（1.〜3.）を実際の数に直す。

**確認**: 出力画像に **純マゼンタの球が関節数ぶん**あるか目視。無ければ再生成（ここで妥協しない）。

### 2. Tripo3D で3D化（Web UI）

整形画像をアップロードし、GLB をダウンロード → `data/<名前>/<名前>.glb` に置く。

### 3. GLB → XML（自動・一発）

```bash
bash scripts/run_tripo_pipeline.sh data/<名前>/<名前>.glb data/<名前> <名前>
```

**詰まったら**: マーカー色は生成過程でずれる（純マゼンタが保存されない）。既定は自動キャリブレーションだが、
検出0なら `JOINT_TOL=60 bash scripts/...` のように許容幅を広げる。それでも駄目なら
`JOINT_COLOR="230 40 220"` のように実測色を指定する（GIMP 等でスポイト）。

**確認**: `assets/mujoco_envs/<名前>.xml` が生成され、`data/<名前>/meshes/link_*.stl` がリンク数ぶんあること。

### 4. 幾何診断（自動・約1秒）

```bash
docker exec b38ea459f886 bash -lc \
  'cd /userdir/StackelbergPPO && python3 scripts/diagnose_morphology.py --xml <名前> --task reach'
```

ここで B が ❌ を返せば**診断機能の実演が撮れる**。返した倍率でスケールした XML を作れば閉ループも再現できる:

```bash
python3 scripts/make_scaled_arm.py --base <名前> --scale <返された倍率> --name <名前>_fix
```

### 5. co-design 判定（200ep）

```bash
docker exec b38ea459f886 bash -lc 'cd /userdir/StackelbergPPO && \
  nohup env USE_CHOREONOID=1 OMP_NUM_THREADS=1 /choreonoid_ws/install/bin/choreonoid --no-window \
  --python scripts/choreonoid_train.py cfg=pusher_gearonly xml_name=<名前> \
  num_threads=4 max_epoch_num=200 enable_wandb=false fix_skeleton=true seed=0 \
  +robot_param_scale=1 +reward_specs.use_reach=true +reward_specs.target_x=0.8 \
  +reward_specs.target_y=0.0 +reward_specs.target_z=0.15 \
  +reward_specs.ctrl_cost_coeff=0.2 +env_specs.check_init_contact=false \
  hydra.run.dir=single_run/e2e_<名前> > single_run/e2e_<名前>/stdout.log 2>&1 &'
```

## 記録すること

通ったら **第6章の未達 (g) を「実証済み」へ移す**。あわせて:

- 各段の所要時間（(f) の「学習を要しない工程は数秒」の裏付けになる）
- 途中で失敗した回数（マーカー検出の歩留まりは軸1 の頑健性の実データ）
- 3パターンの第1層出力と co-design スコア → 実験系譜に新しい段として追記

⚠️ **失敗も記録する。** 何回描き直したかは「非専門家が使えるか」の唯一の実データになる（利用者評価は未実施なので）。
