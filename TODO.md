# TODO

## メッシュセグメンテーション → StackelbergPPO 接続

### スケール正規化スクリプト（未実装）
**優先度**: 低（エンドツーエンド接続を実装するタイミングでセット）

topology.json を StackelbergPPO の初期形態として食わせる際、Tripo3D 等の生成メッシュのスケールを環境スケールに合わせる必要がある。

やること:
- `cfg` から リンク長の min/max 範囲を読み、その中点を target_link_length として使う
- AABB 対角長で正規化: `mesh.apply_scale(TARGET_DIAG / diag)`
- 姿勢正規化（主軸→Z）とセットで前処理化する
- pitch パラメータも target_diag から自動計算（`pitch = target_diag * 0.01` 程度）

参考スケール（`assets/mujoco_envs/*.xml` 調べ）:
- ant: 胴 r=0.25m、脚 0.57m → AABB 対角 ~1.2m
- hopper: リンク 0.4m + 0.45m → 高さ ~1.25m
- UR3: リーチ 0.47m（StackelbergPPO と同スケール）

実装場所の候補: `khrylib/utils/mesh_preprocess.py`（新規）または `mine/compare_*.py` に `--target-diag` オプション追加

---

## その他

- [ ] Dockerfile に `pip install mujoco` 追加（次回コンテナビルド時）
- [ ] Choreonoid v3 best checkpoint（epoch ~791）の eval_cnoid_numerical.py 実行
