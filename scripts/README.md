# scripts/ 索引

62 本あり、**再利用するもの**と**一度きりで役目を終えたもの**が混在している。
新しく作業を始めるときは、まずこの表で「使ってよいか」を確認すること。

削除はしていない。一度きりの起動スクリプトも「その実験をどう起動したか」という
一次記録であり、実験系譜.md の記述を裏取りするのに使うため。

---

## 1. 常用（学習を回すたびに通る）

| スクリプト | 役割 |
|---|---|
| `choreonoid_train.py` | **Choreonoid 内で走る学習エントリポイント。** すべての学習がここを通る |
| `worker_sampler.py` / `mujoco_worker_sampler.py` | ロールアウト並列化のワーカー |
| `run_cnoid_train.sh` | 学習起動ラッパー。多くの起動スクリプトが内部で呼ぶ |

## 2. パイプライン（M1〜M5、スケッチ → 学習可能なモデル）

| スクリプト | 役割 |
|---|---|
| `run_tripo_pipeline.sh` | **GLB → XML を一発で通す。** 通常はこれを使う |
| `glb_to_links.py` | M3: 色検出で関節を見つけメッシュを分割 |
| `mesh_to_params.py` | M4: OBB からリンク長・カプセル半径を抽出 → topology.json |
| `topology_to_xml.py` | M5: topology.json → MuJoCo 互換 XML |
| `run_2axis_mvp.sh` | rrbot 用の一気通貫（topology.json → XML/body → 学習） |
| `eval_pipeline_robustness.py` | **M3 の頑健性評価**（第4章 4.2.2 = 軸1）。色ドリフトに対する許容幅を tolerance 掃引で測る。表 4.4・4.5 と 図 4.1 の出典 |
| `make_scaled_arm.py` | 既存アーム XML の**リンク長だけ**を倍率スケールした XML を作る。診断の助言（「N 倍に伸ばせ」）を機械的に適用するのに使う（第4章 4.4.1） |
| `check_mesh_interference.py` | 静止姿勢の凸包による自己干渉チェック（v2系のみ） |
| `save_morphology_urdf.py` | 収束形態を URDF として書き出す |

⚠️ **`dynamic_body_updater.py` は実質デッドコード。** `run_2axis_mvp.sh` から呼ばれてはいるが、
学習時の Choreonoid `.body` 生成は `khrylib/rl/envs/common/mujoco_env_choreonoid.py` が行う。
**描画・形態まわりを直すときにここを触っても効かない**（2026-07-31 に実際に間違えた）。

## 3. 評価・診断（再利用する）

| スクリプト | 役割 |
|---|---|
| `diagnose_morphology.py` | **形態診断 M7**（第3章 3.12）。第1層は学習不要・約1秒・Choreonoid 不要 |
| `rank_settle.py` | **判定所要エポックの較正**（第4章 4.3.4.2）。学習不要、log_train.txt を読むだけ |
| `audit_runs.sh` | 全 run の一次データ表。**md より先にこれを見る** |
| `check_docs_consistency.py` | **docs とログの突き合わせ。** 古い状態語・数値の食い違い・記録漏れを検出。値を更新したら必ず走らせる |
| `compare_morphology.py` | 複数 run の収束形態を並べて比較 |
| `boundary_compare.py` | 境界張り付きの条件間比較（matched epoch） |
| `eval_cnoid_numerical.py` | 数値で成功率・報酬を確認 |
| `eval_cnoid_visual.py` | 動画（mp4）で記録 |
| `eval_cnoid_viewer.py` | GUI でリアルタイム再生 |
| `eval_cross_env.py` | ネイティブ物理エンジンでの評価（サブプロセス分離） |
| `plot_rewards.py` | 学習曲線グラフ |
| `visualize_morph_changes.py` | 形態変化の可視化 |
| `generate_comparison_report.py` | 学習曲線 + eval を Markdown/PNG レポート化 |

使い方は [docs/リポジトリ説明/評価スクリプト.md](../docs/リポジトリ説明/評価スクリプト.md)。

## 4. 一度きりの調査（probe 系。結論は docs にあるので再実行は通常不要）

| スクリプト | 何を調べたか | 結論の記録先 |
|---|---|---|
| `probe_cube_trace.py` | cube の軌跡（damping confound の切り分け） | 実験系譜 第8段 |
| `probe_k1_trajectory.py` | K1 の形態推移（転用が効かない理由） | 第5章 5.3 |
| `probe_reach_convergence.py` | Reach の収束速度仮説 | 実験系譜 9-3（**反証された**） |
| `probe_reach_trajectory.py` / `probe_reach_multi_episode.py` | Reach の到達・保持挙動 | 第5章 5.1.2 |
| `probe_L0_intervention.py` | 運動学的に不活性な L0 の介入実験 | 第4章 4.3.6 |
| `probe_v3_contact_check.py` | v3 の初期接触判定（Bug 16 関連） | デバッグ戦記 Bug 16 |
| `probe_joint_axes.py` | 関節軸の平面性 | — |
| `collect_m_ablation_results.py` | M系 ablation の集計 | 第5章 5.4 |
| `analyze_reach_kinematics.py` / `check_strategy.py` / `eval_reach_hover.py` | 個別調査 | — |

## 5. 一度きりの起動スクリプト（履歴。**再利用しない**）

その時々の空きスロットに合わせて書かれており、run 名・エポック数が固定されている。
**新しい実験を回すときは流用せず、`launch_pj_1000.sh` のように意図をコメントに書いた新規スクリプトを作ること。**

`launch_pj_experiment.sh` `launch_pj_tripo_experiment.sh` `launch_M_ablation.sh`
`launch_curriculum_transfer.sh` `launch_next3_20260803.sh` `launch_nsteps_ns1.sh`
`launch_pjp_1000.sh` `launch_pj_1000.sh` `launch_pj_dist.sh` `launch_recourse.sh`
`auto_launch_next_transfer.sh` `auto_launch_queue2.sh` `auto_launch_v2b_reach.sh`
`auto_launch_pj_pusher_matrix.sh` `auto_launch_queue_20260804.sh`
`resume_after_reboot_20260731.sh`（**再開手順の参考としては今も有用**）

## 6. スケジューラ（現行は1つだけ）

| スクリプト | 状態 |
|---|---|
| `experiment_queue.sh` | ✅ **現行。** 完走マーカーで判定する版（Bug 19 対応済み） |
| `weekend_queue.sh` | ✅ **現行（2026-08-07 追加）。** 空きメモリを見て軸3 の補強実験を順に投入する無人運転用。メモリ・ディスクの下限と投入期限を持つ |
| `ns1_scheduler.sh` | ❌ 廃止。`experiment_queue.sh` に統合済み |
| `tp2_scheduler.sh` / `weekend_scheduler.sh` / `restart_ready_watcher.sh` / `m_s2_watcher.sh` | ❌ 役目を終えた |

⚠️ **ウォッチャーを書くときの注意（Bug 19）**: 完走待ちに
`pgrep -f "hydra.run.dir=single_run/<run>"` を使わないこと。**そのコマンドライン自身が
文字列を含むため、他のウォッチャーから学習プロセスと誤認される。**
完走マーカー（`All workers terminated`）で判定する。`experiment_queue.sh` の
`finished()` / `running()` が参考実装。

## 7. その他

| スクリプト | 役割 |
|---|---|
| `smoke_test_cnoid.py` | Choreonoid 接続のスモークテスト |
| `cnoid_transfer.py` | 転用まわりの補助 |
| `monitor_training.py` | 学習監視（現在は使っていない） |
| `plot_pj_comparison.py` | PJ実験の比較図 |
| `build_thesis_pdf.py` | **修論 PDF ビルド**（`docs/研究応用/修論ドラフト/` の全 md → xelatex で1冊に）。下記の注意を読んでから使う |
| `make_thesis_figures.py` | **修論の図を一次データから生成**（`figures/*.png`）。学習曲線は `log/log_train.txt` から読む。図の数値は手で直さず、原本を確認してこれを再実行する |
| `eval_morphology.py` | 形態評価（docs・コードのどちらからも参照なし。`compare_morphology.py` に役割が吸収されたとみられる） |

---

## build_thesis_pdf.py の注意（節番号の罠）

```bash
python3 scripts/build_thesis_pdf.py --out docs/修論ドラフト_YYYYMMDD.pdf
```

⚠️ **本スクリプトは md の節番号を捨て、LaTeX に振り直させる**（`strip_heading_number`）。
つまり **PDF の節番号は「章の中で何番目の `##` か」で決まり、md に書いた番号とは無関係**。

そのため **md 側で節を増減・移動したら、`CHAPTER_ORDER` の構成が正しいか必ず確認する**。
ずれると本文中の「3.12 節」のような参照が全部無効になる（2026-08-07 に実際に踏みかけた）。

`CHAPTER_ORDER` は `(ファイル名, unnumbered, opts)` の3要素:

| opts | 意味 |
|---|---|
| `{'merge': True}` | **章を起こさず前の章の続き**として出力（H1 を落とす） |
| `{'title': '...'}` | 章タイトルを md の H1 ではなくこれにする |

**第3章はこの merge を使っている。** 前提（`第3章前段_前提.md`）と提案手法（`第3章_提案手法.md`）は
案A（2026-08-06 決定）により**1つの第3章**で、前提が 3.1〜3.4、提案手法が 3.5〜3.13 を占める。
2ファイルを別章にすると提案手法の節が 3.1 から振り直されて壊れる。

**ビルド後の検証手順**:

```bash
pdftotext -f 1 -l 5 docs/修論ドラフト_YYYYMMDD.pdf - | grep -E "^\s*3\.[0-9]+"
```

で目次を出し、**md の節番号と一致するか**を見る。とくに他章から参照されている節
（`grep -rn "3\.12" docs/`）が合っているかを確認する。

---

## 新しくスクリプトを足すときの約束

1. **先頭に日付と目的を書く。** 「何を確かめたくて作ったか」が分かれば、後から再利用可否を判断できる
2. **一度きりなら §5 に、再利用するなら §3 に追記する**
3. 学習を起動するなら**二重起動ガード**を入れる（`experiment_queue.sh` の各段が参考）
4. ログ解析なら **`log/log_train.txt` を読む**。`stdout.log` は再開で先頭が消える（Bug 18）
