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
| `glb_to_links.py` | M3: 色検出で関節を見つけメッシュを分割 |。**2026-09-02 変更**: `meshes/joints.json` に関節の 3D 座標と**関節間距離**を出力するようになった（Bug 27）。根元リンクのローカル原点も XY=(0,0) 決め打ちから**第1関節の XY** へ変更（Bug 28）。**2026-09-02（2回目）**: マーカーのクラスタリングを **3 次元の連結成分**へ（Bug 30。Z のみだと短い腕で隣の球と融合する）。先端リンクの長さも「最後の関節からの最遠点距離」を `joints.json` に出す（Bug 31）
| `mesh_to_params.py` | M4: OBB からリンク長・カプセル半径を抽出 → topology.json |。**2026-09-02 変更**: `--joints-json` で bone_offset に OBB 主軸長ではなく**関節間距離**を使う（Bug 27）。`--vertical` で**根元ヨー + 以降ピッチ・ボーン +Z** の縦型を出力する（Bug 29。`FIXED_BASE=0` と併用）。**2026-09-02（2回目）**: 先端リンクも `--joints-json` の値を使うようになった（Bug 31。OBB は短い腕で +52.6 % 過大）
| `topology_to_xml.py` | M5: topology.json → MuJoCo 互換 XML |
| `run_2axis_mvp.sh` | rrbot 用の一気通貫（topology.json → XML/body → 学習） |
| `eval_pipeline_robustness.py` | **M3 の頑健性評価**（第4章 4.2.2 = 軸1）。色ドリフトに対する許容幅を tolerance 掃引で測る。表 4.4・4.5 と 図 4.1 の出典 |
| `make_scaled_arm.py` | 既存アーム XML の**リンク長だけ**を倍率スケールした XML を作る。診断の助言（「N 倍に伸ばせ」）を機械的に適用するのに使う（第4章 4.4.1） |
| `audit_xml_reach.py` | **アーム XML の公称リーチと実効リーチを検算**（Bug 23 の再発防止）。`fromto` だけ伸ばして `body pos` を伸ばし忘れた XML を検出する。不一致があれば終了コード 1。原論文由来の環境と分岐トポロジーは対象外 |
| `check_mesh_interference.py` | 静止姿勢の凸包による自己干渉チェック（v2系のみ） |
| `save_morphology_urdf.py` | 収束形態を URDF として書き出す |

⚠️ **`dynamic_body_updater.py` は実質デッドコード。** `run_2axis_mvp.sh` から呼ばれてはいるが、
学習時の Choreonoid `.body` 生成は `khrylib/rl/envs/common/mujoco_env_choreonoid.py` が行う。
**描画・形態まわりを直すときにここを触っても効かない**（2026-07-31 に実際に間違えた）。

## 3. 評価・診断（再利用する）

| スクリプト | 役割 |
|---|---|
| `diagnose_morphology.py` | **形態診断 M7**（第3章 3.12）。第1層は学習不要・約1秒・Choreonoid 不要 |。**2026-09-02**: 第3層に**関節の使用率**を追加（可動域の 10% 未満しか動かない関節に「固定してよい」と提案）。3.12.5 の「第2層・第3層は反実仮想的な助言を構成できていない」という限界を、**設計を減らす方向**で部分的に埋めるもの。断定せず警告に留める。総合判定が「幾何的な障害はありません」のとき、**第3層が未達を報告していればその旨を添える**（判定の権限は第1層のみだが、添えないと読み手には矛盾に見えるため）
| `rank_settle.py` | **判定所要エポックの較正**（第4章 4.3.4.2）。学習不要、log_train.txt を読むだけ |
| `audit_runs.sh` | 全 run の一次データ表。**md より先にこれを見る** |
| `check_docs_consistency.py` | **docs とログの突き合わせ。** 古い状態語・数値の食い違い・記録漏れを検出。値を更新したら必ず走らせる |
| `check_citations.py` | **本文の引用と参考文献リストを双方向で照合**。リスト起点で本文を探す方向（確実）と、本文から抽出してリストに無いものを探す方向（取りこぼし検出）の2つ。不一致で終了コード 1 |
| `compare_morphology.py` | 複数 run の収束形態を並べて比較 |
| `boundary_compare.py` | 境界張り付きの条件間比較（matched epoch） |
| ⭐ **`joint_fix_watch.py`** | **指摘13の自動化。**関節使用率を監視し、安定したら固定版を並行起動、完走後に判定する（9-95） |
| **`band_overlap.py`** | 2 群の best 値の帯が重なるかを機械判定（9-27/9-61/9-62 の目算を関数化） ⚠️ **2026-09-20 追加: どちらかが 1 seed なら「帯を作れない」と出して止める**（9-136 で実際に 1 点を帯として判定しかけた。9-34 の再発防止） |
| ⭐ **`probe_joint_mass.py`** | **段1b（駆動系の質量 = gear / トルク密度）の投入前確認**（9-97）。`--check-cfgs` で全 cfg が既定無効なことを機械で確認、既定でトルク密度 × gear の自重余裕と押し残りの表、`--write-xml` で `probe_shot_speed.py` に渡す XML を出す |
| **`probe_hockey_arm_wall.py`** | 腕を壁に衝突させても届くかを測る（9-94）。⭐ **割り切りの「根拠」を測る型のプローブ** |
| ⭐⭐ **`search_joint_axes.py`** | 関節軸を第1層で枝刈り（9-127）。81 通りを 0.2 秒、53 % 棄却 |
| ⭐ **`make_realistic_density_xml.py`** | リンク密度に根拠を与えた XML を作る（9-126）。薄肉パイプの実効密度 |
| **`make_obstacle_reach_xml.py`** | 障害物のある Reach の XML を作る（9-104）。⚠️ **障害物は `<body>` に入れる**（9-98） |
| **`gemini_api.py`** | M1 を Gemini API で回す骨組み（9-123）。⚠️ **未着手・キー待ち** |
| ⭐ **`tripo_api.py`** | Tripo3D API で画像→3D を生成（9-115）。`--balance` / `--verify-seed`。⚠️ キーは `.env` |
| ⭐⭐ **`check_before_conclusion.py`** | **結論を書く前に通す検査**（完走・再生・タスク達成・版・事前登録）。9-114 ⚠️⚠️ **2026-09-20 追加: ①判定対象が末尾でまだ動いていれば「打ち切り」警告（9-137）②XML に固定 body があれば「再生ごとに揺れる。1 エピソードで判定するな」警告（9-138）** |
| ⭐ **`audit_run_validity.py`** | **完走 run が正当な実験として扱えるか。**起動時刻から XML/cfg の版を逆引きし、**同じ名前で中身が違う組**を検出（9-105） |
| ⭐⭐ **`probe_shot_choreonoid.py`** | **Choreonoid の実走で**、初期 y ごとにゴールへ入る撃ち方があるかを数える（9-103）。⚠️ **9-65 の MuJoCo 版を置き換えるもの** |
| ⭐⭐ **`probe_wall_choreonoid.py`** | **壁が Choreonoid の実走で効くかを測る**（9-99）。⚠️ **壁を測るときは必ずこれ。** MuJoCo のプローブは壁について実走と無関係な答えを出す（9-98） |
| ⭐⭐ **`probe_joint_axis_choreonoid.py`** | **関節軸が Choreonoid の実走まで届くかを測る**（9-131）。各関節を 1 つずつ曲げ先端の動く向きを見る。⚠️ **軸を変えたら MuJoCo だけで済ませない**（9-98） |
| ⛔⭐ **`check_obstacle_clearance.py`** | **障害物 Reach で腕が柱を避けたかを機械判定**（9-132）。⚠️ **先端だけ見ない。**9-118 では先端が迂回していたのに途中のリンクが貫通していた |
| ⛔⛔⭐ **`check_cube_penetration.py`** | ⭐⭐ **主戦場の Pusher で腕が cube に食い込んでいないかを判定**（Bug 45 の未確認部分）。全リンクのカプセル（軸＋半径）と cube の直方体を突き合わせる。⚠️ **半径は `record_arm_trace.py` の `geom_size` から取る。**古い軌跡は半径を持たないので**軸だけの下限判定**に落ちる（その旨を出力する） |
| ⭐ **`check_extremes.py`** | **動く物体すべての極値を出し、境界を越えていないか全軌跡に掛ける**（9-96 の一般化）。⭐ **対象物が slide の可動限界に達していないか**（Bug 39 の型）と **腕が壁の内部に入っていないか**（9-80 の型）。⚠️ 到達タスクでは cube が動かないのが正常なので場合分けしてある |
| ⭐ **`rerecord_all_traces.sh`** | **全 run の軌跡を取り直す**（2026-09-23）。1 本 10 秒・GPU 不要なので学習と同時に回せる。⭐ **目視記録の空欄を埋める**のと **`geom_size` を入れ直す**のが目的 |
| ⛔⭐ **`probe_obstacle_collision_choreonoid.py`** | **柱が腕に効いているかを実走で測る**（9-133）。⭐ **同じ姿勢・同じゼロ制御で柱の有無だけ変えて比較する** |
| ⭐⭐ **`probe_episode_bias.py`** | **1 回の読み込みで N 話走らせ、1 話目が当てにならないかを測る**（Bug 47 / 9-142）。⚠️ **軌跡を反復しても検出できない** |
| ⭐⭐ **`plot_run.py`** | **軌跡を図にする**（9-155）。⭐ **全リンクの z 最小を題に焼き込む**ので床下が一目で分かる。`--all` で全 run |
| ⭐⭐ **`audit_design_space.py`** | **設計空間と物理のパラメータを全部並べ、根拠の有無を突き合わせる**（9-149）。⛔ **3 回数え直した末に最初の数えが取りこぼしていたので機械化した** |
| ⭐⭐ **`diagnose_morphology.py`（9-144 追加）** | **第1層が障害物を扱えるようになった。**⚠️ **目撃型**（避けて届く姿勢を見つけたときだけ断定。棄却側の確実性を壊さない） |
| ⭐ **`audit_implicit_physics.py`** | **XML に書かれていない物理パラメータを挙げる。**⭐ **Choreonoid が扱えない形状型（cylinder 等）も検出する**（9-118）。 ⭐ **新しいタスクを入れる前に必ず**（9-86。`solref` が既定のままでホッケーが 6 回沼った） |
| **`verify_run_replay.py`** | 学習済み run を 5 話再生し env の reward を積んで log と比べる。頭 2 話を捨てる（9-66） |
| **`probe_m1_compliance.py`** | M1 の仕様遵守率（マーカー数・比の保存）。マゼンタ検出は `probe_m1_tilt` を import |
| **`probe_m1_tilt.py`** | M1 画像の傾き。⚠️ **この量は合否を分けない**（9-79）。姿勢の判定は `check_glb_pose.py` |
| `check_glb_pose.py` | **M2 の後**に関節の Z 広がりで姿勢を判定。閾値 0.25 は実測較正済み |
| `check_stale_claims.py` | 主張の側から全 MD を横断検出 |
| ⭐ `check_section_refs.py` | **修論本文の節参照が実在する節を指しているか**。⛔ **章を繰り上げたとき、既存の 5 本はどれも検出できなかった**（旧 第6章 への参照 13 箇所を 1 週間見逃した。2026-10-02 新設） |
| `check_docs_inventory.py` | 孤立ファイル・壊れリンク・archive の注記漏れ。⭐ **冒頭に前回の棚卸しからの経過日数を出す**（1 週間を超えると ⛔。周期を散文ではなく検査に持たせるため。2026-09-25） |
| ⭐⭐ `probe_material_table_inert.py` | **ホッケー Phase 1 の関門**（9-172）。材質を足しても既存 run の物理が変わらないことを実測する。⭐ **再現性が無いときは判定せずに止まる**（方策の再生は `cube_y_noise` で 851 mm 揺れる） |
| ⭐⭐ `probe_puck_wall_restitution.py` | **パックが側壁で跳ね返るか**を実走で測る（9-173・9-176）。⭐ **期待値と実測を並べて出す**ので、効いていないときに推測が要らない |
| `probe_puck_coast.py` | パックの惰行距離を測る（9-173）。⚠️ **9-174 型④の実例**: 惰行のつもりで「中央板までの距離」を測っていた |
| ⭐⭐⭐ `probe_arm_wall_block.py` | **腕の壁貫通ブロックが効いているかを実走で測る**（9-189）。⭐ **捕捉したリンク名・壁名・発火回数・壁の中の標本・外面を越えた最大距離**を必ず出す（§5-2 ⑤-3-2）。⛔⛔ **掴めていないときは判定を拒否する** — 0 リンク・0 壁で「すり抜けなし」という偽の合格を実際に出した。⭐ `ARM_SWEEP_OFF=1` で旧実装（2 点判定）に戻して**既知の失敗を再現できるか**確かめられる |
| ⭐ `probe_shot_strikeability.py` | **狙える向きに当てられるか**を幾何で測る（9-178）。判定器に基準を足す前に「差が出るか」を確かめる用 |
| `probe_replay_shot.py` | 9-135 の実際の撃ち出しを反射あり/なしで再現する（9-179）。⚠️ **本番を再現できておらず信用しない**（腕を畳んでいる） |
| ⭐⭐ `check_code_docs_sync.py` | **コードと md のずれ**を見る（2026-09-27 新設）。⛔ **既存 4 本は docs の中しか見ず、「コードを変えたのに md が追随していない」を素通りする**。A 索引 / B 壊れ参照 / C 環境変数 / D queue の辿れなさ |
| ⭐⭐⭐ `audit_all_traces.py` | **全軌跡を過去の事故の型で洗う**（9-182）。⭐ **床下・step 数だけでは 2 本、6 項目にしたら 21 本。**①1step ②床下 ③可動域の端 ④通り抜け ⑤速度発散 ⑥対象が動かない |
| ⭐⭐⭐ `close_loop.py` | **判定 → 助言どおりスケール → 再判定** を 1 本で回す（助教の指摘12 / 9-183）。⭐ **結合テストの連結子。**無いと「助言に従う」段で人が描き直すことになり、生成の揺れ（18 %）が混入する。⚠️ **余裕を上乗せしない**（9-30）。⛔ **学習は自動投入しない**（事前登録と GPU 競合を人が確かめる） |
| `make_slide8_links_figure.py` | 進捗発表スライド 8「リンクに切り分けて寸法を測った」の図（→ `figures/slide8_links_split.png`） |
| `make_slide9_m1_variance_figure.py` | 同スライド 9「M1 のばらつき」の図（→ `figures/slide9_m1_variance.png`） |
| `queue_damping.sh` | ⭐ **ダンピング感度**（9-171。b=0.1/1/10 × Pusher × 2 seed）。助教の指摘3 の残り |
| ⭐ **`doc_refs.py`** | **識別子の逆引き。「ここを変えたらどこも変わる」を出す。** `--hubs` で影響の広い識別子、`--undefined` で壊れ参照 |
| `dump_thesis_outline.py` | 修論の見出しを書き出す。`--write` で 構成.md の一覧を再生成、`--check` で食い違い検査 |
| `eval_cnoid_numerical.py` | 数値で成功率・報酬を確認 |
| `eval_cnoid_visual.py` | 動画（mp4）で記録 |
| `eval_cnoid_viewer.py` | GUI でリアルタイム再生 |
| `eval_cross_env.py` | ネイティブ物理エンジンでの評価（サブプロセス分離） |
| `plot_rewards.py` | 学習曲線グラフ |
| `visualize_morph_changes.py` | 形態変化の可視化 |
| `generate_comparison_report.py` | 学習曲線 + eval を Markdown/PNG レポート化 |

使い方は [docs/リポジトリ説明/評価スクリプト.md](../docs/リポジトリ説明/評価スクリプト.md)。

| `record_arm_trace.py` | 学習済み方策を1エピソード実行し、各リンクのワールド変換・**最適化後の** `bone_offset`・cube・目標を npz に落とす。デモ動画の前半 ⚠️⚠️ **2026-09-20: 既定を 400 → 1200 step に変更**（旧既定はエピソードの 27〜65 % しか記録せず、打ち切りを黙って通していた。Bug 46 / 9-137） |
| `render_arm_video.py` | その npz と**実物メッシュ**（STL）から mp4 を書き出す。近似の円柱ではない。使い方と踏んだ罠は [評価スクリプト.md](../docs/リポジトリ説明/評価スクリプト.md) |
| `extract_gear.py` | 指定 run の checkpoint から、収束した gear・リンク長を取り出す |
| `check_stale_claims.py` | docs 全体から「事実と食い違う主張」を横断検出。**限界・未達の状態が変わったら必ず** |
| `validate_diagnosis.sh` | 診断 M7 の三層すべてを既知の run に当てて回帰確認（第3章 3.12） |

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
| `probe_m2_variance.py` | 同じ M1 画像から n 個の GLB を通し、**三次元化だけの分散**を測る | 実験系譜 9-39（±0.7 %。18 % は M1 由来と判明） |
| `probe_cube_y_noise.py` | `cube_y_noise` が**実機で**効いているか（パックの y が実際に振れるか） | 実験系譜 9-33 |
| `probe_shot_geometry.py` | 撃ち出し角を振って**「反射が必須」になる初期位置**を測る（幾何のみ・物理エンジン不要） | 実験系譜 9-55 追記の再現 + 9-59 |
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
`august_queue.sh`（2026-08-10〜15 の無人運転）`launch_mechanism_probe.sh`（08-19 の3本）
`queue_fix4.sh`（09-08、第3層の境界判断の検証。実験系譜 9-32）
`queue_hockey.sh`（09-08、**9-50 で畳んだ C の残骸**。何も投入せず 09-11 12:00 に自動終了）
`queue_hockey_wall.sh`（09-09、**壁つき**ホッケー 1 本。9-52 の完走待ち。実験系譜 9-60）
`queue_fix3_s2.sh`（09-09、9-27 の seed=1。実験系譜 9-52）
`queue_actuator_mass.sh`（09-14、**段1a＋段1b** の Pusher 1 本。3 epoch スモーク → NaN 無しなら 200 epoch。実験系譜 9-97）

## 6. スケジューラ（現行は1つだけ）

| スクリプト | 状態 |
|---|---|
| `experiment_queue.sh` | ✅ **現行。** 完走マーカーで判定する版（Bug 19 対応済み） |
| `check_glb_pose.py` | ✅ **現行（2026-09-03）。** GLB が M3 を通せる姿勢かを**パイプラインの前に**判定する（9-23。A3 で Tripo3D が 3/4 の斜め線を「奥へ伸びる」と解釈して腕が寝た）。マゼンタのクラスタ数・関節の Z 方向の広がり・隣接差を見る。**Bug 30 で検出は 3D 化したが分割は今も Z なので「腕は立っていること」が前提** |
| `check_docs_inventory.py` | ✅ **現行（2026-09-02）。** docs の棚卸しの機械的な部分（孤立ファイル・壊れた内部リンク・archive の注記漏れ・archive なのに現役参照・長期未更新）。**週に1回**走らせる（CLAUDE.md §4-2 定期棚卸し）。⚠️ **「消せ」とは言わない。**孤立＝不要ではなく、台帳が索引から漏れているだけのことがある |
| `make_fixed_joint_arm.py` | ✅ **現行（2026-09-02）。** 指定した関節を固定した XML を作る（第3層の「この関節は固定してよい」という助言を閉ループで検証するため。9-20）。**`<joint>` を削除せず range=0 にして actuator を外す**ので、リンク長・質量は元と完全に同一で動く自由度だけが減る |
| `make_m1_prompt.py` | ✅ **現行（2026-09-02）。** M1 用の Gemini プロンプトを貼ってすぐ使える形で組み立てる。**共通部の正本はこのファイル**（md に貼ると 5 枚ぶん腐るため）。比は `data/test/<名前>/sketch/measured.json` の**実測値**を使う。`--sketch B1` / `--ratios 0.55 0.5 0.5 --emphasis short` |。**2026-09-03**: 【描いてはいけないもの】節を追加（寸法線・数値・文字／指やグリッパ／複数ビュー）と「1つの関節に球は1個だけ」を追加（Bug 33。比を数値で指示すると生成側が図に描き込む）
| `queue_e2e_a1.sh` | ✅ **現行（2026-09-02）。** A1 の学習を **2 本ずつ直列**に投入する（平面 seed=0 → 縦型 seed=0 → 縦型 seed=1）。4 本同時では **GPU が 92 % で頭打ち**になり `T_update` が 2.4 倍に伸びたため。完走判定は `training done!` の有無（`pgrep -f` / `pkill -f` は自分のコマンドラインに当たって誤爆する。Bug 19） |
| `queue_pjd_pusher.sh` | ✅ **現行（2026-08-24）。** 配分判別の Pusher 版4本を空きを見て順に投入（9-17）。投入前に `audit_xml_reach.py` で XML を検算する |
| `queue_pjd_real.sh` | ✅ **完了（2026-09-22）。** ⭐ **実寸条件（ギア上限 150）の配分判別・Reach 版 4 本**（系譜 9-160 → 結果 9-163） |
| `queue_pjdp_real.sh` | ✅ **現行（2026-09-23）。** ⭐ **同じ実寸条件の Pusher 版 4 本**（系譜 9-165）。⚠️ 旧 Pusher（`tripo_pjdp_*`）との差はギア上限 400→150 と XML の `_real` のみ |
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
| `make_hockey_court_xml.py` | ホッケー系の XML を `e2e_hockeyv.xml` から生成（9-38）。`--wall` で**側壁・ゴール口・ゴール前の板**を物理に入れる（9-55）。⚠️ **同時に表示用 `.body` 2 種も生成する**（9-58）。`assets/choreonoid/bodies/hockey/` の台とパックは**この生成物なので手で編集しない** |
| `draw_hockey_template.py` | ホッケー用アームの**下書きテンプレート**を描く。⚠️ 研究の入力ではない（下敷き） |
| `test_cube_y_noise.py` | `cube_y_noise` の実装・範囲・評価時の決定性の自己チェック（`python3` で直接実行） |
| `smoke_test_cnoid.py` | Choreonoid 接続のスモークテスト |
| `cnoid_transfer.py` | 転用まわりの補助 |
| `monitor_training.py` | 学習監視（現在は使っていない） |
| `plot_pj_comparison.py` | PJ実験の比較図 |
| `build_thesis_pdf.py` | **修論 PDF ビルド**（`docs/研究応用/修論ドラフト/` の全 md → xelatex で1冊に）。下記の注意を読んでから使う |
| `build_handout_pdf.py` | **中間発表の配布資料（論文2枚版・6枚版）を PDF 化**。`build_thesis_pdf.py` の変換ロジックを再利用し、プリアンブルだけ article へ差し替える。⚠️ 余白や行送りを変えるとページ数が名前とずれるので要確認 |
| `make_thesis_figures.py` | **修論の図を一次データから生成**（`figures/*.png`）。学習曲線は `log/log_train.txt` から読む。図の数値は手で直さず、原本を確認してこれを再実行する |
| `eval_morphology.py` | 形態評価（docs・コードのどちらからも参照なし。`compare_morphology.py` に役割が吸収されたとみられる） |

---

## build_thesis_pdf.py の注意（節番号の罠）

```bash
python3 scripts/build_thesis_pdf.py --out docs/pdf/修論ドラフト_YYYYMMDD.pdf
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
pdftotext -f 1 -l 5 docs/pdf/修論ドラフト_YYYYMMDD.pdf - | grep -E "^\s*3\.[0-9]+"
```

で目次を出し、**md の節番号と一致するか**を見る。とくに他章から参照されている節
（`grep -rn "3\.12" docs/`）が合っているかを確認する。

---

## 新しくスクリプトを足すときの約束

1. **先頭に日付と目的を書く。** 「何を確かめたくて作ったか」が分かれば、後から再利用可否を判断できる
2. **一度きりなら §5 に、再利用するなら §3 に追記する**
   ⚠️ **`docs/リポジトリ説明/評価スクリプト.md` にも書く。2 箇所ある**
   （2026-09-13 に 3 本が両方から漏れていた。CLAUDE.md §4-2 の「2 箇所に同じことを書く構造」）
3. 学習を起動するなら**二重起動ガード**を入れる（`experiment_queue.sh` の各段が参考）
4. ログ解析なら **`log/log_train.txt` を読む**。`stdout.log` は再開で先頭が消える（Bug 18）
