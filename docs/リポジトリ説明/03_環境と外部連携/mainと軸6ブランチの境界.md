# main と軸6 ブランチの境界

RobotAssembler（軸6）の設定が、main の実験（カプセルの腕・実在の物理）に混ざらないようにするための一覧とチェックリスト。逆向き（main の変更が軸6 の run を黙って変える）も同じく防ぐ。

**確かめ方は検査 1 本**（2026-11-02、ユーザー「何回も言っているけど、混ざらないようになっている？」。11-01 にも同じ問いがあり、そのときは口頭の答えだけで、検査も一覧も残さなかった）:

```bash
python3 scripts/check_branch_isolation.py              # ⛔ が出たら混ざっている疑い（終了コード 1）
python3 scripts/check_branch_isolation.py --reproduce  # 検査そのものが既知の混ざり方を拾えるか
```

下の一覧は、この検査が毎回 git から出し直す一覧の 11-02 時点の写しである。食い違ったら検査の出力を正とする。

## 混ざってはいけない 3 つの層

| 層 | 中身 | どこにあるか | 混ざらない仕組み |
|---|---|---|---|
| ① RobotAssembler の設定 | 部品・スケール表・部品の質量・組み方・1/5 の腕の XML と cfg | **ブランチだけ**（`e2e_a1v_ra15.xml`・`e2e_a1v_ra15_mass.json`・`pusher_ra15.yml`・道具 19 本） | main には存在しない。main の run はこの XML と cfg を名指しできない |
| ② 幾何のリンクの設定 | カプセルの骨・太さ・横のずれ・長さから作る質量 | **共用**（`khrylib/robot/xml_robot.py`）。ブランチの版だけが軸6 の口を持つ | 口は cfg のキー（`capsule_lateral`・`part_mass_file`）。キーが無ければ従来どおり。main の cfg と main の run の config にこのキーは無い（検査 C） |
| ③ 物理の環境 | `CNOID_*` の環境変数、`*_phys` の XML の値 | **共用**（`mujoco_env_choreonoid.py`・`choreonoid_train.py`）。値は run ごとの `launch_env.json` に残る | 軸6 だけの値は環境変数の口（`CNOID_JOINT_LIMIT_TORQUE`）で、既定では効かない。main の run の launch_env に無く、軸6 の run の launch_env には必ずある（検査 C）。例外は四脚（`cfg=quad_walk_ra`、系譜 9-342）だけで、軸6 の部品を近似した物理の起動行を main で使うので `CNOID_JOINT_LIMIT_TORQUE` を持ってよい（検査の `MAIN_OK_ENV`。2026-11-10 ユーザー判断。腕の main の run が持てば ⛔ のまま） |

見た目だけのもの（画面のデモ・mp4）は物理に触れない。`CNOID_VISUAL_RAW`・`ra_visual/` のメッシュは、学習・再生・判定のどれも読まない。

## 共用しているファイル（main にあり、軸6 の口を持つ。変えるとブランチの run にも効く）

| ファイル | 層 | 軸6 の口 | 口が無いとき |
|---|---|---|---|
| `khrylib/robot/xml_robot.py` | ② 幾何 | （ブランチの版だけ）`capsule_lateral`・`part_mass_file`。main の版は `<inertial>` を消さない直し（Bug 86）を持つ | 従来どおり |
| `khrylib/rl/envs/common/mujoco_env_choreonoid.py` | ③ 物理 | `CNOID_JOINT_LIMIT_TORQUE`・`_SLACK`（可動域をトルクで掛ける、Bug 85） | 書き戻しの可動域（従来どおり） |
| `scripts/choreonoid_train.py` | ③ 物理 | 上の 2 つを `launch_env.json` に書き残す | 指定していない run のファイルは変わらない |
| `scripts/record_arm_trace.py` | ③ 物理（再生） | 同じ 2 つを再生の一覧に持つ | 同上 |
| `design_opt/envs/pusher.py` | タスク | `init_contact_margin`（既定 0.03）・`cube_init_noise`（無ければ乱数の引き方も同じ） | 従来どおり |
| `scripts/diagnose_morphology.py` | 判定 | `length_scale`・`bone_len`・部品で固定した設計変数 | 既定 1 ＝従来どおり（`--self-check` 3 項目） |
| `scripts/compare_morphology.py` | 判定 | （`bone_len` は出力の列の名前で、軸6 の cfg キーとは別。検査が語で拾っている） | — |
| `khrylib/rl/envs/common/visual_mesh.py` | 見た目 | `CNOID_VISUAL_RAW`・`joints.json` の `tip`・頂点色の GLB | 既定の出力はバイト単位で同じ（11-02 に md5 で確かめた） |
| `scripts/render_arm_video.py` | 見た目 | `tip`・頂点色の GLB | 従来どおり |
| `scripts/prune_archived_ckpts.sh`・`sync_branch_md.sh`・`queue_1029_fix1_seed2.sh` | 運用 | 軸6 の run を見分ける・md をブランチへ取り込む（コメントでの言及を含む） | — |

## 両方にあって中身が違うファイル（理由のあるものだけ。増えたら検査 A が ⛔）

| ファイル | 理由 |
|---|---|
| `khrylib/robot/xml_robot.py` | ② 幾何の軸6 の口（系譜 9-328） |
| `scripts/demo_cnoid.sh` | 画面のデモの `ra_*`（XML がブランチにしか無い） |
| `scripts/watch_midrun.sh`・`scripts/queue_postrun_replay.sh` | 見張りと自動再生を軸6 の run に向ける（系譜 9-329） |

`single_run/` は worktree から main へのシンボリックリンクで、結果は 1 か所に集まる（意図した作り）。どちらで走ったかは各 run の `.hydra/hydra.yaml` の `runtime.cwd` に残り、検査 B がそれを見る。

## チェックリスト

**A. main の共用ファイルを変えるとき**
1. 軸6 だけで要る振る舞いは、cfg のキーか環境変数の口にし、**既定では今までと同じ**にする
2. 既定で同じことを、数値かバイト単位で比べて確かめる（例: 11-02 の `visual_mesh.py` は A1 の 4 リンクの .dae の md5 を新旧で比べた）
3. main にコミットする → worktree で動いている bash が無いことを確かめて `git merge main`（`check_branch_isolation.py` の D が「取り込み済み」になる）
4. `python3 scripts/check_branch_isolation.py`

**B. 軸6 の run を投入するとき**
1. worktree（`/userdir/StackelbergPPO_ra`）から起動する。キューに `git rev-parse --short HEAD` を書く
2. 検査 D が「取り込み済み」（main の共用コードの直しが入っている）
3. 起動行に `CNOID_JOINT_LIMIT_TORQUE=1`（完走後に検査 C が launch_env で確かめる）

**C. main の run を投入するとき**
1. main（`/userdir/StackelbergPPO`）から起動する
2. 軸6 の XML・cfg・cfg キー・環境変数を使わない（検査 C）

**D. 軸6 の結果を再生する・描くとき**
1. 再生は worktree から（XML がブランチにしか無い）。main から描く道具には `--xml /userdir/StackelbergPPO_ra/assets/mujoco_envs/e2e_a1v_ra15.xml` を渡す（省くと箱が点で描かれる。系譜 9-330 の 11-02 の追記）
2. 画面のデモは `cd /userdir/StackelbergPPO_ra && bash scripts/demo_cnoid.sh ra_beta`

**E. ブランチで共用ファイルを変えたくなったとき**
1. まず main で口を足せないかを考える（足せるなら A へ）
2. ブランチだけで変えるなら、`check_branch_isolation.py` の `KNOWN_DIFF` と上の表に理由を書く

## いつ回すか

- 軸6 の run を投入する前と、main の共用ファイルを変えてコミットした後
- 週に 1 回の棚卸し（CLAUDE.md §4-2「定期棚卸し」）
