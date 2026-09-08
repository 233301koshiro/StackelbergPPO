# コンテナのビルドと起動

> ## ⚠️ まず読む — **コンテナの定義はこの repo に 2 つある**
>
> | | 場所 | 状態 |
> |---|---|---|
> | **経路A** | **`docker/`（ここ）＋ VS Code の「実行中のコンテナにアタッチ」** | ✅ **これで運用中**（39 日連続稼働） |
> | 経路B | [`.devcontainer/`](../.devcontainer/) | **未使用**。リビルドが要るので選ばれていない |
>
> **経路Aを既定とする判断は [.devcontainer/README.md](../.devcontainer/README.md) に書いてある**
> （稼働中コンテナに `docker exec` で Node 22 + Claude Code を入れる方が低リスク、という理由）。
>
> ⚠️ **両者は同じではない。** とくに `/root/.claude`:
> 経路B（`.devcontainer/devcontainer.json`）は名前付きボリューム
> `stackelbergppo-claude-home` で**保持する**が、
> **経路A（いま動いている方）はマウントしていないので、コンテナを作り直すと消える。**
> これは `.devcontainer/README.md` も警告している既知の状態であり、**放置は意図的**。
>
> **片方を直したらもう片方も見ること**（CLAUDE.md §4-2 の「2 箇所に同じことを書く構造」）。

**ホスト側**（`/home/irsl/irsl_docker_irsl_system/` 配下）にあったものを 2026-09-08 に取り込んだ。
経緯と、いまの環境との差分は [../docs/リポジトリ説明/環境の再現.md](../docs/リポジトリ説明/環境の再現.md)。

| ファイル | 役割 |
|---|---|
| `Dockerfile.add_akita_sp` | `irsl_system:24.04_one` に torch 2.7.0+cu128 等を足して `akita_sp` を作る |
| `build_akita_sp.sh` | 上のビルド。⚠️ **pull するタグが Dockerfile と違う**（下記） |
| `docker-compose.yml` | **コンテナの起動。これが本命** |

## 起動

```bash
docker compose up -d      # コンテナ名 akita_sp_ppo、sleep infinity で常駐
docker exec -it akita_sp_ppo bash
```

⚠️ **`docker-compose.yml` の `../../userdir` は相対パス。**
compose ファイルの2つ上の階層に `userdir` がある前提（＝ `/home/irsl/irsl_docker_irsl_system/userdir`）。
**置き場所を変えるとマウントが外れる。**

## 起動設定の意味（消すと何が壊れるか）

| 設定 | 効いているもの |
|---|---|
| `volumes: ../../userdir:/userdir` | **リポジトリ・`single_run/` 53 GB・`data/`。外すと全部見えない** |
| `volumes: /tmp/.X11-unix` + `DISPLAY=$DISPLAY` | **Choreonoid の GUI**（`DISPLAY=:1` で起動している） |
| `deploy...capabilities: [gpu, ...]` | **学習。外すと GPU が見えず torch が CPU に落ちる** |
| `network_mode: host` | ROS・Jupyter のポート |
| `privileged` / `SYS_ADMIN` / `seccomp:unconfined` | Choreonoid の一部機能 |
| `DOCKER_ROS_SETUP=/choreonoid_ws/install/setup.bash` | Choreonoid のパス解決 |

## `/root/.claude` が消える件（**対策済み。ただし未発動**）

**経路A では `/root/.claude`（会話履歴と memory、130 MB）がコンテナの書き込み層にあり、
作り直すと消える。** これは既知で、**対策は経路B に用意されている**
（`.devcontainer/devcontainer.json` の名前付きボリューム `stackelbergppo-claude-home`）。

**いまは経路A なので発動していない。** 選択肢は 2 つ。

| | やること | 代償 |
|---|---|---|
| **そのまま（現状）** | 何もしない。**作り直す前に手で退避する** | 退避を忘れると消える |
| 経路B へ移る | `.devcontainer/` でリビルド | **コンテナを作り直す＝学習が止まる** |

**作り直す前に必ず退避する:**

```bash
docker cp akita_sp_ppo:/root/.claude /home/irsl/irsl_docker_irsl_system/claude_home
```

⚠️ **権限の deny ルールは影響を受けない。** あれはリポジトリ内の
`.claude/settings.json`（git 管理下）にあるので消えない。
⚠️ **研究の中身も影響を受けない。** `docs/` は `/userdir` のマウント上にあり git にも入っている。
**消えるのは会話履歴と memory だけ。**

## ⚠️ `build_akita_sp.sh` の不整合

```bash
docker pull ... irsl_system:noetic       # ← スクリプト
FROM        ... irsl_system:24.04_one    # ← Dockerfile
```

**pull しているタグが使われていない。** `docker build` が `24.04_one` を自分で取りに行くので
動きはするが、`noetic`（Ubuntu 20.04 系）を無駄に取っている。直すなら pull 側を合わせる。

## 補足: VS Code は「実行中のコンテナにアタッチ」で入っている

`.devcontainer/` があるが、**それでコンテナを作ってはいない**。
compose で起動したコンテナに、VS Code の
**Dev Containers: Attach to Running Container** で入る運用である
（`/root/.vscode-server` がコンテナ内にあるのがその痕跡。39 日連続稼働）。

**この方式では `devcontainer.json` は読まれない。** だから経路B のマウント設定も効かない。

## 補足: `run.sh` はこのコンテナの起動には使っていない

ホストにある `run.sh` は **irsl_system 全般の起動スクリプト**で、
`irsl_system:24.04_one` を直接起動して choreonoid や jupyter を走らせるためのもの。
**本研究のコンテナ（`akita_sp_ppo`）は compose で起動している。**
混同しないこと。`run.sh` が呼ぶ `files/run_docker_main.sh` はリポジトリに無い。
