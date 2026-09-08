# コンテナのビルドと起動

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

## ⚠️ 足したほうがよいマウント

**`/root/.claude` がマウントされていない。** コンテナを作り直すと
**会話履歴と memory（130 MB）が消える**（[作業環境とスマホからの接続.md](../docs/リポジトリ説明/作業環境とスマホからの接続.md)）。
`volumes` にこの1行を足せば解決する。

```yaml
            - ../../claude_home:/root/.claude
```

⚠️ **足す前に、いまの中身をホストへ退避すること**（マウントすると空で上書きされる）。

```bash
docker cp akita_sp_ppo:/root/.claude /home/irsl/irsl_docker_irsl_system/claude_home
```

⚠️ **権限の deny ルールは影響を受けない。** あれはリポジトリ内の
`.claude/settings.json`（git 管理下）にあるので消えない。

## ⚠️ `build_akita_sp.sh` の不整合

```bash
docker pull ... irsl_system:noetic       # ← スクリプト
FROM        ... irsl_system:24.04_one    # ← Dockerfile
```

**pull しているタグが使われていない。** `docker build` が `24.04_one` を自分で取りに行くので
動きはするが、`noetic`（Ubuntu 20.04 系）を無駄に取っている。直すなら pull 側を合わせる。

## 補足: `run.sh` はこのコンテナの起動には使っていない

ホストにある `run.sh` は **irsl_system 全般の起動スクリプト**で、
`irsl_system:24.04_one` を直接起動して choreonoid や jupyter を走らせるためのもの。
**本研究のコンテナ（`akita_sp_ppo`）は compose で起動している。**
混同しないこと。`run.sh` が呼ぶ `files/run_docker_main.sh` はリポジトリに無い。
