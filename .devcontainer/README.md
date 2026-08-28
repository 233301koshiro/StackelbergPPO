# devcontainer（Node 22 + Claude Code + Ponytail）

研究用コンテナ `akita_sp`（Choreonoid 2.3 / Python 3.12 / PyTorch 2.7+cu128）に、
Claude Code CLI と Ponytail プラグインを載せるための devcontainer 定義。

## 2つの導入経路（リビルドは必須ではない）

このリポジトリの開発コンテナ `akita_sp_ppo` は **4 週間連続稼働**しており、
`/root/.claude` などコンテナの書き込み層に状態を持つ。目的（Node 22 + Claude Code + Ponytail）は
リビルドなしでも達成できるので、**経路A を既定とする**。

### 経路A: 稼働中のコンテナにそのまま入れる（推奨・低リスク）

```bash
# ホストから
docker exec -it akita_sp_ppo bash -lc '
  apt-get update &&
  apt-get install -y --no-install-recommends ca-certificates curl gnupg &&
  install -d -m 0755 /etc/apt/keyrings &&
  curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key |
    gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg &&
  echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list &&
  apt-get update && apt-get install -y --no-install-recommends nodejs &&
  npm install -g @anthropic-ai/claude-code
'
# プラグイン（認証が要るならログイン後に再実行）
docker exec -it akita_sp_ppo bash -lc 'bash .devcontainer/post-create.sh'
```

- **利点**: 稼働中の状態・設定・履歴が一切失われない。実験データにも触れない。
- **欠点**: コンテナを作り直すと消える。だから下の定義も残しておく。

### 経路B: devcontainer としてリビルド（再現性は上がるがリスクあり）

**この経路を選ぶ前に、下の「リビルド前に必ずやること」を読むこと。**

## リビルド前に必ずやること

1. **稼働中の学習が 0 本であること**
   ```bash
   docker exec akita_sp_ppo pgrep -fc choreonoid_train.py   # → 0
   ```
2. **Claude Code の状態を退避**（`/root/.claude` は書き込み層にあり消える）
   ```bash
   docker cp akita_sp_ppo:/root/.claude ./claude-backup-$(date +%Y%m%d)
   ```
   リビルド後、名前付きボリューム `stackelbergppo-claude-home` へ戻す:
   ```bash
   docker cp ./claude-backup-YYYYMMDD/. <新コンテナ名>:/root/.claude
   ```
3. **`/root` 直下のその他**（`.gitconfig` `.bashrc` `.config` 等）も必要なら退避する
4. **古いコンテナは消さずに残す**。問題があれば戻れるようにしておく

## 稼働中コンテナから写した起動設定（2026-08-28 実測）

`devcontainer.json` の `runArgs` / `mounts` / `containerEnv` は、
稼働中の `akita_sp_ppo` を `docker inspect` して写した実値である。**憶測ではない。**

| 項目 | 実値 | 落とすとどうなるか |
|---|---|---|
| イメージ | `akita_sp` | — |
| `/userdir` | `/home/irsl/irsl_docker_irsl_system/userdir` を全体マウント | 他プロジェクトが見えなくなる |
| `/tmp/.X11-unix` | バインド | Choreonoid の GUI・可視化が動かない |
| `NetworkMode` | `host` | ROS（`localhost:11311` 前提）が繋がらない |
| `Privileged` / `CAP_SYS_ADMIN` / `seccomp=unconfined` | 有効 | 研究室イメージの前提が崩れる |
| `ShmSize` | 既定 64MB | 稼働実績があるので変更しない |
| GPU | `NVIDIA_VISIBLE_DEVICES=all` + ドライバ注入 | 学習が CPU に落ちる |

⚠️ **GPU の渡し方だけ未確定。** `Runtime` は `runc` で、nvidia ライブラリは
コンテナ内に注入されている（`/proc/mounts` で確認）。`--gpus all` を指定してあるが、
リビルド後に必ず検証すること:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 使う前に必ず確認すること

## 使う前に必ず確認すること

`devcontainer.json` の `build.args.BASE_IMAGE` を**実環境のイメージ名に合わせる**。

```bash
# ホスト側で実行
docker images | grep akita_sp
```

実イメージ名は `akita_sp`（`docker inspect` で確認済み）。このイメージはリポジトリ外の `Dockerfile.add_akita_sp`
（研究室標準 `repo.irsl.eiiris.tut.ac.jp/irsl_system:noetic` を継承）が生成している。

> ## ⚠️ 適用タイミング: 学習が全部終わってから
>
> **Rebuild Container は稼働中の学習プロセスを全て落とす。**
> この定義は作成済みだが、稼働中の run があるうちは適用しないこと。
>
> ```bash
> pgrep -fc choreonoid_train.py    # → 0 になってからリビルドする
> ```
>
> 現在地と残りの実験は [docs/研究応用/引き継ぎ_再起動後.md](../docs/研究応用/引き継ぎ_再起動後.md) を見る。

## 構成

| 手順 | 場所 | 理由 |
|---|---|---|
| Node.js 22（NodeSource） | Dockerfile レイヤー1 | 変更頻度が最も低い。キャッシュを最大限効かせる |
| Claude Code CLI（npm -g） | Dockerfile レイヤー2 | Node より更新が速いので層を分ける |
| Ponytail marketplace 追加 | `post-create.sh` | $HOME 配下の利用者状態。認証を要求しうる |
| Ponytail plugin 導入 | `post-create.sh` | 同上。失敗してもビルドを落とさない |

### Node を nvm で入れない理由

nvm は `~/.nvm/nvm.sh` を**対話シェルの初期化時に source する**方式のため、
`bash -c` の非対話シェルや VS Code 拡張が spawn するプロセスの PATH に乗らない
（`.bashrc` は非対話シェルで読まれない）。Ponytail の自動起動フックは node を
PATH に要求するので、この方式では要件を満たせない。

NodeSource なら実体が `/usr/bin/node` に置かれ、PATH 設定なしで
対話・非対話・VS Code 拡張のいずれからも見える。

### Claude Code CLI を $HOME 配下に置かない理由

`npm -g` の標準（`/usr/lib/node_modules`、実行ファイルは `/usr/bin/claude`）に置く。
prefix を `$HOME/.npm-global` に移すと PATH に `~/.npm-global/bin` を足す必要が生じ、
nvm と同じ「非対話シェルから見えない」問題を招くため。

実行ユーザーが root 自身なので `sudo npm -g` は使わず、所有者の混在は起きない。
利用者ごとの状態（設定・プラグイン）は Claude Code 自身が `$HOME/.claude` に置く。

## セットアップ後の動作確認

```bash
# 1. Node が 22 以上か（対話シェル）
node --version          # → v22.x.x 以上

# 2. 非対話シェルからも見えるか ★Ponytail の前提
bash -c 'command -v node && node --version'
env -i /bin/bash -c 'command -v node'    # 環境変数を空にしても見えること

# 3. Claude Code CLI
claude --version
command -v claude       # → /usr/bin/claude

# 4. Ponytail が有効か
claude plugin list      # 一覧に ponytail が出ること
claude plugin marketplace list
```

`claude plugin` が認証エラーで失敗した場合は、一度 `claude` を起動してログインしてから:

```bash
bash .devcontainer/post-create.sh   # 冪等。導入済みならスキップされる
```
