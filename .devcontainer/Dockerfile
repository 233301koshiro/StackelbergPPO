# syntax=docker/dockerfile:1
#
# StackelbergPPO 開発用 devcontainer
#
# 土台は研究室標準イメージ irsl_system:noetic を継承した個人用イメージ akita_sp。
# Choreonoid 2.3 / Python 3.12 / PyTorch 2.7+cu128 はそのイメージが既に持っているので
# **ここでは再ビルドしない**。このレイヤーが足すのは Node.js 22 と Claude Code CLI だけ。
#
# BASE_IMAGE は環境ごとに違うので devcontainer.json の build.args で上書きする。
#   docker images | grep akita_sp   で実際の <イメージ名>:<タグ> を確認すること。
ARG BASE_IMAGE=akita_sp:latest
FROM ${BASE_IMAGE}

# 実行ユーザーは root のまま維持する。
#   - 既存の研究環境がこの前提で組まれている（/root/.claude に Claude Code 設定が既にある）
#   - 実行ユーザー自身が root なので `sudo npm -g` を使う必要がなく、所有者の混在が起きない
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# レイヤー1: Node.js 22（NodeSource / システム全体）
#
# ⚠️ nvm を使わない理由:
#   nvm は ~/.nvm/nvm.sh を **対話シェルの初期化時に source する** 方式なので、
#   `bash -c "node --version"` のような非対話シェルや、VS Code 拡張が spawn する
#   プロセスの PATH には乗らない（.bashrc は非対話シェルで読まれない）。
#   Ponytail の自動起動フックは node を PATH に要求するため、この方式では要件を満たせない。
#   NodeSource なら実体が /usr/bin/node に置かれ、PATH 設定なしで
#   対話・非対話・VS Code 拡張のいずれからも見える。
#
# 変更頻度が最も低いのでこのレイヤーを先に置き、キャッシュを効かせる。
# ---------------------------------------------------------------------------
ARG NODE_MAJOR=22
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 # ビルド時点で「非対話シェルから node が見える」ことを検証する。
 # ここで落ちれば Ponytail の前提が壊れた状態のイメージを配らずに済む。
 && bash -c 'command -v node && node --version && npm --version' \
 && node -e 'const v=process.versions.node.split(".")[0]; if (+v < 22) { console.error("node >= 22 が必要: "+process.versions.node); process.exit(1); }'

# ---------------------------------------------------------------------------
# レイヤー2: Claude Code CLI
#
# 実行ユーザー（root）自身で入れるので sudo は使わない = 所有物の混在は起きない。
# 配置先は npm -g の標準（/usr/lib/node_modules、実行ファイルは /usr/bin/claude）。
#   $HOME 配下に prefix を移す案もあるが、それだと PATH に ~/.npm-global/bin を
#   足す必要が生じ、nvm と同じ「非対話シェルから見えない」問題を招くため採らない。
#   ユーザーごとの状態（設定・プラグイン）は Claude Code 自身が $HOME/.claude に置く。
#
# CLI は Node より更新が速いのでレイヤーを分け、CLI 更新時に Node 層を再ビルドしない。
# ---------------------------------------------------------------------------
RUN npm install -g @anthropic-ai/claude-code \
 && npm ls -g --depth=0 @anthropic-ai/claude-code \
 && bash -c 'command -v claude'

# Ponytail プラグインの導入はここでは行わない（理由は post-create.sh の冒頭を参照）。
