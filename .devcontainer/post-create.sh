#!/usr/bin/env bash
#
# コンテナ作成後に1回だけ走るフック（devcontainer.json の postCreateCommand から呼ばれる）。
#
# ここに置いた理由 ─ ビルド時に実行できない/すべきでない手順だから:
#   1. `claude plugin ...` はプラグインを $HOME/.claude 配下に書き込む。
#      ビルド時に書いてもイメージに焼き付いてしまい、利用者ごとの状態としては不適切。
#   2. Claude Code の login/認証はビルド時にできない。marketplace の追加や
#      plugin install が認証やネットワークを要求した場合、ビルドを落とさずに
#      「起動後にやり直せる」形で扱いたい。
#   3. 認証が絡む手順で `docker build` が失敗すると、Node や CLI のレイヤーごと
#      作り直しになりキャッシュ効率が悪い。
#
# したがってこのスクリプトは **失敗してもコンテナ作成自体は成功させる**（fail soft）。
# 失敗時は理由と再実行方法を明示する。

set -uo pipefail   # -e は付けない: プラグイン導入の失敗でコンテナ作成を止めないため

log() { printf '[post-create] %s\n' "$*"; }

log "実行ユーザー: $(id -un) (uid=$(id -u))  HOME=${HOME}"

# --- 前提の検証: node が非対話シェルから見えるか -----------------------------
# Ponytail の自動起動フックは node を PATH に要求する。ここが通らなければ
# あとの手順は全て無意味なので、最初に確認して落とす。
if ! bash -c 'command -v node' >/dev/null 2>&1; then
  log "❌ 非対話シェルから node が見えない。Ponytail の自動起動フックが動かない。"
  log "   Dockerfile の Node インストール層を確認すること。"
  exit 1
fi
log "node   : $(command -v node) $(node --version)"
log "npm    : $(command -v npm) $(npm --version)"

if ! command -v claude >/dev/null 2>&1; then
  log "❌ claude CLI が見つからない。Dockerfile の npm install -g 層を確認すること。"
  exit 1
fi
log "claude : $(command -v claude) $(claude --version 2>/dev/null || echo '(バージョン取得不可)')"

# --- Ponytail プラグイン -----------------------------------------------------
# 対話モードの /plugin ではなく非対話サブコマンドを使う。
# 冪等にしたいので、既に入っていればスキップする。
MARKETPLACE="DietrichGebert/ponytail"
PLUGIN="ponytail@ponytail"

log "--- Ponytail プラグインの導入 ---"

if claude plugin marketplace list 2>/dev/null | grep -qi 'ponytail'; then
  log "marketplace は登録済み（スキップ）"
else
  log "marketplace add ${MARKETPLACE}"
  if ! claude plugin marketplace add "${MARKETPLACE}"; then
    log "⚠️ marketplace の登録に失敗した。認証が必要な可能性がある。"
    log "   コンテナ内で 'claude' を一度起動してログインしたのち、次を実行:"
    log "     bash .devcontainer/post-create.sh"
    exit 0
  fi
fi

if claude plugin list 2>/dev/null | grep -qi 'ponytail'; then
  log "plugin は導入済み（スキップ）"
else
  log "plugin install ${PLUGIN}"
  if ! claude plugin install "${PLUGIN}"; then
    log "⚠️ plugin install に失敗した。認証が必要な可能性がある。"
    log "   コンテナ内で 'claude' を一度起動してログインしたのち、次を実行:"
    log "     bash .devcontainer/post-create.sh"
    exit 0
  fi
fi

log "--- 導入結果 ---"
claude plugin list 2>&1 | sed 's/^/    /' || log "(plugin list を取得できなかった)"
log "完了"
