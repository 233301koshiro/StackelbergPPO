#!/usr/bin/env python3
"""M1（画像整形）を Gemini API で回す骨組み（実験系譜 9-123）。

⚠️⚠️ **未着手。キーが無いので動かない。**
**2 段階認証に指導教員の許可が要るため、キーの取得は保留**（2026-09-19、ユーザー判断）。
⭐ **構造と設計判断だけを先に確定させておく。** 実行は許可が下りてから。

---

## ⚠️ ユーザー提供のサンプルとの差異（**重要**）

ユーザーが示したサンプルは `gemini-robotics-er-1.5-preview` を使い、
**画像内の点を検出して JSON で返す**ものだった。

⚠️ **M1 が必要とするのはそれではない。**
**M1 は手描きスケッチを入力に、マーカー規約に従った整形画像を「生成」する工程**である
（出力は `B1_m1.jpeg` のような画像。`make_m1_prompt.py` がそのプロンプトを組む）。

| | ユーザーのサンプル | ⭐ **M1 が要るもの** |
|---|---|---|
| モデル | `gemini-robotics-er-1.5-preview` | **画像生成モデル**（`gemini-2.5-flash-image` 等） |
| 入力 | 画像 ＋ 指示 | **画像（手描き）＋ 指示** ← 同じ |
| 出力 | **JSON（点の座標）** | ⭐ **画像** |

**⭐ 入力の渡し方（`types.Part.from_bytes`）は同じなので、サンプルはそこだけ流用できる。**

---

## ⚠️ 再現性について — **Tripo とは性質が違う**（研究方針 段 2〜4）

| | 公式の保証 | 実測 |
|---|---|---|
| **Tripo3D（M2）** | 「同じ seed なら同一」 | ⚠️ 厳密には偽だが**後段が使う量は完全一致**（9-115） |
| ⭐ **Gemini（M1）** | ⛔ **決定性を保証しない**と明記 | **未測定** |

**⭐ したがって主張の形が決まっている**（研究方針より）:

> ✕ 「temperature を 0 にしたので出力は変わらない」
> ✅ **「ばらつきを最小に抑えた条件で、この最終プロンプトなら仕様を満たすことを実験で示した」**

⚠️ **低温でも出力は揺れる。** これを「限界」ではなく「測定対象」にするのが 9-84 の設計である。

---

## 使い方（キー取得後）

    export GEMINI_API_KEY=...        # ⚠️ .env に置く（gitignore 済み）
    pip install google-genai         # ⚠️ 未インストール

    python3 scripts/gemini_api.py --check                       # 疎通のみ
    python3 scripts/gemini_api.py --sketch data/test/B1/sketch/B1_hand.jpeg \
        --links 3 --ratios 0.55 0.5 0.5 --out data/api/B1_m1.png
    python3 scripts/gemini_api.py --sketch ... --n 10            # ⭐ 遵守率の測定（9-84）
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# ⚠️ 画像生成モデル。⭐ **版を固定して記録する**（Web UI ではこれができなかった）
# ⭐⭐ 既定は **gemini-3-pro-image**（2026-09-30）。
#   ⚠️ ブラウザ時代の M1 は **3.1 Pro（拡張機能）**であり、2.5-flash に落とすと
#   ⛔ **マゼンタ球がリンクに埋まり、先端カプセルが画面端で切れた**（実測比較）。
#   ⭐ 区間比の再現も 3-pro が最良（1.00 対 0.99。ブラウザは 0.97）。
MODEL_ID = os.environ.get('GEMINI_MODEL', 'gemini-3-pro-image')


def key():
    k = os.environ.get('GEMINI_API_KEY')
    if not k:
        env = ROOT / '.env'
        if env.exists():
            for ln in env.read_text().split('\n'):
                if ln.startswith('GEMINI_API_KEY='):
                    k = ln.split('=', 1)[1].strip()
    return k


def build_prompt(name, ratios):
    """⭐ プロンプトは `make_m1_prompt.py` が唯一の出所。**ここで組み直さない。**

    ⚠️ **2 箇所に同じことを書く構造を作らない**（CLAUDE.md §4-2）。
    付録A.2 もこのスクリプトの出力を転記する形にしてある。

    ⛔⛔ **2026-09-30 の事故**: ここは `--links` を渡していたが
    **`make_m1_prompt.py` にその引数は無い**。引数エラーで stdout が空になり、
    ⛔ **`returncode` を見ていなかったので空のプロンプトで 2 枚生成した**（課金つき）。
    ⭐ **失敗と空文字列を必ず例外にする。**
    """
    import subprocess
    cmd = [sys.executable, str(ROOT / 'scripts' / 'make_m1_prompt.py')]
    if name:
        cmd += ['--sketch', name]
    if ratios:
        cmd += ['--ratios'] + [str(r) for r in ratios]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f'make_m1_prompt.py が失敗（exit {r.returncode}）: '
                           + (r.stderr or '')[-300:])
    if not r.stdout.strip():
        raise RuntimeError('make_m1_prompt.py の出力が空。⛔ 空のプロンプトで生成しない')
    return r.stdout


def generate(sketch, prompt, out, temperature=0.0, seed=None):
    """⭐ 2026-09-30 に実疎通を確認（gemini-2.5-flash-image / gemini-3-pro-image）。"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key())
    cfg = dict(temperature=temperature)
    if seed is not None:
        cfg['seed'] = seed        # ⚠️ **決定性は保証されない**（公式明記）
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=Path(sketch).read_bytes(),
                                  mime_type='image/jpeg'),
            prompt,
        ],
        config=types.GenerateContentConfig(**cfg),
    )
    # ⭐⭐ **課金の手がかりを必ず出す**（2026-09-30）。
    #   ⚠️ 単価は API から取れないが、**トークン数は取れる**。
    #   ⭐ 画像 1 枚が何トークンかが分かれば、公式の単価表と掛けて費用が出る。
    u = getattr(resp, 'usage_metadata', None)
    if u is not None:
        print(f'    [usage] prompt={getattr(u,"prompt_token_count",None)} '
              f'output={getattr(u,"candidates_token_count",None)} '
              f'total={getattr(u,"total_token_count",None)}'
              + (f' 内訳={u.candidates_tokens_details}'
                 if getattr(u, 'candidates_tokens_details', None) else ''), flush=True)
    # ⚠️ 画像生成の戻り値は inline_data に入る。text ではない
    for part in resp.candidates[0].content.parts:
        if getattr(part, 'inline_data', None):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(part.inline_data.data)
            return out
    raise RuntimeError('画像が返らなかった。応答: ' + str(resp)[:300])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='疎通のみ（キーの有無を見る）')
    ap.add_argument('--sketch')
    ap.add_argument('--name', help='data/test/<名前>/sketch/measured.json から比を読む')
    ap.add_argument('--ratios', nargs='*', type=float)
    ap.add_argument('--out', default='data/api/m1_out.png')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--n', type=int, default=1, help='⭐ 同条件で n 回生成（遵守率の測定。9-84）')
    a = ap.parse_args()

    k = key()
    if not k:
        print('⛔ GEMINI_API_KEY が無い。')
        print('   ⚠️ **2 段階認証に指導教員の許可が要るため、キーの取得は保留中**（9-123）。')
        print('   ⭐ 許可が下りたら .env に GEMINI_API_KEY=... を追記する。')
        return 1
    try:
        import google.genai  # noqa: F401
    except ImportError:
        print('⛔ google-genai が未インストール。`pip install google-genai`')
        return 1
    if a.check:
        # ⛔⛔ **キーの有無だけを見ていた（2026-09-30 まで）。**
        #   無効なキーでも「✅ 準備完了」と出るので、**1 文字欠けたキーを合格と答えた。**
        #   ⭐ `models.list` は**無料**なので、実際に叩いて確かめる。
        from google import genai
        # ⚠️ `models.list()` は遅延評価。**クライアントを変数で保持しないと**
        #   途中で回収され `Cannot send a request, as the client has been closed` になる。
        cli = genai.Client(api_key=k)
        try:
            names = [m.name.replace('models/', '') for m in cli.models.list()]
        except Exception as e:
            print(f'⛔ **API に届かない**: {type(e).__name__} {str(e)[:160]}')
            print(f'   ⚠️ キーの長さ {len(k)} 文字（Google AI Studio は通常 39 文字）。'
                  '⭐ **途中で欠けていないか確かめる**')
            return 1
        ok = MODEL_ID in names
        print(f'✅ 疎通成功。呼べるモデル {len(names)} 個')
        print(f'{"⭐ **目的のモデルあり**" if ok else "⛔ **目的のモデルが無い**"}: {MODEL_ID}')
        if not ok:
            print('   ⚠️ **権限がモデル単位で制限されている可能性がある**（外部API.md の確認事項 1）')
        return 0 if ok else 1
    if not a.sketch:
        ap.error('--sketch を指定してください')

    prompt = build_prompt(a.name, a.ratios)
    print(f'プロンプト {len(prompt)} 文字（make_m1_prompt.py 由来）')
    for i in range(a.n):
        out = a.out if a.n == 1 else a.out.replace('.png', f'_{i+1}.png')
        print(f'  [{i+1}/{a.n}] → {generate(a.sketch, prompt, out, a.temperature, a.seed)}')
    if a.n > 1:
        print('\n⭐ 遵守率は `scripts/probe_m1_compliance.py` で測る（9-84）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
