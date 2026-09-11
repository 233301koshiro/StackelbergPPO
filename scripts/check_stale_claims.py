#!/usr/bin/env python3
"""docs 全体から「事実と食い違っている可能性のある記述」を洗い出す。

**なぜ要るか**: CLAUDE.md §4-2 の反映先テーブルは「どこに書くか」を定めるが、
**テーブルに載っていないファイル**（配布資料・想定問答・凍結スナップショット等）が
更新から漏れる。2026-09-02 に実際に漏れた:

  - 論文6枚/2枚・中間発表原稿が「E2E は未実施」と書いていた（9/1 に 1 例通っている）
  - 第4章は 4.5.3 を直したのに 4.6 の要約が古いまま残っていた
  - 「系→システム」の統一が配布資料へ届いていなかった

トリガー駆動の逐次反映だけでは、**同じ主張が複数ファイルに散っている**場合に漏れる。
本スクリプトは主張の側から横断して検出する。

使い方: python3 scripts/check_stale_claims.py
終了コード: 検出 0 件なら 0、1 件以上なら 1
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'

# 「系」は複合語・run 系統の略記が大半なので、**システムの意味で使われている形だけ**を狙う。
# 「系が動く」「系の成立」「系全体」「系を構築」のように、単独の名詞として使われるもの。
SYSTEM_SENSE = re.compile(
    r'(?<![ぁ-んァ-ヶ一-龥A-Za-z0-9`_])系(?=[がをはにのも]?\s*(?:動く|成立|構築|実現|全体|'
    r'として|主軸|の内側|が要求|の中核|を作った|が頑健|としての))')

# 「余裕 10 %」は**否定されたことを述べている文**が大半。反証語が近くにあれば正常。
# 「消さずに経緯を残す」のがこの repo の訂正作法（CLAUDE.md §8）なので、
# 打ち消し線つきの旧文や「→ 解消」注記は**訂正済み**として扱う。
# これを入れないと、正しく訂正した箇所ほど検出される（2026-09-03 に 10 件の誤検出）。
REFUTED = re.compile(r'(?:間違いだった|根拠を失|否定され|反証|単調に悪化|しかし|当初は|改めた|'
                     r'成立していない|再設計|誤り|撤回|取り消|~~|→ *\*\*|解消|済み)')

CHECKS = [
    dict(name='E2E を「未実施」と断定している',
         pat=r'(?:E2E|end-to-end)[^。\n]{0,25}(?:実走|通し実行)[^。\n]{0,10}は?(?:未実施|未実走|未着手)',
         why='⚠️ 2026-09-11 更新: 5 枚描き 4 枚通過・うち 4 例は学習まで完走。'
             '「1 例のみ」も既に古い。「5 枚描き 4 枚通過」「成功率を出せる件数ではない」と書く'),
    dict(name='E2E を「1 例のみ」と書いている（2026-09-04 以降は古い）',
         pat=r'(?:E2E|通し実行)[^。\n]{0,30}(?:1|一)\s*例のみ',
         why='5 枚描き 4 枚通過。うち 4 例は co-design 判定まで学習を完走した'),
    dict(name='co-design まで含めた通しを「未完」と書いている',
         pat=r'co-design\s*判定まで[^。\n]{0,20}未完',
         why='A1 を平面 1 seed・縦型 2 seed で完走。さらに A1v/A2v/B2v も完走（計 26 run）'),
    dict(name='縦型モードの出力での学習を「未完」と書いている',
         pat=r'縦型モード[^。\n]{0,40}(?:学習|その出力)[^。\n]{0,10}(?:は)?未完',
         why='e2e_a1v / a2v / b2v が完走済み'),
    dict(name='余裕 10 % を「推奨」として肯定的に書いている',
         pat=r'余裕[^。\n]{0,12}10\s*%[^。\n]{0,16}(?:推奨|併記|加え|示す)',
         why='実験で否定済み（9-11）。到達タスクでは余裕を広げるほど単調に悪化する',
         filter='refuted'),
    # IME の誤変換で日本語のつもりがキリル・ハングルになる事故が繰り返し起きている
    # （2026-08 に 명示→明示・성績→成績、2026-09-02 に 手описき）。
    # ギリシャ文字は τ φ π Δ θ など数式記号として正当なので**対象にしない**。
    dict(name='キリル・ハングルの混入（IME の誤変換）',
         pat=r'[Ѐ-ӿ가-힯ᄀ-ᇿ]',
         why='日本語のつもりが別の文字になっている。見た目が似ていて気づきにくい'),
    # 2026-09-03 追加: 今月ひっくり返った主張のうち、数値照合では捕まらない3件。
    # いずれも「数値は正しいが記述が現在を指していない」型で、
    # check_docs_consistency（ログとの突き合わせ）では原理的に検出できない。
    dict(name='非平面でのタスク識別を「未検証」と書いている',
         # ⚠️「**パイプライン由来の**非平面形態で識別が成立するか」は別の限界で、
         # これは今も未検証（縦型モードの学習が稼働中）。混同しないよう除外する。
         pat=r'(?<!パイプライン由来の)非平面[^。\n]{0,20}(?:タスク識別|識別)[^。\n]{0,12}(?:は)?未検証',
         exclude=r'パイプライン',
         why='2026-08-23 に 9-16 で成立を確認（2 seed）。残る限界は「v3 がパイプライン出力でない」ことの方',
         filter='refuted'),
    dict(name='v3 で「Reach/TP 未実施」と書いている',
         pat=r'Pusher\s*のみ[^。\n]{0,20}(?:Reach|TP)[^。\n]{0,8}未実施',
         why='tripo_v3_reach2 が 2026-08-23 に完走（best −4.69 / −5.05）',
         filter='refuted'),
    dict(name='手描き E2E の残る空欄を「E2E 実走のみ」と書いている',
         pat=r'残る空欄[^。\n]{0,30}E2E\s*実走',
         why='2026-09-01〜02 に手描き 3 枚（A1・B1・B2）が M1→M7 まで通った（9-18〜9-19）',
         filter='refuted'),
    dict(name='参考文献リストが「無い」と書いている',
         pat=r'参考文献[^。\n]{0,12}(?:リスト)?[^。\n]{0,8}(?:が)?(?:無い|ない|存在しない|未整備)',
         why='2026-08-28 に 21 件を Crossref / arXiv の一次情報で確定済み。check_citations.py が双方向一致を検査する',
         filter='refuted'),
    dict(name='「系」をシステムの意味で使っている',
         pat=SYSTEM_SENSE,
         why='2026-08-29 に「システム」へ統一した。run 系統の略記・複合語（報酬系 等）は対象外'),
]

# 凍結スナップショットなど、意図的に古いまま残すファイル
FROZEN = {'要旨_詳細版.md'}

# 一次台帳は「各段の当時の状態」を残すのが役目なので、
# 「今は解決済み」型の検査（filter='refuted'）の対象から外す。
# 数値の検査（check_docs_consistency）は従来どおり掛かる。
LEDGER = {'実験系譜.md'}


def main() -> int:
    total = 0
    for p in sorted(DOCS.rglob('*.md')):
        if 'archive' in p.parts:
            continue
        s = p.read_text(encoding='utf-8')
        frozen = p.name in FROZEN
        for c in CHECKS:
            hits = []
            if c.get('filter') == 'refuted' and p.name in LEDGER:
                continue
            for m in re.finditer(c['pat'], s):
                if c.get('exclude') and re.search(c['exclude'], s[max(0, m.start()-60):m.end()+60]):
                    continue
                if c.get('filter') == 'refuted':
                    # 同じ段落に反証の記述があれば「否定した文」なので正常
                    # 段落境界に頼ると「撤回」が次行にある場合を取りこぼすので
                    # 前後 400 字の窓で反証語を探す
                    para = s[max(0, m.start() - 400):m.end() + 400]
                    if REFUTED.search(para):
                        continue
                hits.append(s[max(0, m.start() - 30):m.start() + 40].replace('\n', ' '))
            if not hits:
                continue
            if frozen:
                print(f"  ⏸ {p.relative_to(DOCS)}: {c['name']} {len(hits)} 件"
                      f"（凍結ファイルなので対象外）")
                continue
            total += len(hits)
            print(f"  ❌ {p.relative_to(DOCS)}: {c['name']} {len(hits)} 件")
            print(f"     → {c['why']}")
            for h in hits[:3]:
                print(f"       …{h}…")

    print()
    if total == 0:
        print("✅ 古い主張は検出されなかった。")
        return 0
    print(f"❌ 合計 {total} 件。**同じ主張が他のファイルにも無いか grep で洗うこと**"
          "（CLAUDE.md §8「訂正した主張を記憶を頼りに一部のMDだけ直して終わりにしない」）。")
    return 1


if __name__ == '__main__':
    sys.exit(main())
