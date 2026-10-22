#!/usr/bin/env python3
"""**結論を書く前に必ず通す検査**（実験系譜 9-114）。

⚠️⚠️ **なぜ要るか**: CLAUDE.md に書いた規律のうち、
**散文だけのものは 9-114 の 1 週間で 10 件破られ、スクリプトがあるものは 0 件だった。**
⭐ **「md に書く」では守られない。実行できる検査に変換する。**

本スクリプトは、**ある run について結論を書いてよいか**を機械で判定する。

    python3 scripts/check_before_conclusion.py e2e_a1_fix3_auto13
    python3 scripts/check_before_conclusion.py --all        # 完走 run 全部

検査するもの（すべて実際に破った規律）:
  1. **完走しているか**（training done!）
  2. ⭐ **再生して軌跡があるか**（§5-2 ⑤-3。9-96・9-113 で破った）
  3. ⭐ **タスクを達成しているか**（未到達どうしを比較しない。9-113 で破った）
  4. **版が混入していないか**（9-105 で判明。audit_run_validity と同じ判定）
  5. **事前登録があるか**（キューか台帳に「読み」が書かれているか。9-101 で破った）
  6〜7. 目視の記録・execution へ入ったか（9-155・9-211）
  8. ⭐ **再生が学習と同じ物理の環境変数か**（launch_env.json と軌跡を照合。9-229 で破った。9-236）
  9. ⭐ **離れたリンクどうしのめり込みが 50 mm 未満か**（Bug 57。9-232）
  10. ⭐ **縦型アームの台座が倒れていないか**（Bug 58。9-235）
  12. ⭐ **物理的にありえない値が無いか**（`check_invariants.py`。2026-10-12）
  11. ⭐ **比べる相手と起動行が、比べる変数以外で違わないか**（Bug 59。9-238）。組は COMPARISONS。
      任意の 2 本は `--pair A B`（XML と cfg 以外の差を出す）
  13. ⭐ **起動行の robot_param_scale が学習に届いたか**（Bug 74・Bug 75）。記録された値と、
      `log_train.txt` の「⭐ robot_param_scale = …」の行（無い run は当時の config.py が読んだ yml の値）を比べる。
      `--reproduce` で、既知の失敗（凍結が効かなかった R1）と既定値の取り違えを拾えるかを確かめる
"""
import argparse
import re
from functools import lru_cache
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))


def run_dir(r):
    return ROOT / 'single_run' / r


def ck_done(r):
    log = run_dir(r) / 'log' / 'log_train.txt'
    if not log.exists():
        return False, '学習ログが無い'
    return ('training done!' in log.read_text(encoding='utf-8', errors='ignore'),
            '完走' if 'training done!' in log.read_text(encoding='utf-8', errors='ignore') else '未完走')


def ck_trace(r):
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return False, '⛔ 再生していない（§5-2 ⑤-3）。record_arm_trace.py を回すこと'
    return True, f'軌跡あり'


def has_static_body(r):
    """⭐ XML に固定 body（柱・壁）があるか。あると再生ごとに結果が揺れる（9-138）。"""
    import xml.etree.ElementTree as ET
    import yaml
    try:
        cfg = yaml.safe_load(open(run_dir(r) / '.hydra' / 'config.yaml'))
        t = ET.parse(ROOT / 'assets' / 'mujoco_envs' / f"{cfg.get('xml_name')}.xml")
    except Exception:
        return []
    out = []
    for b in t.findall('worldbody/body'):
        n = b.get('name', '')
        if n in ('0',) or b.find('joint') is not None:
            continue          # 腕の根元と、関節を持つ可動体（cube 等）は除く
        if b.find('geom') is not None:
            out.append(n)
    return out


def ck_seen(r):
    """⭐⭐ **図にして目で見たか**（9-155）。

    ⛔⛔ **2026-09-21 まで、36 本の軌跡があって目視の記録は 0 本だった。**
    数値の検査 5 項目を全部通していたのに、**図にしたら 36 本中 10 本でリンクが床より下に出ていた**。
    ⭐ **検査は「見るべきと分かっている量」しか見ない。**
    `docs/研究応用/台帳/目視記録.md` に run 名が書かれているかで判定する。
    """
    log = ROOT / 'docs' / '研究応用' / '台帳' / '目視記録.md'
    png = run_dir(r) / 'trace' / 'view.png'
    if not log.exists():
        return None, '（目視記録.md が無い）'
    txt = log.read_text(encoding='utf-8')
    # ⛔ 2026-10-16（系譜 9-259）: 本文のどこかに `run` があれば通していたので、改名した run の行の注記
    #   「（旧 `tripo_pjp_long_phys`）」で、見ていない 4 本が「見た」になっていた。⭐ 行の最初の `…` だけを run 名とみなす
    seen = {m.group(1) for line in txt.splitlines() for m in [re.match(r'^\|[^`|]*`([^`]+)`', line)] if m}
    if r not in seen:
        hint = ''
        if r.rsplit('_', 1)[0] in txt or r.split('_')[0] in txt:
            hint = ('  ⚠️ **似た名前はある。略記（`_s2` 等）で書いていないか確認すること** — '
                    '⛔ **略記は検査が拾えず「見たのに見ていない」と出る**（2026-09-22 に 11 本で発生）')
        return False, (f'⛔ **目視記録に無い。**`python3 scripts/plot_run.py {r}` で図を作り、'
                       f'**実際に見てから** `docs/研究応用/台帳/目視記録.md` へ '
                       f'**フルネームで** 1 行書くこと（9-155）' + hint)
    if not png.exists():
        return None, '⚠️ 記録はあるが図が無い（`plot_run.py` で作り直せる）'
    return True, '目視記録あり'


def ck_achieved(r):
    """⭐ タスクを達成しているか。未到達どうしの比較を防ぐ（9-113）。"""
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    if not p.exists():
        return None, '（再生していないので判定不能）'
    d = np.load(p)
    # ⛔⛔ 9-138: 固定 body と接触するタスクは、同じ checkpoint でも再生ごとに揺れる
    #   （ホッケー 64〜308 mm・障害物 254 %）。1 エピソードで達成を判定してはいけない。
    static = has_static_body(r)
    # ⛔⛔ Bug 47 / 9-142: 軌跡は「読み込み直後の 1 話目」であり、その 1 話は当てにならない
    #   （対照 Pusher で 1 話目 15.11 m 対 3〜6 話 6.54〜15.97 m）。
    #   ⚠️ 反復しても検出できない（毎回 1 話目をやり直すだけ）。
    #   ⭐ 二値の到達判定は 1 話目でも変わらないことが多いが、**量の比較には使えない**。
    # ⭐ Bug 47 / 9-142 の対応後に作られた軌跡は、頭の話を捨てた複数話の要約を持つ。
    if 'ep_cube_dx' in d:
        dx = np.asarray(d['ep_cube_dx'], dtype=float)
        _sk = int(d["ep_skipped"])
        warn = (f'  ⭐ **{len(dx)} 話' + (f'（頭 {_sk} 話を捨てた）**' if _sk else '（全話）**')
                + f'  話ごとの対象移動 [{dx.min():.2f}, {dx.max():.2f}] m ')
        if len(dx) > 1 and dx.max() - dx.min() > 0.2 * max(abs(dx.mean()), 1e-9):
            warn += (f'⚠️ **話ごとの幅が {(dx.max()-dx.min())/max(abs(dx.mean()),1e-9)*100:.0f} % ある。'
                     f'単一の値で順位を付けないこと** ')
    else:
        warn = ('  ⛔ **この軌跡は Bug 47 の対応前に作られており「読み込み直後の 1 話目」である。'
                '取り直すこと**（`record_arm_trace.py` は既定で 5 話走らせ、全話を使う。9-237） ')
    warn += ('' if not static else
            f"  ⛔ **固定 body {static} がある。再生ごとに結果が揺れる（9-138）。"
            f"1 エピソードで判定しないこと** ")
    # ⛔ 9-137: 打ち切られた軌跡で最終位置・到達を語らない。
    #   ⭐ ただし「末尾でまだ動いている」ときだけ警告する。
    #     対象が止まっていれば打ち切りは無害で、毎回出る警告は読まれなくなる（CLAUDE.md §5-2 ①）。
    ov = (run_dir(r) / '.hydra' / 'overrides.yaml')
    txt = ov.read_text(encoding='utf-8') if ov.exists() else ''
    is_reach = 'use_reach=true' in txt
    xp, xm, bo = d['xpos'], d['xmat'], d['bone_offset']
    tip = xp[:, -1, :] + np.einsum('tij,j->ti', xm[:, -1], bo[-1])
    # ⛔ 9-137: 打ち切られた軌跡で最終位置・到達を語らない。
    #   ⭐ ただし **判定に使う量そのもの**が末尾でまだ動いているときだけ警告する。
    #     Pusher は対象が 400 step で止まるので、腕の動きで警告を出すと毎回出て読まれなくなる
    #     （CLAUDE.md §5-2 ①「毎回同じ件数が出る検査は読まれなくなる」）。
    watched = tip if is_reach else d.get('cube', tip)
    n = len(watched)
    tail = watched[int(n * 0.9):]
    moving = float(np.abs(np.diff(tail, axis=0)).max()) if len(tail) > 1 else 0.0
    # ⛔⛔ 2026-09-25: **エピソードが 1 step で終わっている軌跡があった**
    #   （`tripo_v3_reach` / `_smoke` / `rrbot_arm_pusher_fixmorph_s1`）。
    #   ⭐ **図にして step=1 と題に出ていたので気づいた。**数値の検査は全部通っていた。
    #   ⚠️ **この軌跡からは何も言えない。**初期姿勢が写っているだけである。
    if n < 50:
        warn += (f'  ⛔⛔ **エピソードが {n} step で終わっている。初期姿勢が写っているだけで、'
                 f'この軌跡からは何も言えない**（2026-09-25） ')
    if n >= 1200 and moving > 1e-3:
        warn += (f'  ⚠️ **{n} step で打ち切られ、判定対象が末尾でもまだ動いている'
                 f'（{moving*1000:.1f} mm/step）。TRACE_STEPS を増やすこと（9-137）** ')
    if is_reach:
        tgt = d['target']
        dist = np.linalg.norm(tip - tgt, axis=1)
        if dist.min() < 0.01:
            return True, f'到達（最小 {dist.min()*1000:.0f} mm）' + warn
        return False, (f'⛔ **未到達**（最小 {dist.min()*1000:.0f} mm・最終 {dist[-1]*1000:.0f} mm）。'
                        f'⚠️ **未到達どうしの差を「良化」と呼ばないこと**（9-113）' + warn)
    c = d['cube']
    moved = abs(c[-1, 0] - c[0, 0])
    if moved < 1e-3:
        return False, f'⛔ **対象が動いていない**（{moved*1000:.1f} mm）' + warn
    return True, f'対象が {moved:.3f} m 動いた' + warn


@lru_cache(maxsize=None)
def _audit_output():
    """audit_run_validity.py は全 run を 1 回で監査する（約 8.5 秒）。--all で run ごとに回していたので 1 回だけにする（2026-10-19）。"""
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / 'audit_run_validity.py')],
                          capture_output=True, text=True, cwd=ROOT).stdout


def ck_version(r):
    """同じ xml_name の他 run と版が違わないか（9-105）。"""
    out = _audit_output()
    seg = out.split('同じ xml_name なのに中身が違う組')[-1]
    for block in seg.split('⛔ ')[1:]:
        if re.search(rf'^\s+{re.escape(r)}\s', block, re.M):
            xml = block.split(':')[0]
            return False, f'⛔ **{xml} に版の混入あり**。同じ版の run とだけ比較すること（9-105）'
    return True, '版の混入なし'


def ck_prereg(r):
    """事前登録（読み）がキューか台帳にあるか。"""
    for q in (ROOT / 'scripts').glob('queue_*.sh'):
        t = q.read_text(encoding='utf-8')
        if r in t and ('読み' in t or '停止条件' in t):
            return True, f'{q.name} に読みあり'
    led = (ROOT / 'docs/研究応用/台帳/実験系譜.md').read_text(encoding='utf-8')
    for m in re.finditer(r'^### .*$', led, re.M):
        seg = led[m.start():m.start() + 6000]
        if r in seg and ('事前' in seg or '読み' in seg):
            return True, '台帳に読みあり'
    return False, '⚠️ 事前登録が見つからない（結果を見てから読み方を決めていないか）'


def ck_entered_exec(r):
    """⭐⭐⭐ **execution フェーズへ本当に入ったか**（2026-10-03、系譜 9-211）。

    ⛔⛔ **`hockey_bank7` は 20 epoch すべて `exec_R_eps` が厳密に 0.00 だった。**
      初期姿勢の探索が全滅して `transit_execution()` が例外で落ち、
      **1 エピソードも execution へ入っていなかった。**
      ⛔ だが裸の `except:` が例外を握りつぶしていたので、ログは「学習が進んでいない」に見えた。
      ⭐ 既知良好の `hockey_bank6` は同じ ep18 で **−0.38** だった。

    ⚠️⚠️ **他の 6 項目はどれもこれを検出できない。**完走するし、再生もでき、
      軌跡ファイルも作られる（中身が空でも）。⭐ **「厳密に 0.00 が続く」が署名である。**

    ⚠️ 0.00 が 1〜2 epoch なら偶然もありうるので、**5 epoch 以上続いたら赤**にする。
    """
    log = run_dir(r) / 'log' / 'log_train.txt'
    if not log.exists():
        return False, '学習ログが無い'
    import re as _re
    vals = _re.findall(r'exec_R_eps\s+(-?[\d.]+)', log.read_text(encoding='utf-8', errors='ignore'))
    if not vals:
        return None, '⚠️ exec_R_eps が読めない（古い形式？）'
    zeros = sum(1 for v in vals if abs(float(v)) == 0.0)
    if zeros == len(vals) and len(vals) >= 5:
        return False, (f'⛔⛔ **exec_R_eps が {len(vals)} epoch すべて厳密に 0.00。**'
                       f'1 エピソードも execution へ入っていない疑い（9-211）。'
                       f'⭐ stdout を `grep arm_safe_init` し、`transit_execution` の例外を見ること')
    return True, f'exec_R_eps が非ゼロの epoch {len(vals)-zeros}/{len(vals)}'


def _launch_env(r):
    p = run_dir(r) / 'launch_env.json'
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text(encoding='utf-8'))


def ck_replay_env(r):
    """⭐ 系譜 9-236: **再生が学習と同じ物理の環境変数で行われたか**。

    ⛔ 9-229 で `HOCKEY_*` を渡し忘れ、別の物理で再生して結論を書きかけた。
    """
    le = _launch_env(r)
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    if le is None:
        return None, '⚠️ launch_env.json が無い（9-236 より前の run）。再生の環境は照合できない'
    if not p.exists():
        return False, '⛔ 再生していない'
    import json
    d = np.load(p, allow_pickle=True)
    if 'replay_env' not in d.files:
        return False, '⛔ 軌跡に再生の環境が記録されていない（9-236 より前の再生）。取り直すこと'
    re_ = json.loads(str(d['replay_env']))
    diff = {k: (le.get(k), re_.get(k)) for k in set(le) | set(re_) if le.get(k) != re_.get(k)}
    if diff:
        return False, f'⛔ 学習と再生で物理の環境変数が違う: {diff}'
    return True, f'学習と同じ環境で再生（{le}）'


def ck_far_pen(r):
    """⭐ 系譜 9-227・9-232: **離れたリンクどうしのめり込みが 50 mm 未満か**（Bug 57 が直っているか）。"""
    if not (run_dir(r) / 'trace' / 'arm_trace.npz').exists():
        return False, '⛔ 再生していない'
    sys.path.insert(0, str(ROOT / 'scripts'))
    from check_self_penetration import measure
    w, fr, pr, T, far = measure(r)
    msg = f'離れた組の最大めり込み {far[0]*1000:.0f} mm（step の {far[1]*100:.1f} %{"、" + "×".join(far[3]) if far[3] else ""}）'
    # ⭐ Bug 72: 古い軌跡はカプセルの始点を持たず、関節の後ろへのはみ出しが見えない
    if not far[2]:
        msg += ('  ⚠️ 軌跡にカプセルの始点が無い（2026-10-20 より前の再生）。関節の後ろへのはみ出しは見ていない'
                '（`audit_capsule_ext.py` で見る）')
    le = _launch_env(r) or {}
    if not le.get('CNOID_SELF_COLLISION'):
        return None, f'⚠️ 自己干渉なしで学習（または記録なし）。{msg}'
    if far[0] >= 0.05:
        return False, f'⛔ {msg}。5 cm を越えている＝接触が捨てられた疑い（Bug 57）'
    return True, msg


def ck_fold(r):
    """⭐ 系譜 9-287: **隣り合うリンクを 120° を超えて折り返す姿勢を使っていないか**（同じ平面に並べた実機では当たる）。

    学習は隣り合う組を判定しない（関節で必ず重なる）ので、ここは止めずに ⚠️ を出す。使っていれば、結論に
    「実機では段違いの取り付けが要る」を添える（禁じたときの成績は学習し直さないと分からない）。
    """
    if not (run_dir(r) / 'trace' / 'arm_trace.npz').exists():
        return False, '⛔ 再生していない'
    sys.path.insert(0, str(ROOT / 'scripts'))
    from check_self_penetration import fold, FOLD_DEG
    mx, fr, pr, ne = fold(r)
    msg = f'隣り合う組の最大の曲げ {mx:.1f}°、{FOLD_DEG:.0f}° 超は全 {ne} 話の step の {fr*100:.1f} %'
    if fr > 0.01:
        return None, (f'⚠️ {msg}（{"×".join(pr)}）。同じ平面の実機では当たる姿勢を使っている → 結論に'
                      '「実機では段違いの取り付けが要る」を添える（修論 5.3 (C)）')
    return True, msg


def ck_base(r):
    """⭐ 系譜 9-235・Bug 58: **縦型アームの台座が倒れていないか**。"""
    import json as _j
    p = run_dir(r) / 'trace' / 'arm_trace.npz'
    cfgp = run_dir(r) / '.hydra' / 'config.yaml'
    if not p.exists() or not cfgp.exists():
        return None, '⚠️ 軌跡か設定が無い'
    import yaml
    c = yaml.safe_load(cfgp.read_text(encoding='utf-8'))
    xp = ROOT / 'assets' / 'mujoco_envs' / f"{c.get('xml_name')}.xml"
    m = re.search(r'<body name="1"[^>]*>.*?fromto="([^"]+)"', xp.read_text(encoding='utf-8'), re.S) if xp.exists() else None
    if not m:
        return None, '⚠️ 台座の形を XML から読めない'
    ft = [float(x) for x in m.group(1).split()]
    d0 = np.array(ft[3:]) - np.array(ft[:3])
    if abs(d0[2]) < 0.9 * np.linalg.norm(d0):
        return True, '横型（柱の台座なし）'
    d = np.load(p, allow_pickle=True)
    names = [str(n) for n in d['body_names']]
    b = np.asarray(d['ep_bone'][0] if 'ep_bone' in d.files else d['bone_offset'], float)[names.index('1')]
    tilt = float(np.degrees(np.arccos(np.clip(b[2] / np.linalg.norm(b), -1, 1))))
    if c.get('fix_base'):
        return (tilt < 0.5), (f'台座 {tilt:.1f}°（fix_base あり）' if tilt < 0.5 else f'⛔ fix_base なのに台座が {tilt:.1f}° 傾いている')
    return (None if tilt > 10 else True), (f'⚠️ 縦型で台座が {tilt:.1f}° 倒れている（Bug 58。fix_base なし）' if tilt > 10 else f'台座 {tilt:.1f}°')


# ⭐ 11. 比べる相手と起動行が「比べる変数」以外で違わないか（Bug 59・系譜 9-238）。
#   ⛔ 9-136 は XML だけ diff して「1 変数」とし、起動行の報酬の違いを見落とした。
#   ⭐ 修論が比べる組をここに書く（名前は seed0。_s2 どうしも同じ組として見る）。
#   各組: (名前, run の列, 違ってよいキー)。'TASK' は到達 対 押し＝報酬のキーは違って当然なので COMMON だけ比べる。
COMPARISONS = [
    # ⭐ 9-246: 実在の物理で全部やり直した *_phys（それ以前の *_scgc は 9-232〜9-245）
    ('押しの柱（4.4.3）', ['tripo_pjp_long_phys', 'tripo_pjp_mid_phys', 'tripo_pjdp_prox_phys', 'tripo_pjdp_dist_phys'], {'xml_name'}),
    ('armature 1.71 対 1（long）', ['tripo_pjp_long_phys', 'tripo_pjp_long_phys_arm1'], {'xml_name'}),
    ('armature 1.71 対 1（mid）', ['tripo_pjp_mid_phys', 'tripo_pjp_mid_phys_arm1'], {'xml_name'}),
    # ⛔ Bug 74: tripo_pjp_mid_frozen_phys は凍結が効いていなかった（mid と全 epoch 一致）ので組から外し、_fx で取り直す（9-281）。
    #   same_curve はこの旧い組で「200 epoch すべて一致」を拾うことを 10-22 に確かめた（組に残すと正しい mid まで ⛔ になる）
    ('R1 形の凍結（4.4.1）', ['tripo_pjp_mid_phys', 'tripo_pjp_mid_frozen_phys_fx'], {'robot_param_scale'}),
    ('R2 縦型の到達 対 押し', ['e2e_a1v_reach_phys', 'e2e_a1v_pusher_phys'], 'TASK'),
    # ⭐ Bug 72（10-21）: 短腕も 11 のはみ出しで台座に食い込んでいたので、門を直して取り直す（queue_short_capgate.sh）
    ('到達マトリクス（4.4.3.2）', ['tripo_pj_mid_phys', 'tripo_pj_long_phys', 'tripo_pj_short_phys_cg'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('短腕 到達 門の修正の前後（Bug 72）', ['tripo_pj_short_phys', 'tripo_pj_short_phys_cg'], {'env_specs.init_self_contact_caps'}),
    ('マトリクス 到達 対 押し（4.4.3.2）', ['tripo_pj_mid_phys', 'tripo_pjp_mid_phys'], 'TASK'),
    ('配分・到達（4.4.3.3）', ['tripo_pjd_prox_phys', 'tripo_pjd_dist_phys'], {'xml_name'}),
    # ⭐ Bug 72: 門を本当のカプセルで測る取り直し（queue_dist_capgate.sh）。prox は構造上食い込めないので設定の有無で環境は同じ
    ('配分・到達 門の修正後（4.4.3.3、Bug 72）', ['tripo_pjd_prox_phys', 'tripo_pjd_dist_phys_cg'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('配分・押し 門の修正後（Bug 72）', ['tripo_pjdp_prox_phys', 'tripo_pjdp_dist_phys_cg'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('dist 到達 門の修正の前後（Bug 72）', ['tripo_pjd_dist_phys', 'tripo_pjd_dist_phys_cg'], {'env_specs.init_self_contact_caps'}),
    ('閉ループ 1.99 倍（4.5）', ['tripo_pjr199_phys', 'tripo_pj_short_phys_cg'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('閉ループ 関節固定（4.5.5）', ['e2e_a1_reach_phys', 'e2e_a1_fix3_reach_phys'], {'xml_name'}),
    ('付録D 到達 対 押し', ['tripo_v3_reach2_phys', 'tripo_arm_v3_pusher_phys'], 'TASK'),
    ('マトリクス 押し（4.4.3.2）', ['tripo_pjp_short_phys_cg', 'tripo_pjp_mid_phys', 'tripo_pjp_long_phys'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('閉ループ 2.11 倍 押し（4.5.4）', ['tripo_pjr211p_phys', 'tripo_pjp_short_phys_cg'], {'xml_name', 'env_specs.init_self_contact_caps'}),
    ('閉ループ 関節固定 押し（4.5.5）', ['e2e_a1_pusher_phys', 'e2e_a1_fix3_pusher_phys'], {'xml_name'}),
    ('関節固定 到達 対 押し（4.5.5）', ['e2e_a1_reach_phys', 'e2e_a1_pusher_phys'], 'TASK'),
]
# 到達 対 押しでも同じでなければならないキー
COMMON = {'reward_specs.ctrl_cost_coeff', 'fix_base', 'robot_param_scale', 'fix_skeleton',
          'env_specs.init_self_contact', 'max_epoch_num', 'num_threads', 'launch_env'}
# 指定が無いときにコードが使う値（pusher.py）。⛔ Bug 59 は「書いていない＝既定値」が見えなかった
# ⛔ Bug 75: これは yml にも無いときの値。run の cfg の yml にあればそちらが効く（_yml_defaults）
DEFAULTS = {'reward_specs.ctrl_cost_coeff': '0.0001', 'reward_specs.contact_weight': '1.0',
            'reward_specs.init_contact_penalty': '50.0', 'env_specs.arm_safe_init': 'False',
            'fix_base': 'False', 'env_specs.init_self_contact': 'False', 'robot_param_scale': '0.1'}
# ⭐ 承知の上で残した差（理由を必ず書く）
ACCEPTED = {}   # 9-238 の実在の到達の arm_safe_init は *_phys では無くなった（全部の到達がそろう）
IGNORE = {'seed', 'hydra.run.dir', 'enable_wandb'}


def _norm(v):
    try:
        return repr(float(v))
    except ValueError:
        return str(v).lower()


def _raw_overrides(r):
    import yaml
    p = run_dir(r) / '.hydra' / 'overrides.yaml'
    if not p.exists():
        return None
    o = {}
    for line in yaml.safe_load(p.read_text(encoding='utf-8')) or []:
        k, _, v = str(line).partition('=')
        o[k.lstrip('+~')] = v
    # ⭐ 自分自身からの再開（restore_dir が自分のフォルダ）は条件の違いではないので外す
    if o.get('restore_dir', '').rstrip('/').endswith('/' + r):
        o.pop('restore_dir'); o.pop('epoch', None)
    return o


@lru_cache(maxsize=None)
def _yml_defaults(cfg):
    """⭐ Bug 75: 起動行に無いキーで効く値のうち、cfg の yml が決めるもの（config.py と同じ読み方）。"""
    import yaml
    p = ROOT / 'design_opt' / 'cfg' / f'{cfg}.yml'
    if not p.exists():
        return {}
    y = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    out = {}
    if 'robot_param_scale' in y:
        out['robot_param_scale'] = y['robot_param_scale']
    if (y.get('robot') or {}).get('fix_base') is not None:
        out['fix_base'] = y['robot']['fix_base']
    for s in ('reward_specs', 'env_specs'):
        for k, v in (y.get(s) or {}).items():
            out[f'{s}.{k}'] = v
    return out


_EFF_SCALE = re.compile(r'⭐ robot_param_scale = (\S+?)（')


def _effective_scale(r, o):
    """学習に効いた robot_param_scale と、その出どころ。

    Bug 74 の修正（c86c04e）より後の run は train.py が ⭐ の行を出す。その行が無い run は、
    当時の config.py が起動行を捨てて yml だけを読んでいた。r=None は修正後のコードで走らせた場合。
    """
    yml = _yml_defaults(o.get('cfg', ''))
    log = run_dir(r) / 'log' / 'log_train.txt' if r else None
    if log is not None and log.exists():
        m = _EFF_SCALE.search(log.read_text(encoding='utf-8', errors='ignore'))
        if m:
            return m.group(1), 'log_train.txt'
        return yml.get('robot_param_scale', 0.1), 'yml（Bug 74 の修正前の run）'
    return o.get('robot_param_scale', yml.get('robot_param_scale', 0.1)), '起動行か yml'


def _resolve(o, r=None):
    """起動行の上書きを、効いた値の辞書にする: コードの既定 → cfg の yml → 起動行。robot_param_scale は効いた値。"""
    full = {k: _norm(v) for k, v in DEFAULTS.items()}
    full.update({k: _norm(v) for k, v in _yml_defaults(o.get('cfg', '')).items()})
    full.update({k: _norm(v) for k, v in o.items()})
    full['robot_param_scale'] = _norm(_effective_scale(r, o)[0])
    return full


def _overrides(r):
    o = _raw_overrides(r)
    if o is None:
        return None
    full = _resolve(o, r)
    full['launch_env'] = str(sorted((_launch_env(r) or {}).items()))
    return full


def diff_runs(a, b, allowed):
    """a と b の起動行で、allowed 以外に違うキーを返す。allowed='TASK' なら COMMON だけ比べる。"""
    oa, ob = _overrides(a), _overrides(b)
    if oa is None or ob is None:
        return None
    keys = COMMON if allowed == 'TASK' else (set(oa) | set(ob)) - IGNORE - set(allowed)
    out = []
    for k in sorted(keys):
        va, vb = oa.get(k, '(なし)'), ob.get(k, '(なし)')
        if va != vb and (a, k) not in ACCEPTED and (b, k) not in ACCEPTED:
            out.append(f'{k}: {va} 対 {vb}')
    return out


def _exec_curve(r):
    log = run_dir(r) / 'log' / 'log_train.txt'
    return re.findall(r'exec_R_eps\s+(-?[\d.]+)', log.read_text(encoding='utf-8', errors='ignore')) if log.exists() else []


def same_curve(a, b, n_min=5):
    """⛔ Bug 74（系譜 9-281）: 同じ seed で比べる変数だけ違うのに exec_R_eps が全 epoch 一致 ＝ その変数が学習に届いていない。"""
    ca, cb = _exec_curve(a), _exec_curve(b)
    n = min(len(ca), len(cb))
    return n >= n_min and ca[:n] == cb[:n], n


def ck_pair(r):
    base, s2 = (r[:-3], '_s2') if r.endswith('_s2') else (r, '')
    msgs, bad, n = [], False, 0
    if s2 and (run_dir(base) / 'log').exists():          # seed の対は seed 以外同じのはず
        d = diff_runs(base, r, set())
        if d:
            bad = True
            msgs.append(f'seed0 と: {"; ".join(d)}')
    for name, runs, allowed in COMPARISONS:
        if base not in runs:
            continue
        for o in runs:
            if o == base or not (run_dir(o + s2) / 'log').exists():
                continue
            d = diff_runs(r, o + s2, allowed)
            n += 1
            if d:
                bad = True
                msgs.append(f'{name} の {o + s2} と: {"; ".join(d)}')
            same, m = same_curve(r, o + s2)
            if same:
                bad = True
                msgs.append(f'⛔⛔ {name} の {o + s2} と exec_R_eps が {m} epoch すべて一致（比べる変数が学習に届いていない、Bug 74）')
    if bad:
        return False, '⛔ 比べる相手との比較が成り立たない（起動行の違い Bug 59・曲線の一致 Bug 74）: ' + ' ／ '.join(msgs)
    return True, (f'比べる相手 {n} 本と、比べる変数以外は同じ' if n else '比べる相手が無い・まだ無い（COMPARISONS に無い組は見ない）')


def ck_invariants(r):
    """⭐ 12. 物理的にありえない値（`check_invariants.py`。報酬の桁・制御入力・触れる前／置いた瞬間・床・壁）。"""
    from check_invariants import run as inv
    res = inv(r)
    bad = [f'{n}: {m}' for n, ok, m in res if ok is False]
    warn = [f'{n}: {m}' for n, ok, m in res if ok is None]
    if bad:
        return False, ' ／ '.join(bad)
    if warn:
        return None, ' ／ '.join(warn)
    return True, f'{len(res)} 項目とも成り立つ'


def ck_effective(r):
    """⭐ 13. 起動行（無ければ yml）の robot_param_scale と、学習に効いた値が同じか（Bug 74・Bug 75）。"""
    o = _raw_overrides(r)
    if o is None:
        return None, '⚠️ 起動行の記録（.hydra）が無い'
    rec = o.get('robot_param_scale', _yml_defaults(o.get('cfg', '')).get('robot_param_scale', 0.1))
    eff, src = _effective_scale(r, o)
    if _norm(rec) != _norm(eff):
        return False, f'⛔ 指定は {rec}、効いた値は {eff}（{src}）。起動行が学習に届いていない（Bug 74）'
    return True, f'robot_param_scale {eff}（{src}）'


CHECKS = [
    ('完走', ck_done), ('再生', ck_trace), ('⭐⭐ 目視', ck_seen), ('⭐ 達成', ck_achieved),
          ('版', ck_version), ('事前登録', ck_prereg),
          ('⭐⭐ exec 到達', ck_entered_exec),
          ('⭐ 再生の環境', ck_replay_env), ('⭐ 離れた組', ck_far_pen), ('⭐ 折り返し', ck_fold), ('⭐ 台座', ck_base),
          ('⭐ 比べる相手', ck_pair), ('⭐ 不変条件', ck_invariants), ('⭐ 効いた値', ck_effective)]


def reproduce():
    """既知の失敗を拾えるか（拾えない検査の ✅ は根拠にならない）。"""
    cases = [
        ('凍結が効かなかった R1（Bug 74）を ⛔', ck_effective('tripo_pjp_mid_frozen_phys')[0] is False),
        ('凍結しない mid は ✅', ck_effective('tripo_pjp_mid_phys')[0] is True),
        # ⛔ Bug 75: yml が 1 の cfg で「0.1 を明示」と「指定なし」を同じと見ていた
        ('明示の 0.1 と指定なし（yml 1）を違うと見る',
         _resolve({'cfg': 'pusher_gearonly_real', 'robot_param_scale': '0.1'})['robot_param_scale']
         != _resolve({'cfg': 'pusher_gearonly_real'})['robot_param_scale']),
        ('到達の長腕 seed1 の手首の折り返し（9-287）を ⚠️', ck_fold('tripo_pj_long_phys_s2')[0] is None),
        ('折り返さない中間は ✅', ck_fold('tripo_pj_mid_phys')[0] is True),
        ('R1 の旧い組は効いた値が同じ（=比べる変数が無い）',
         _overrides('tripo_pjp_mid_frozen_phys')['robot_param_scale'] == _overrides('tripo_pjp_mid_phys')['robot_param_scale']),
    ]
    for name, ok in cases:
        print(f'  {"✅" if ok else "⛔"} {name}')
    return 0 if all(ok for _, ok in cases) else 1


def audit(r):
    print(f'\n=== {r}')
    ng = 0
    for name, fn in CHECKS:
        ok, msg = fn(r)
        mark = '✅' if ok else ('⚠️' if ok is None else '⛔')
        if ok is False:
            ng += 1
        print(f'  {mark} {name:<8} {msg}')
    print(f'  → {"⛔ 結論を書く前に上を直すこと" if ng else "✅ 結論を書いてよい"}')
    return ng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--all', action='store_true', help='完走した run 全部')
    ap.add_argument('--pair', nargs=2, metavar=('A', 'B'), help='2 本の起動行を比べる（xml_name・cfg 以外の差）')
    ap.add_argument('--reproduce', action='store_true', help='既知の失敗（Bug 74・75）を拾えるかを確かめる')
    a = ap.parse_args()
    if a.reproduce:
        return reproduce()
    if a.pair:
        d = diff_runs(a.pair[0], a.pair[1], {'xml_name', 'cfg'})
        print('\n'.join(d) if d else ('起動行が読めない' if d is None else '✅ xml_name・cfg 以外は同じ'))
        return 1 if d else 0
    runs = a.runs
    if a.all:
        runs = sorted(d.name for d in (ROOT / 'single_run').iterdir()
                      if d.is_dir() and (d / 'log' / 'log_train.txt').exists()
                      and 'training done!' in (d / 'log' / 'log_train.txt').read_text(
                          encoding='utf-8', errors='ignore'))
    if not runs:
        ap.error('run 名か --all を指定してください')
    tot = sum(audit(r) for r in runs)
    print(f'\n合計 {tot} 件の要対応')
    return 0


if __name__ == '__main__':
    sys.exit(main())
