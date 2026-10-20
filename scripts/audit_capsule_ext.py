"""**関節の後ろへはみ出したカプセル**（設計変数 `ext_start`）まで含めて、隣り合わないリンクどうしのめり込みを測る（2026-10-20、Bug 72）。

⛔ なぜ要るか: 軌跡（`arm_trace.npz`）はボーン（関節 → 次の関節）と半径しか持たず、`plot_run.py`・
  `check_before_conclusion.py` の「離れた組」・系譜 9-264 の隙間の計算はどれも「カプセルは関節から先にだけ伸びる」と仮定していた。
  実際のカプセルは `fromto` の始点が関節より後ろにあり（`ext_start`、`pusher_gearonly_real` で 0〜0.2 m）、
  配分 dist（1 本目 0.137 m）では `11` の後ろが台座の球に 13 cm 入っていた。検査は 0 mm と出していた。

手順: ① 各 run の最終形態の XML を書き出す（`probe_contact_log.py` を `CONTACT_DUMP_XML` 付きで 1 話・5 step 回す）
      ② この検査で、run の `trace/arm_trace.npz` の全話の姿勢に本当のカプセルを重ねる

    D=<scratch>/xml; for d in single_run/*_phys*; do r=$(basename $d); \\
      EVAL_RESTORE_DIR=$d EVAL_CHECKPOINT=best TRACE_EPISODES=1 TRACE_STEPS=5 TRACE_OUT=$D/$r.npz \\
      CONTACT_LOG_OFF=1 CONTACT_LOG_OUT=$D/$r.contact CONTACT_DUMP_XML=$D/$r.xml USE_CHOREONOID=1 OMP_NUM_THREADS=1 \\
      timeout -k 10 120 /choreonoid_ws/install/bin/choreonoid --no-window --python scripts/probe_contact_log.py > $D/$r.log 2>&1; done
    python3 scripts/audit_capsule_ext.py $D
"""
import sys, glob, os
import numpy as np
import xml.etree.ElementTree as ET


def seg_seg(p1, q1, p2, q2, n=41):
    """2 線分の最短距離（各 n 点の標本。リンク長 1 m で刻み 25 mm、半径の和より十分細かい）"""
    u = np.linspace(0, 1, n)[:, None]
    A = p1 + (q1 - p1) * u
    B = p2 + (q2 - p2) * u
    return float(np.min(np.linalg.norm(A[:, None] - B[None], axis=2)))


def shapes(xml_path):
    """腕の body ごとの (始点, 終点, 半径)（body の局所座標）と親。球は始点＝終点"""
    caps, parent = {}, {}

    def walk(b, par):
        n = b.get('name'); parent[n] = par
        g = b.find('geom')
        if g is not None:
            t, r = g.get('type', 'sphere'), float(g.get('size').split()[0])
            if t == 'capsule':
                ft = np.array(g.get('fromto').split(), float); caps[n] = (ft[:3], ft[3:], r)
            elif t == 'sphere':
                p = np.array((g.get('pos') or '0 0 0').split(), float); caps[n] = (p, p, r)
        for c in b.findall('body'):
            walk(c, n)
    for b in ET.parse(xml_path).getroot().find('worldbody').findall('body'):
        if b.get('name') == '0':
            walk(b, None)
    return caps, parent


def audit(xml_path, trace_path, every=5):
    caps, parent = shapes(xml_path)
    # 後ろへのはみ出し: 始点が関節（局所原点）より後ろにある長さ
    ext = {n: round(float(np.linalg.norm(c[0])), 3) for n, c in caps.items()
           if np.linalg.norm(c[0]) > 1e-6 and c[0] @ (c[1] - c[0]) < 0}
    z = np.load(trace_path, allow_pickle=True)
    names = [str(n) for n in z['body_names']]
    worst = (0.0, None)
    for e in range(int(z['ep_count']) if 'ep_count' in z else 1):
        xp = np.asarray(z['ep_xpos'][e], float); xm = np.asarray(z['ep_xmat'][e], float)
        for k in range(0, len(xp), every):
            W = {n: (xp[k, i] + xm[k, i] @ caps[n][0], xp[k, i] + xm[k, i] @ caps[n][1], caps[n][2])
                 for i, n in enumerate(names) if n in caps}
            ns = list(W)
            for a in range(len(ns)):
                for b in range(a + 1, len(ns)):
                    na, nb = ns[a], ns[b]
                    if parent.get(na) == nb or parent.get(nb) == na:
                        continue          # 隣り合う組は Choreonoid が当たり判定から外す
                    pen = W[na][2] + W[nb][2] - seg_seg(W[na][0], W[na][1], W[nb][0], W[nb][1])
                    if pen > worst[0]:
                        worst = (pen, (na, nb, e, k))
    return ext, worst


def main():
    D = sys.argv[1]
    for xf in sorted(glob.glob(f'{D}/*.xml')):
        r = os.path.basename(xf)[:-4]
        tr = f'single_run/{r}/trace/arm_trace.npz'
        if not os.path.exists(tr):
            continue
        ext, (pen, at) = audit(xf, tr)
        flag = '⛔' if pen > 0.02 else '  '
        print(f'{flag} {r:36s} 後ろへのはみ出し(m) {ext}  隣り合わない組の最大めり込み {pen*1000:6.1f} mm {at[:2] if at else ""}')


if __name__ == '__main__':
    main()
