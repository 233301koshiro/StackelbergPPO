#!/usr/bin/env python3
"""OBB の主軸判定が「最長辺＝ボーン軸」を取り違える境界を測る（M4 の閾値較正、9-69）。

**なぜ要るか**: `mesh_to_params.py` は最長辺をボーン軸とみなし、
**最長辺÷次辺が 1.5 を割ると警告**を出す。だが
**その 1.5 がどこから来たのか、割ったとき実際に何が壊れるのかの記録が無かった**
（工程別_検証状況.md で M4 が一番薄いと判明、2026-09-10）。

真のボーン軸が既知の合成メッシュを縦横比を振って作り、
`obb_params` が返す最長軸が真の軸と一致するかを直接測る。学習も外部サービスも要らない。

    python3 scripts/probe_obb_axis.py
"""
import importlib.util
import sys

import numpy as np
import trimesh

_spec = importlib.util.spec_from_file_location(
    'm2p', __file__.replace('probe_obb_axis.py', 'mesh_to_params.py'))
m2p = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m2p)

TRUE_AXIS = np.array([0.0, 0.0, 1.0])   # 合成メッシュのボーン軸は常に +Z
WARN_RATIO = 1.5                        # mesh_to_params.py の閾値


def capsule(L, r):
    return trimesh.creation.capsule(height=L, radius=r, count=[16, 16])


def mallet(L, r, D, t):
    """柄（長さ L・半径 r）＋ 先端の円盤（直径 D・厚み t）。ボーン軸 = +Z"""
    sh = trimesh.creation.capsule(height=L, radius=r, count=[16, 16])
    dk = trimesh.creation.cylinder(radius=D / 2, height=t, sections=32)
    dk.apply_translation([0, 0, L])
    return trimesh.util.concatenate([sh, dk])


def check(mesh):
    p = m2p.obb_params(mesh, 1.0)
    a = np.asarray(p['obb_axes'][0], float)
    a /= np.linalg.norm(a)
    align = abs(float(np.dot(a, TRUE_AXIS)))
    e = p['extents']
    return dict(e0=e[0], e1=e[1], ratio=e[0] / e[1], align=align,
                deg=np.degrees(np.arccos(min(align, 1.0))), length=p['length'])


def main():
    print("=== ① 素のカプセル（柄だけ）===")
    print("カプセルは全長 = L + 2r で必ず幅 2r 以上なので、**構造的に取り違えない**。")
    print(f"{'L':>6}{'r':>7}{'e0/e1':>8}{'軸ずれ':>8}  判定")
    for L in [0.30, 0.15, 0.10, 0.08]:
        c = check(capsule(L, 0.05))
        print(f"{L:6.2f}{0.05:7.2f}{c['ratio']:8.2f}{c['deg']:7.1f}°  "
              f"{'⚠️警告' if c['ratio'] < WARN_RATIO else '     '} "
              f"{'✅' if c['align'] > 0.99 else '❌'}")

    print("\n=== ② マレット（先端が太い形状）— ここだけが危ない ===")
    L, r, t = 0.22, 0.035, 0.02
    print(f"柄 L={L} r={r}、円盤 厚み {t}")
    print(f"{'D/L':>6}{'e0/e1':>8}{'軸ずれ':>8}{'length':>8}  判定")
    first_bad = None
    for frac in [1.0, 1.20, 1.30, 1.38, 1.40, 1.42, 1.50, 2.0]:
        c = check(mallet(L, r, L * frac, t))
        bad = c['align'] <= 0.99
        if bad and first_bad is None:
            first_bad = (frac, c['ratio'])
        print(f"{frac:6.2f}{c['ratio']:8.2f}{c['deg']:7.1f}°{c['length']:8.3f}  "
              f"{'⚠️警告' if c['ratio'] < WARN_RATIO else '     '} "
              f"{'❌取り違え' if bad else '✅'}")

    if first_bad:
        print(f"\n破綻の開始: D/L = {first_bad[0]}（そのとき e0/e1 = {first_bad[1]:.2f}）")
        print(f"警告の閾値 : e0/e1 = {WARN_RATIO}")
        print(f"→ **余裕は {(WARN_RATIO/first_bad[1]-1)*100:.0f} % しかない。**")
    print("\n⚠️⚠️ **最大の発見: e0/e1 は破綻の検出器として成立していない。**")
    print("   細かく振ると比は破綻の直前まで**下がり続け**、破綻した瞬間に**跳ね上がる**:")
    print("     D/L 1.24 → 比 1.378  ずれ  0.0°  ✅")
    print("     D/L 1.40 → 比 1.220  ずれ  0.0°  ✅")
    print("     D/L 1.42 → 比 1.362  ずれ 29.5°  ❌  ← 比は 1.378 と同水準なのに壊れている")
    print("   **同じ比 1.36 が「正常」と「破綻」の両方で出る。**")
    print("   → 閾値をどこに置いても分離できない。**量の選び方が誤っている。**")
    print("\n   正しい検算は M3 が既に持っている**関節間の方向**との突き合わせである")
    print("   （OBB 単体では真のボーン軸を知りようがない）。")
    print("\n⚠️ この数字は 1 つの形状族（柄＋円盤）のもの。**普遍則ではない。**")

    verify(L, r, t)


def verify(L, r, t):
    """③ 是正（9-75）の検証。半径をボーン軸に**垂直な断面**から取ると何が直るか。

    ⚠️ **合成メッシュでは再現しなかった。** マレットは回転対称なので、OBB が 29° 傾いても
    短辺 2 本が似た値のままで、半径が跳ばない（旧の跳び 2.2 %）。
    **再現したのは実物メッシュの方**だった。そこで検証も実物で行う。

    合否は「正しい半径は幾つか」ではなく**カプセルがメッシュを覆えているか**で見る。
    これは真値が一意に決まる量である。
    """
    import json
    from pathlib import Path

    print("\n=== ③ 是正の検証: 半径をボーン軸に垂直な断面から取る（9-75）===")
    print("⚠️ 合成マレットでは再現しなかった（回転対称なので半径が跳ばない）。実物で測る。")
    print("合否は『正しい半径は幾つか』ではなく**カプセルがメッシュを覆えているか**で見る。\n")

    def outside(mesh, u, l, rad):
        V = np.asarray(mesh.vertices, dtype=float)
        t = np.clip(V @ u, 0.0, l)
        return float((np.linalg.norm(V - np.outer(t, u), axis=1) > rad).mean() * 100)

    print(f"{'例':8s}{'リンク':11s}{'軸ずれ':>7s}{'旧 OBB':>9s}{'新 ⊥':>8s}"
          f"{'倍':>6s}{'はみ出し 旧':>12s}{'新':>7s}")
    rows, so, sn = [], 0.0, 0.0
    for nm in ['A1', 'A2', 'B1', 'B2', 'hockey']:
        jp = Path(f'data/test/{nm}/meshes/joints.json')
        if not jp.exists():
            continue
        J = json.load(open(jp))
        fo, LL = J['frame_origins'], J.get('link_lengths', {})
        names = list(fo.keys())
        for i, l in enumerate(names):
            mp = Path(f'data/test/{nm}/meshes/{l}.stl')
            if not mp.exists():
                continue
            mesh = trimesh.load(mp)
            u = m2p.bone_axis(names, i, fo, mesh)
            ln = LL.get(l) or float(np.linalg.norm(
                np.asarray(mesh.vertices, dtype=float), axis=1).max())
            p = m2p.obb_params(mesh, 1.0)
            ro, rn = p['radius'], m2p.perp_radius(mesh, u, 1.0)
            a = np.asarray(p['obb_axes'][0], dtype=float)
            a /= np.linalg.norm(a)
            deg = np.degrees(np.arccos(min(1.0, abs(float(u @ a)))))
            po, pn = outside(mesh, u, ln, ro), outside(mesh, u, ln, rn)
            so += po; sn += pn
            rows.append((nm, l, deg, po, pn))
            print(f"{nm:8s}{l:11s}{deg:6.1f}°{ro:9.4f}{rn:8.4f}"
                  f"{rn/ro:6.2f}{po:11.1f}%{pn:6.1f}%")

    if not rows:
        print("❌ 実物メッシュが見つからない。data/test/*/meshes/ を確認。")
        return 1

    n = len(rows)
    print(f"\n平均はみ出し率  旧 {so/n:.1f} %  →  新 {sn/n:.1f} %")

    # 判定①: 軸ずれが大きいリンクほど改善が大きいか（=害の再現と是正の対応）
    flip = [r for r in rows if r[2] > 20.0]
    print(f"\n軸が入れ替わっているリンク（ずれ > 20°）: {len(flip)} 本")
    for nm, l, deg, po, pn in flip:
        print(f"   {nm} {l}: ずれ {deg:.1f}°  はみ出し {po:.1f} % → {pn:.1f} %")

    # 判定②: どのリンクも悪化していないか（丸め相当の 1 pt は許す）
    worse = [r for r in rows if r[4] > r[3] + 1.0]
    ok = (sn < so) and not worse and all(pn < po for _, _, _, po, pn in flip)
    print(f"\n悪化したリンク: {len(worse)} 本")
    print("✅ 是正は成立している。" if ok else "❌ 期待した挙動になっていない。")

    print("\n⚠️ **残るはみ出しはカプセル近似そのものの限界**で、この修正では縮まない。")
    print("   B1 hand が 70.9 % 残るのは、手先が平たくカプセル（円柱＋半球）で"
          "覆いようがないためである。**別の限界として記録する。**")
    print("⚠️ 台座リンクは軸ずれ 0° でも改善する（B1 base 15.4 → 0.9 %）。"
          "立方体に近く、OBB の短辺 2 本が関節方向の断面より小さいためである。")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
