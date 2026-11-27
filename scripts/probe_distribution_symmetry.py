#!/usr/bin/env python3
"""押しの根元重 対 先端重に差が出なかったのは、条件が 2 形態を同じ物にしていたからかを確かめる（系譜 9-363）。学習なし・数秒。

根元重 0.5/0.25/0.137 と先端重 0.137/0.25/0.5 は、リンクの順序を逆にした形である。
  ① 先端の最大の速さ（9-295 の指標）を、接触点と関節ごとの速度の上限を変えて出す
  ② 伸ばした姿勢での根元まわりの慣性と重心（質量は長さに比例する細い棒）

    python3 scripts/probe_distribution_symmetry.py
"""
import numpy as np
src=open('scripts/probe_kinematic_indices.py').read().split('out={}')[0]
exec(src)
def vmax(l,pt,W3):
    best=0
    for q in ik_all(l,np.array(pt),n=3600):
        J=jac_from_pts(fk(l,q)); best=max(best,(np.abs(J[0])*np.array(W3)).sum())
    return best
eq=[3.1416]*3
for pt in [(0.85,0),(0.6,0),(0.6,0.3),(0.5,0.5),(0.3,0.6)]:
    a=vmax(L['prox'],pt,eq); b=vmax(L['dist'],pt,eq)
    print(f'接触点 {pt} 等速度上限: 根元重 {a:.3f} 先端重 {b:.3f} 差 {100*(a-b)/max(a,b):+.0f}%')
for name,W3 in [('根元を半分の速さ',[1.5708,3.1416,3.1416]),('先端を倍の速さ',[3.1416,3.1416,6.2832])]:
    a=vmax(L['prox'],(0.85,0),W3); b=vmax(L['dist'],(0.85,0),W3)
    print(f'接触点 (0.85,0) {name}: 根元重 {a:.3f} 先端重 {b:.3f}')
# 伸ばした姿勢での根元まわりの慣性（質量は長さに比例、細い棒）
for k in ['prox','dist']:
    l=np.array(L[k]); x0=np.concatenate([[0],np.cumsum(l)[:-1]])
    I=sum(li*((x+li)**3-x**3)/3/li for li,x in zip(l,x0))  # ρ=1: ∫r^2 dr
    com=sum(li*(x+li/2) for li,x in zip(l,x0))/l.sum()
    print(f'{k}: 根元まわりの慣性（ρ=1）{I:.4f}  重心までの距離 {com:.3f} m')
# ③ 関節にアクチュエータの質量 m を置いた場合（今のモデルには無い）の、伸ばした姿勢での根元まわりの慣性への寄与
for k in ['prox','dist']:
    l=np.array(L[k]); r=np.cumsum(l)[:-1]
    print(f'{k}: 第2・第3関節に質量 m を置いたときの寄与 {np.sum(r**2):.3f} m  （関節の位置 {r.round(3)} m）')
