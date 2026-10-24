#!/usr/bin/env python3
"""運動学の指標（Yoshikawa の可操作度・先端の最大の速さ）で、水平 3 関節アームの形の順位を予測する（系譜 9-295）。学習なし・数秒。

「計算 1 つで分かるのでは。なぜ学習するのか」という批判への試験。リンク長は軌跡の関節の位置から取った値（9-295 の表）。
  到達: 目標 (0.8, 0) での w = sqrt(det(J J^T))（先端の位置 2 × 関節 3）。冗長な 1 自由度の逆運動学の解の中の最大値
  押し: 先端を箱の手前の面 (0.85, 0) に置く解の中で、各関節 |dq| <= 3.1416 rad/s のときの +x 方向の先端の最大の速さ

    python3 scripts/probe_kinematic_indices.py
"""
import numpy as np

L={'mid':[0.334,0.329,0.224],'long':[0.547,0.538,0.366],'prox':[0.5,0.25,0.137],'dist':[0.137,0.25,0.5],'short':[0.152,0.150,0.102]}
W=3.1416
def fk(l,q):
    a=np.cumsum(q); p=[np.zeros(2)]
    for li,ai in zip(l,a): p.append(p[-1]+li*np.array([np.cos(ai),np.sin(ai)]))
    return np.array(p)
def jac_from_pts(p):
    tip=p[-1]; return np.array([[-(tip[1]-p[i][1]) for i in range(3)],[tip[0]-p[i][0] for i in range(3)]])
def ik_all(l,goal,n=3600):
    sols=[]
    for phi in np.linspace(-np.pi,np.pi,n,endpoint=False):
        w=goal-l[2]*np.array([np.cos(phi),np.sin(phi)]); d=np.linalg.norm(w)
        c=(d*d-l[0]**2-l[1]**2)/(2*l[0]*l[1])
        if abs(c)>1: continue
        for s in (1,-1):
            q2=s*np.arccos(c); q1=np.arctan2(w[1],w[0])-np.arctan2(l[1]*np.sin(q2),l[0]+l[1]*np.cos(q2))
            q3=phi-q1-q2; q=np.array([q1,q2,(q3+np.pi)%(2*np.pi)-np.pi])
            if np.all(np.abs(q)<=np.pi): sols.append(q)
    return sols
out={}
for k,l in L.items():
    s=ik_all(l,np.array([0.8,0.0]))
    if not s: out[k]=None; print(k,'目標に届く解なし'); continue
    ws=[]; vx=[]
    for q in s:
        p=fk(l,q); assert np.linalg.norm(p[-1]-[0.8,0])<1e-6
        J=jac_from_pts(p); ws.append(np.sqrt(np.linalg.det(J@J.T)))
    s2=ik_all(l,np.array([0.85,0.0]))
    for q in s2:
        J=jac_from_pts(fk(l,q)); vx.append(W*np.abs(J[0]).sum())
    print(f'{k:5s} 到達 w の最大 {max(ws):.4f}（解 {len(s)}）  押し +x 最大の速さ {max(vx) if vx else float("nan"):.3f} m/s')
