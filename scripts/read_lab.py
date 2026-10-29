import os, numpy as np
from cnoid.Body import BodyLoader
b = BodyLoader().load('/userdir/StackelbergPPO/data/test/tomoki/simplehand.body')
b.calcForwardKinematics()
out=[]
for i in range(b.numLinks):
    L=b.link(i)
    jt=L.jointType
    if L.isRevoluteJoint():
        p=np.array(L.p)*1000; a=np.array(L.R)@np.array(L.a)
        par=L.parent.name if L.parent else None
        out.append((L.name,p,a,par))
        print(f'{L.name:28s} 親 {par:28s} 位置 {np.round(p,1)} 軸 {np.round(a,2)} 範囲 {np.degrees(L.q_lower):.0f}〜{np.degrees(L.q_upper):.0f}', flush=True)
print('総質量 kg', round(b.mass,3), flush=True)
os._exit(0)
