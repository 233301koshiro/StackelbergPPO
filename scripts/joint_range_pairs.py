import json, sys, numpy as np, yaml
sys.path.insert(0, '/userdir/StackelbergPPO_ra/scripts')
import joint_range_a1v as J
P={p['type']:p for p in yaml.safe_load(open(J.PROJ+'/irsl_assembler_config.yaml'))['PartsSettings']}
d=json.load(open(sys.argv[1])); k=int(sys.argv[2]); qs=[float(x) for x in sys.argv[3].split(',')]
names=[p['name'] for p in d['parts']]; parts=[(p['type'],np.array(p['T'])) for p in d['parts']]
T=[M.copy() for _,M in parts]; sv=[i for i,(t,_) in enumerate(parts) if t=='XL-430']; up=np.array([0,0,1.])
S=[J.Shape(t,P) for t,_ in parts]
def horn(i): return T[sv[i]][:3,3].copy(), T[sv[i]][:3,2].copy()
def move_after(kk,M):
    for i in range(sv[kk]+1,len(T)): T[i]=M@T[i]
def tip():
    o,a=horn(len(sv)-1); w=np.vstack([(S[i].pts@T[i][:3,:3].T)+T[i][:3,3] for i in range(sv[-1]+1,len(T))]); v=w-o
    return w[np.argmax(np.linalg.norm(v-np.outer(v@a,a),axis=1))]
def signed(u,w,a):
    u=u-(u@a)*a; w=w-(w@a)*a; return np.degrees(np.arctan2(np.cross(u,w)@a,u@w))
for kk in (1,2,3):
    o,a=horn(kk); b=horn(kk+1)[0] if kk+1<len(sv) else tip(); move_after(kk,J.rot_about(o,a,signed(b-o,up,a)))
a_ref=horn(1)[1]; o=T[sv[k]][:3,3]
def pairs(q):
    M=J.rot_about(o,a_ref,q); out={}
    for m in range(sv[k]+1,len(T)):
        wm=S[m].pts@(M@T[m])[:3,:3].T+(M@T[m])[:3,3]
        for s in range(0,sv[k]+1):
            v=S[s].inside(wm,T[s])
            if v: out[(names[m],names[s])]=v
    return out
base=pairs(0)
for q in qs:
    p=pairs(q)
    diff={kk:p.get(kk,0)-base.get(kk,0) for kk in set(p)|set(base) if p.get(kk,0)-base.get(kk,0)>2}
    print(q, sorted(diff.items(), key=lambda x:-x[1])[:4])
