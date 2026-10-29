import json, os, sys
import numpy as np, trimesh, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, '/userdir/StackelbergPPO_ra/scripts')
import joint_range_a1v as J
import yaml
for f in font_manager.findSystemFonts():
    if 'NotoSansCJK' in f or 'NotoSerifCJK-Regular' in f: font_manager.fontManager.addfont(f)
nm=[f.name for f in font_manager.fontManager.ttflist if 'CJK JP' in f.name]
plt.rcParams['font.family']=sorted(nm,key=lambda n:'Sans' not in n)[0]
P={p['type']:p for p in yaml.safe_load(open(J.PROJ+'/irsl_assembler_config.yaml'))['PartsSettings']}
d=json.load(open(sys.argv[1])); lims=json.loads(sys.argv[2]); out=sys.argv[3]; title=sys.argv[4]
parts=[(p['type'],np.array(p['T']),p['visual']) for p in d['parts']]
T=[M.copy() for _,M,_ in parts]; sv=[i for i,(t,_,_) in enumerate(parts) if t=='XL-430']; up=np.array([0,0,1.])
S=[J.Shape(t,P) for t,_,_ in parts]
def horn(i): return T[sv[i]][:3,3].copy(), T[sv[i]][:3,2].copy()
def move_after(k,M):
    for i in range(sv[k]+1,len(T)): T[i]=M@T[i]
def tip():
    o,a=horn(len(sv)-1); w=np.vstack([(S[i].pts@T[i][:3,:3].T)+T[i][:3,3] for i in range(sv[-1]+1,len(T))]); v=w-o
    return w[np.argmax(np.linalg.norm(v-np.outer(v@a,a),axis=1))]
def signed(u,w,a):
    u=u-(u@a)*a; w=w-(w@a)*a; return np.degrees(np.arctan2(np.cross(u,w)@a,u@w))
for k in (1,2,3):
    o,a=horn(k); b=horn(k+1)[0] if k+1<len(sv) else tip(); move_after(k,J.rot_about(o,a,signed(b-o,up,a)))
a_ref=horn(1)[1]; T0=[M.copy() for M in T]
def mesh(v):
    if 'box' in v: m=trimesh.creation.box(extents=v['box'])
    else: m=trimesh.load(v['url'],force='mesh',process=False); m.apply_scale(v['scale_mm'])
    m.apply_transform(np.array(v['T'])); return m
fig=plt.figure(figsize=(4.2*len(lims),5),dpi=130)
for n,(k,q) in enumerate(lims):
    Tq=[M.copy() for M in T0]; o=T0[sv[k]][:3,3]; Mr=J.rot_about(o,a_ref,q)
    for i in range(sv[k]+1,len(Tq)): Tq[i]=Mr@Tq[i]
    ax=fig.add_subplot(1,len(lims),n+1,projection='3d'); F=[];C=[]
    for i,(t,_,vis) in enumerate(parts):
        for v in vis:
            m=mesh(v); m.apply_transform(Tq[i])
            col=np.array(v['color'][:3]) if v.get('color') else (np.array([.18,.18,.2]) if t=='XL-430' else np.array([.85,.85,.83]))
            if i>sv[k]: col=col*0.6+np.array([0.4,0.1,0.1])*0.4
            sh=0.45+0.55*np.abs(m.face_normals@np.array([0.4,-0.6,0.7])/np.linalg.norm([0.4,-0.6,0.7]))
            F.append(m.vertices[m.faces]); C.append(np.clip(col[None]*sh[:,None],0,1))
    ax.add_collection3d(Poly3DCollection(np.concatenate(F),facecolors=np.concatenate(C),edgecolors='none'))
    allv=np.concatenate(F).reshape(-1,3); c=(allv.min(0)+allv.max(0))/2; s=(allv.max(0)-allv.min(0)).max()/2
    ax.set_xlim(c[0]-s,c[0]+s); ax.set_ylim(c[1]-s,c[1]+s); ax.set_zlim(c[2]-s,c[2]+s); ax.view_init(elev=0,azim=0)
    ax.set_title(f'関節 {k+1} を {q:+.0f}°（赤みが回した側）',fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
fig.suptitle(title,fontsize=12); plt.tight_layout(); fig.savefig(out); print(out)
