"""ФИНАЛЬНЫЙ ПРОГОН на всех базах: A (без Фристона) vs B (с Фристоном) + структура одеяла, по доменам."""
import numpy as np, pandas as pd, glob, os, gzip, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
rng=np.random.default_rng(0)
def skill(t,p,ref):
    d=np.sum((t-ref)**2); return 1-np.sum((t-p)**2)/d if d>1e-9 else np.nan
def embed(x,d=10,tau=2):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<d*tau+150: return None
    return np.column_stack([x[i*tau:len(x)-(d-1-i)*tau] for i in range(d)])
def sysA(Ztr,Zte,lam=5.0):
    X=Ztr[:-1];Y=Ztr[1:];N=X.shape[1];K=np.linalg.solve(X.T@X+lam*np.eye(N),X.T@Y)
    return skill(Zte[1:],Zte[:-1]@K,Zte[:-1])
def sysB(Ztr,Zte,r):
    X=Ztr[:-1].T;Df=np.diff(Ztr,axis=0).T;U,S,Vt=np.linalg.svd(X,full_matrices=False);rr=min(r,len(S)-1,X.shape[0]-1)
    Ur,Sr,Vr=U[:,:rr],S[:rr],Vt[:rr].T;D=Ur@(Ur.T@Df@Vr@np.diag(1/Sr))@Ur.T
    return skill(Zte[1:],Zte[:-1]+(D@Zte[:-1].T).T,Zte[:-1])
def P1(Ztr,r=8):
    cut=int(len(Ztr)*0.7);T=Ztr[:cut];X=T[:-1].T;Df=np.diff(T,axis=0).T;U,S,Vt=np.linalg.svd(X,full_matrices=False);rr=min(r,len(S)-1)
    Dh=(U[:,:rr]@(U[:,:rr].T@Df@Vt.T[:,:rr]@np.diag(1/S[:rr]))@U[:,:rr].T)@T[:-1].T
    return 1-np.sum((Df-Dh)**2)/np.sum((Df-Df.mean(1,keepdims=True))**2)
def blanket(Ztr,r=8):
    mu=Ztr.mean(0);Xc=Ztr-mu;U,S,Vt=np.linalg.svd(Xc,full_matrices=False);rr=min(r,len(S)-1,Xc.shape[1])
    A=Xc@Vt[:rr].T;K=np.linalg.lstsq(A[:-1],A[1:],rcond=None)[0]
    G=A[:-1].T@A[:-1];A1=A[:-1].T@A[1:];A2=A[1:].T@A[1:];lam,W=np.linalg.eig(K);res=np.zeros(rr)
    for i in range(rr):
        w=W[:,i];gg=np.real(np.conj(w)@G@w)+1e-12
        res[i]=np.sqrt(max(np.real(np.conj(w)@A2@w-np.conj(lam[i])*(np.conj(w)@A1@w)-lam[i]*np.conj(w)@A1.T@w+abs(lam[i])**2*(np.conj(w)@G@w)),0)/gg)
    am=np.abs(lam);rn=res/(res.max()+1e-9)
    o=((rn<0.4)&(am>0.9)).mean();b=((rn>=0.3)&(rn<0.75)&(am>0.85)).mean();return o,b,1-o-b

# ===== ЗАГРУЗКА ВСЕХ БАЗ =====
DB={}  # (category, name) -> matrix
import scipy.io as sio
def nm(x):
    v=x
    while hasattr(v,'ravel') and getattr(v,'dtype',None)==object and v.size>0: v=v.ravel()[0]
    s=str(v).strip("[]'\" ");return s if s and s!='0' else None
# биология: нематода (с чего всё началось) + MEG
for f in sorted(glob.glob('/home/claude/celegans/kato5/*.mat')):
    wb=sio.loadmat(f)['wbData'][0,0];X=np.array(wb['deltaFOverF'],dtype=float)
    col=~np.all(np.isnan(X),axis=0);X=np.where(np.isnan(X),np.nanmean(X,axis=0),X)[:,col]
    DB[('БИОЛОГИЯ','нематода_'+os.path.basename(f)[:8])]=X
try:
    import mne;raw=mne.io.read_raw_fif('/tmp/meg_raw.fif',preload=True,verbose=False).pick('mag',verbose=False)
    DB[('БИОЛОГИЯ','MEG_мозг')]=raw.get_data()[:40,:4000].T
except: pass
# климат/природа
DB[('КЛИМАТ','CO2')]=embed(sm.datasets.co2.load_pandas().data['co2'].interpolate().dropna().values)
DB[('КЛИМАТ','sunspots')]=embed(sm.datasets.sunspots.load_pandas().data['SUNACTIVITY'].values)
eln=sm.datasets.elnino.load_pandas().data.iloc[:,1:].values.flatten();DB[('КЛИМАТ','elnino')]=embed(eln[~np.isnan(eln)])
# энергетика
DB[('ЭНЕРГЕТИКА','ETTh2')]=pd.read_csv('/tmp/val/ETTh2.csv').iloc[:4000,1:].values.astype(float)
DB[('ЭНЕРГЕТИКА','ETTm2')]=pd.read_csv('/tmp/val/ETTm2.csv').iloc[:5000,1:].values.astype(float)
with gzip.open('/tmp/val/electricity.gz','rt') as fh: DB[('ЭНЕРГЕТИКА','electricity')]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:4000,:25]
with gzip.open('/tmp/val/solar.gz','rt') as fh: DB[('ЭНЕРГЕТИКА','solar')]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:4000,:25]
# транспорт
with gzip.open('/tmp/val/traffic.gz','rt') as fh: DB[('ТРАНСПОРТ','traffic')]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:4000,:25]
# финансы
try:
    with gzip.open('/tmp/exchange.csv','rt') as fh: DB[('ФИНАНСЫ','exchange')]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:5000]
except Exception as e: print('exchange skip')
# NAB (метрики серверов/веб)
for p in sorted(glob.glob('/tmp/NAB/data/*/*.csv')):
    Z=embed(pd.read_csv(p)['value'].values.astype(float)[:4000])
    if Z is not None: DB[('NAB_аномалии',os.path.basename(p)[:12])]=Z
# сети UAF
sys_path='/home/claude/uaf/uaf-v6.2';import sys;sys.path.insert(0,sys_path);from uaf import core as CORE
for fn in sorted(glob.glob(sys_path+'/data/real/mjs_other/*.dat'))[:10]:
    try:
        edges=[]
        for ln in open(fn):
            pp=ln.split()
            if len(pp)>=2:
                try: edges.append((int(float(pp[0])),int(float(pp[1]))))
                except: pass
        nodes=sorted(set([a for a,b in edges]+[b for a,b in edges]))[:100];idx={n:i for i,n in enumerate(nodes)};Nn=len(nodes)
        if Nn<20: continue
        A=np.zeros((Nn,Nn))
        for a,b in edges:
            if a in idx and b in idx: A[idx[a],idx[b]]=1;A[idx[b],idx[a]]=1
        traj=CORE.integrate_network(np.clip(0.2+0.05*rng.standard_normal(Nn),.01,.99),A/max(abs(np.linalg.eigvals(A))),T=1200,dt=0.4,sigma=0.015,seed=1)
        if traj.std()>1e-4: DB[('СЕТИ_UAF',os.path.basename(fn)[:12])]=traj
    except: pass
# хаос/канонич.
def lorenz(T=4000,dt=0.02,r=28):
    xyz=np.zeros((T,3));xyz[0]=[1,1,1]
    for i in range(1,T):x,y,z=xyz[i-1];xyz[i]=xyz[i-1]+dt*np.array([10*(y-x),x*(r-z)-y,x*y-8/3*z])
    return xyz
DB[('ХАОС','Lorenz28')]=lorenz();DB[('ХАОС','Lorenz35')]=lorenz(r=35)
t=np.linspace(0,200*np.pi,4000);DB[('ХАОС','sines')]=np.column_stack([np.sin(t*f)+0.3*rng.standard_normal(4000) for f in [1,1.7,2.3]])
DB[('ХАОС','white_noise')]=rng.standard_normal((4000,6));DB[('ХАОС','random_walk')]=np.cumsum(rng.standard_normal((4000,6)),axis=0)
# M4 (все частоты, подвыборка)
for fn,pre,nr in [('/tmp/m4h.csv','H',None),('/tmp/m4w.csv','W',None),('/tmp/m4d.csv','D',600),('/tmp/m4m.csv','M',600),('/tmp/m4q.csv','Q',600)]:
    df=pd.read_csv(fn,nrows=nr)
    for i,(_,row) in enumerate(df.iterrows()):
        Z=embed(row.iloc[1:].values.astype(float))
        if Z is not None: DB[('M4_соревнов',f'M4{pre}_{i}')]=Z

# ===== ПРОГОН =====
from collections import defaultdict
rows=defaultdict(list)
for (cat,name),raw in DB.items():
    if raw is None: continue
    raw=np.asarray(raw,float)[:5000]
    if raw.ndim==1: raw=raw.reshape(-1,1)
    Z=(raw-raw.mean(0))/(raw.std(0)+1e-9);cut=int(len(Z)*0.7);r=min(8,Z.shape[1]-1)
    if Z.shape[1]<2 or cut<100: continue
    try:
        a=sysA(Z[:cut],Z[cut:]);b=sysB(Z[:cut],Z[cut:],r);p=P1(Z);o,bd,ch=blanket(Z)
        if any(np.isnan([a,b,p])): continue
        rows[cat].append((np.clip(a,-5,1),np.clip(b,-5,1),np.clip(p,-1,1),o,bd,ch))
    except: pass

print('='*90)
tot=sum(len(v) for v in rows.values())
print(f'ФИНАЛЬНЫЙ ПРОГОН НА ВСЕХ БАЗАХ: {tot} датасетов, {len(rows)} доменов')
print('='*90)
print(f'{"домен":<16}{"n":<6}{"A(без Фр)":<11}{"B(с Фр)":<10}{"победа":<9}{"нутро/одеяло/хаос":<20}{"P1"}')
print('-'*90)
allA=[];allB=[];allP=[];allO=[]
for cat in sorted(rows):
    M=np.array(rows[cat]);A,B,P,O,BD,CH=M[:,0],M[:,1],M[:,2],M[:,3],M[:,4],M[:,5]
    allA+=list(A);allB+=list(B);allP+=list(P);allO+=list(O)
    win='B' if B.mean()>A.mean()+0.02 else 'A' if A.mean()>B.mean()+0.02 else '≈'
    print(f'{cat:<16}{len(M):<6}{A.mean():+.2f}       {B.mean():+.2f}     {win:<9}{O.mean():.0%}/{BD.mean():.0%}/{CH.mean():.0%}          {P.mean():+.2f}')
allA=np.array(allA);allB=np.array(allB);allP=np.array(allP);allO=np.array(allO)
print('-'*90)
print(f'{"ВСЕГО":<16}{tot:<6}{allA.mean():+.2f}       {allB.mean():+.2f}     {"B" if allB.mean()>allA.mean() else "A":<9}')
print(f'  A катастроф: {np.mean(allA<-0.5)*100:.0f}%  |  B катастроф: {np.mean(allB<-0.5)*100:.0f}%')
print(f'  корр(P1, точность B): {np.corrcoef(allP,allB)[0,1]:+.2f}  |  корр(доля нутра, точность B): {np.corrcoef(allO,allB)[0,1]:+.2f}')
print('='*90)
