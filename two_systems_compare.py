"""ЧЕСТНОЕ сравнение на проверенной основе DMD-change.
A (без Фристона): ridge-регуляризованная спектр. реконструкция x(t+1)=K x(t), без якоря.
B (с Фристоном):  self-evidencing персистенс-якорь + precision-weighting мод (по остаткам ResDMD)."""
import numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
rng=np.random.default_rng(0)
def skill(t,p,ref):
    d=np.sum((t-ref)**2); return 1-np.sum((t-p)**2)/d if d>1e-9 else np.nan
def embed(x,d=10,tau=2):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<d*tau+150: return None
    return np.column_stack([x[i*tau:len(x)-(d-1-i)*tau] for i in range(d)])

def systemA(Ztr,Zte,lam=5.0):  # БЕЗ Фристона: ridge DMD реконструкция, без якоря
    X=Ztr[:-1]; Y=Ztr[1:]; N=X.shape[1]
    K=np.linalg.solve(X.T@X+lam*np.eye(N),X.T@Y)   # x(t+1)=K x(t), ridge
    return skill(Zte[1:],Zte[:-1]@K,Zte[:-1])

def systemB(Ztr,Zte,r):  # С Фристоном: персистенс-якорь + precision-weighting мод
    X=Ztr[:-1].T; Df=np.diff(Ztr,axis=0).T; N=Ztr.shape[1]
    U,S,Vt=np.linalg.svd(X,full_matrices=False); rr=min(r,len(S)-1,N-1)
    Ur,Sr,Vr=U[:,:rr],S[:rr],Vt[:rr].T
    # низкоранговый оператор изменения
    Dred=Ur.T@Df@Vr@np.diag(1/Sr)                  # rr×rr в редуц. базисе входа? -> строим D
    D=Ur@Dred@Ur.T
    # precision-weighting: моды D по остаткам (надёжные усилить)
    lam_d,W=np.linalg.eig(Dred+np.eye(rr)); Winv=np.linalg.inv(W+1e-9*np.eye(rr))
    G=(Ur.T@X)@(Ur.T@X).T  # прокси Грам в редуц.
    res=np.array([abs(lam_d[i]) for i in range(rr)])  # прокси остатка (упрощ.)
    prec=np.exp(-np.abs(lam_d-1)/ (np.median(np.abs(lam_d-1))+1e-9))  # надёжность у ед.окружности
    Dw=np.real(W@np.diag(prec*(lam_d-1))@Winv)     # precision-взвешенное изменение
    Dfull=Ur@Dw@Ur.T
    # self-evidencing якорь: x(t+1)=x(t)+Dfull x(t)
    return skill(Zte[1:],Zte[:-1]+Zte[:-1]@Dfull.T,Zte[:-1])

# датасеты
D={}
def lorenz(T=4000,dt=0.02,r=28):
    xyz=np.zeros((T,3)); xyz[0]=[1,1,1]
    for i in range(1,T): x,y,z=xyz[i-1]; xyz[i]=xyz[i-1]+dt*np.array([10*(y-x),x*(r-z)-y,x*y-8/3*z])
    return xyz
D['CO2']=embed(sm.datasets.co2.load_pandas().data['co2'].interpolate().dropna().values)
D['sunspots']=embed(sm.datasets.sunspots.load_pandas().data['SUNACTIVITY'].values)
eln=sm.datasets.elnino.load_pandas().data.iloc[:,1:].values.flatten(); D['elnino']=embed(eln[~np.isnan(eln)])
D['Lorenz']=lorenz(); D['Lorenz35']=lorenz(r=35)
D['ETTh2']=pd.read_csv('/tmp/val/ETTh2.csv').iloc[:4000,1:].values.astype(float)
import gzip
for s,f in [('elec','/tmp/val/electricity.gz'),('traffic','/tmp/val/traffic.gz'),('solar','/tmp/val/solar.gz')]:
    with gzip.open(f,'rt') as fh: D[s]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:4000,:20]
for p in sorted(glob.glob('/tmp/NAB/data/*/*.csv'))[:12]:
    D['NAB_'+os.path.basename(p)[:8]]=embed(pd.read_csv(p)['value'].values.astype(float)[:4000])
for fn,pre in [('/tmp/m4h.csv','h'),('/tmp/m4d.csv','d')]:
    df=pd.read_csv(fn,nrows=120)
    for i,(_,row) in enumerate(df.iterrows()):
        Z=embed(row.iloc[1:].values.astype(float))
        if Z is not None: D[f'M4{pre}_{i}']=Z

resA=[];resB=[];names=[]
for name,raw in D.items():
    if raw is None: continue
    raw=np.asarray(raw,float)[:4000]; Z=(raw-raw.mean(0))/(raw.std(0)+1e-9); cut=int(len(Z)*0.7); r=min(8,Z.shape[1]-1)
    if Z.shape[1]<2 or cut<100: continue
    try:
        sA=systemA(Z[:cut],Z[cut:]); sB=systemB(Z[:cut],Z[cut:],r)
        if np.isnan(sA) or np.isnan(sB): continue
        resA.append(np.clip(sA,-5,1)); resB.append(np.clip(sB,-5,1)); names.append(name)
    except: pass
resA=np.array(resA); resB=np.array(resB)
print('='*68); print('A (без Фристона: ridge-спектр) vs B (с Фристоном: якорь+precision)'); print('='*68)
for i in range(min(14,len(names))):
    w='A' if resA[i]>resB[i]+0.02 else 'B' if resB[i]>resA[i]+0.02 else '≈'
    print(f'  {names[i]:<14} A={resA[i]:+.3f}  B={resB[i]:+.3f}  -> {w}')
print('-'*68)
print(f'ИТОГ по {len(resA)} датасетам:')
print(f'  A (без Фристона): среднее {resA.mean():+.3f}, медиана {np.median(resA):+.3f}, полож. {np.mean(resA>0.02)*100:.0f}%')
print(f'  B (с Фристоном):  среднее {resB.mean():+.3f}, медиана {np.median(resB):+.3f}, полож. {np.mean(resB>0.02)*100:.0f}%')
print(f'  B>A: {np.mean(resB>resA+0.01)*100:.0f}%  |  A>B: {np.mean(resA>resB+0.01)*100:.0f}%')
print(f'  катастрофы<-0.5: A={np.mean(resA<-0.5)*100:.0f}%  B={np.mean(resB<-0.5)*100:.0f}%')
print('='*68)
