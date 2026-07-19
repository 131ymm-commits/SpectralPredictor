"""Стабильные три системы. B — проверенный DMD-change (прямой, без eigendecomp).
C — тот же СТАБИЛЬНЫЙ прогноз + ДИАГНОСТИКА одеяла (порядок/граница/хаос) как интерпретация."""
import numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
def skill(t,p,ref):
    d=np.sum((t-ref)**2); return 1-np.sum((t-p)**2)/d if d>1e-9 else np.nan
def embed(x,d=10,tau=2):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<d*tau+150: return None
    return np.column_stack([x[i*tau:len(x)-(d-1-i)*tau] for i in range(d)])
def sysA(Ztr,Zte,lam=5.0):
    X=Ztr[:-1]; Y=Ztr[1:]; N=X.shape[1]; K=np.linalg.solve(X.T@X+lam*np.eye(N),X.T@Y)
    return skill(Zte[1:],Zte[:-1]@K,Zte[:-1])
def change_op(Ztr,r):  # проверенный низкоранговый оператор изменения (СТАБИЛЬНЫЙ)
    X=Ztr[:-1].T; Df=np.diff(Ztr,axis=0).T; U,S,Vt=np.linalg.svd(X,full_matrices=False); rr=min(r,len(S)-1,X.shape[0]-1)
    Ur,Sr,Vr=U[:,:rr],S[:rr],Vt[:rr].T
    return Ur@(Ur.T@Df@Vr@np.diag(1/Sr))@Ur.T
def sysB(Ztr,Zte,r):
    D=change_op(Ztr,r); return skill(Zte[1:],Zte[:-1]+(D@Zte[:-1].T).T,Zte[:-1])
def blanket_diag(Ztr,r):  # диагностика одеяла БЕЗ реконструкции (только спектр редуц. оператора)
    mu=Ztr.mean(0); Xc=Ztr-mu; U,S,Vt=np.linalg.svd(Xc,full_matrices=False); rr=min(r,len(S)-1,Xc.shape[1])
    A=Xc@Vt[:rr].T; K=np.linalg.lstsq(A[:-1],A[1:],rcond=None)[0]
    G=A[:-1].T@A[:-1]; A1=A[:-1].T@A[1:]; A2=A[1:].T@A[1:]
    lam,W=np.linalg.eig(K); res=np.zeros(len(lam))
    for i in range(len(lam)):
        w=W[:,i]; gg=np.real(np.conj(w)@G@w)+1e-12
        res[i]=np.sqrt(max(np.real(np.conj(w)@A2@w-np.conj(lam[i])*(np.conj(w)@A1@w)-lam[i]*np.conj(w)@A1.T@w+abs(lam[i])**2*(np.conj(w)@G@w)),0)/gg)
    am=np.abs(lam); rn=res/(res.max()+1e-9)
    order=((rn<0.5)&(am>0.85)).mean()             # μ_pp нутро
    boundary=((rn>=0.3)&(rn<0.7)&(am>0.9)).mean()  # μ_sc одеяло (промежуточный остаток у |λ|~1)
    chaos=1-order-boundary
    ews=am[(rn>=0.3)&(am>0.9)].max() if ((rn>=0.3)&(am>0.9)).any() else 0
    return order,boundary,chaos,ews
def sysC(Ztr,Zte,r):  # прогноз = стабильный DMD-change; + диагностика одеяла
    sB=sysB(Ztr,Zte,r); diag=blanket_diag(Ztr,r); return sB,diag

D={}
def lorenz(T=4000,dt=0.02,r=28):
    xyz=np.zeros((T,3)); xyz[0]=[1,1,1]
    for i in range(1,T): x,y,z=xyz[i-1]; xyz[i]=xyz[i-1]+dt*np.array([10*(y-x),x*(r-z)-y,x*y-8/3*z])
    return xyz
D['CO2']=embed(sm.datasets.co2.load_pandas().data['co2'].interpolate().dropna().values)
D['sunspots']=embed(sm.datasets.sunspots.load_pandas().data['SUNACTIVITY'].values)
D['Lorenz']=lorenz(); D['ETTh2']=pd.read_csv('/tmp/val/ETTh2.csv').iloc[:4000,1:].values.astype(float)
import gzip
for s,f in [('elec','/tmp/val/electricity.gz'),('traffic','/tmp/val/traffic.gz')]:
    with gzip.open(f,'rt') as fh: D[s]=np.array([[float(x) for x in ln.split(',')] for ln in fh])[:4000,:20]
for fn,pre,nr in [('/tmp/m4h.csv','h',None),('/tmp/m4d.csv','d',1000),('/tmp/m4m.csv','m',1500),('/tmp/m4q.csv','q',1000)]:
    df=pd.read_csv(fn,nrows=nr)
    for i,(_,row) in enumerate(df.iterrows()):
        Z=embed(row.iloc[1:].values.astype(float))
        if Z is not None: D[f'M4{pre}_{i}']=Z

rA=[];rB=[];rC=[];bl=[]; byord=[]
for name,raw in D.items():
    if raw is None: continue
    raw=np.asarray(raw,float)[:4000]; Z=(raw-raw.mean(0))/(raw.std(0)+1e-9); cut=int(len(Z)*0.7); r=min(8,Z.shape[1]-1)
    if Z.shape[1]<2 or cut<100: continue
    try:
        sA=sysA(Z[:cut],Z[cut:]); sB=sysB(Z[:cut],Z[cut:],r); sC,diag=sysC(Z[:cut],Z[cut:],r)
        if any(np.isnan([sA,sB,sC])): continue
        rA.append(np.clip(sA,-5,1)); rB.append(np.clip(sB,-5,1)); rC.append(np.clip(sC,-5,1)); bl.append(diag)
        byord.append((diag[0],np.clip(sB,-5,1)))  # связь доли нутра с точностью
    except: pass
rA=np.array(rA);rB=np.array(rB);rC=np.array(rC); bl=np.array(bl); byo=np.array(byord)
print('='*74); print(f'ТРИ СИСТЕМЫ, СЕРЬЁЗНАЯ СТАТИСТИКА: {len(rA)} датасетов (M4+разное)'); print('='*74)
for nm,r_ in [('A — без Фристона (ridge-спектр)',rA),('B — Фристон (якорь, без переглуш.)',rB),('C — прогноз B + структура одеяла',rC)]:
    print(f'  {nm:<36} среднее {r_.mean():+.3f} медиана {np.median(r_):+.3f} полож {np.mean(r_>0.02)*100:.0f}% катастр {np.mean(r_<-0.5)*100:.0f}%')
print('-'*74)
print(f'  A>B:{np.mean(rA>rB+0.01)*100:.0f}%  B>A:{np.mean(rB>rA+0.01)*100:.0f}%  (C прогноз = B)')
print('-'*74)
print(f'СТРУКТУРА ОДЕЯЛА (среднее по {len(bl)} датасетам):')
print(f'  НУТРО (порядок μ_pp):  {bl[:,0].mean():.0%}   ОДЕЯЛО (граница μ_sc): {bl[:,1].mean():.0%}   СРЕДА (хаос μ_ac): {bl[:,2].mean():.0%}')
print(f'  корр(доля НУТРА, точность B) = {np.corrcoef(byo[:,0],byo[:,1])[0,1]:+.2f}  <- одеяло предсказывает применимость?')
print('='*74)
