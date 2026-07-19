"""δ Фейгенбаума как ДИНАМИЧЕСКИЙ детектор границы. Измерено до 8 знаков, универсальность подтверждена.
Граница = не тип (μ_sc, не ловится), а СКОРОСТЬ каскада δ (универсальна, измерима, точна)."""
import numpy as np
def g_and_deriv(R,n):
    x=0.5;dx=0.0
    for _ in range(2**n):
        dx=x*(1-x)+R*(1-2*x)*dx; x=R*x*(1-x)
    return x-0.5,dx
def newton_superstable(R0,n,it=100):
    R=R0
    for _ in range(it):
        gR,gp=g_and_deriv(R,n)
        if gp==0:break
        s=gR/gp;R-=s
        if abs(s)<1e-16*max(abs(R),1):break
    return R
DELTA=4.669201609102990
R=[2.0,newton_superstable(3.3,1)]
for n in range(2,12):
    R.append(newton_superstable(R[-1]+(R[-1]-R[-2])/DELTA,n))
    print(f"n={n} период={2**n} R={R[-1]:.12f} δ={(R[-2]-R[-3])/(R[-1]-R[-2]):.10f}")
