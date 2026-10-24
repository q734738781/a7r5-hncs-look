"""Exact CIEDE2000 residual and piecewise analytic Lab derivative.

Formula: Sharma, Wu, Dalal (2005), Color Research & Application 30:21-30.
Forward-mode differentials propagate three Lab coordinates through the formula;
this is calculus on a fixed formula, not automatic learning of a colour model.
At neutral/hue branch singularities one selected branch derivative is used.
"""
import numpy as np

class D:
    def __init__(self,v,d=None):
        self.v=np.asarray(v,dtype=float)
        self.d=np.zeros(self.v.shape+(3,)) if d is None else np.asarray(d)
    @staticmethod
    def of(x):return x if isinstance(x,D) else D(x)
    def __add__(self,o):
        o=D.of(o);return D(self.v+o.v,self.d+o.d)
    __radd__=__add__
    def __neg__(self):return D(-self.v,-self.d)
    def __sub__(self,o):return self+-D.of(o)
    def __rsub__(self,o):return D.of(o)+-self
    def __mul__(self,o):
        o=D.of(o);return D(self.v*o.v,self.d*o.v[...,None]+o.d*self.v[...,None])
    __rmul__=__mul__
    def __truediv__(self,o):
        o=D.of(o);return D(self.v/o.v,(self.d-o.d*(self.v/o.v)[...,None])/o.v[...,None])
    def __rtruediv__(self,o):return D.of(o)/self
    def __pow__(self,p):
        v=self.v**p
        # Norm derivative at zero is branch-dependent; choose zero there.
        deriv=np.where(self.v!=0,p*np.maximum(np.abs(self.v),1e-100)**(p-1),0)
        if isinstance(p,int) and p%2==0:deriv*=np.sign(self.v)
        return D(v,self.d*deriv[...,None])

def where(mask,a,b):
    a=D.of(a);b=D.of(b)
    return D(np.where(mask,a.v,b.v),np.where(np.asarray(mask)[...,None],a.d,b.d))
def unary(x,f,df):return D(f(x.v),x.d*df(x.v)[...,None])
def sin(x):return unary(x,np.sin,np.cos)
def cos(x):return unary(x,np.cos,lambda t:-np.sin(t))
def exp(x):return unary(x,np.exp,np.exp)
def atan2(y,x):
    den=x.v*x.v+y.v*y.v
    d=(x.v[...,None]*y.d-y.v[...,None]*x.d)/np.maximum(den,1e-100)[...,None]
    return D(np.arctan2(y.v,x.v),d)

def residual_jacobian(predicted,target):
    x=np.asarray(predicted);t=np.asarray(target);n=len(x)
    v=[]
    for k in range(3):
        d=np.zeros((n,3));d[:,k]=1;v.append(D(x[:,k],d))
    L2,a2,b2=v;L1,a1,b1=[D(t[:,k]) for k in range(3)]
    C1=(a1*a1+b1*b1)**.5;C2=(a2*a2+b2*b2)**.5
    Cbar=(C1+C2)/2;g=.5*(1-(Cbar**7/(Cbar**7+25**7))**.5)
    ap1=(1+g)*a1;ap2=(1+g)*a2
    cp1=(ap1*ap1+b1*b1)**.5;cp2=(ap2*ap2+b2*b2)**.5
    h1=atan2(b1,ap1);h2=atan2(b2,ap2)
    h1=where(h1.v<0,h1+2*np.pi,h1);h2=where(h2.v<0,h2+2*np.pi,h2)
    h1=where(cp1.v==0,0,h1);h2=where(cp2.v==0,0,h2)
    dh=h2-h1;dh=where(dh.v>np.pi,dh-2*np.pi,where(dh.v<-np.pi,dh+2*np.pi,dh))
    zero=cp1.v*cp2.v==0;dh=where(zero,0,dh)
    dL=L2-L1;dC=cp2-cp1;dH=2*(cp1*cp2)**.5*sin(dh/2)
    lm=(L1+L2)/2;cm=(cp1+cp2)/2;hs=h1+h2
    hm=where(np.abs(h1.v-h2.v)<=np.pi,hs/2,where(hs.v<2*np.pi,(hs+2*np.pi)/2,(hs-2*np.pi)/2))
    hm=where(zero,hs,hm)
    T=1-.17*cos(hm-np.pi/6)+.24*cos(2*hm)+.32*cos(3*hm+np.pi/30)-.20*cos(4*hm-63*np.pi/180)
    sl=1+.015*(lm-50)**2/(20+(lm-50)**2)**.5
    sc=1+.045*cm;sh=1+.015*cm*T
    dt=np.pi/6*exp(-((hm*180/np.pi-275)/25)**2)
    rt=-2*(cm**7/(cm**7+25**7))**.5*sin(2*dt)
    l=dL/sl;c=dC/sc;h=dH/sh;q=rt/2
    rr=[l,c+q*h,(1-q*q)**.5*h]
    return np.stack([v.v for v in rr],-1),np.stack([v.d for v in rr],1)
