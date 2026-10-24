"""Piecewise analytic Sony operator and Jacobian; no learned pixel model.

The 45 coordinates are the existing REA controls in normalized units. The
derivative is with respect to controls, so fixed RAW SSCS/hue selection remains
inside the known forward operator. At clipping/knot boundaries the derivative
is one active branch, not a claim of global differentiability.
"""
import numpy as np
from scipy.optimize import lsq_linear
from .fit import from_vector,lab
from hncs_native.sony.rea_adapter import expand_matrix,cp_to_raw_cc1

LOW=np.r_[[-.75,-.8,-.8],np.full(42,-.5)]
HIGH=np.r_[[.4,.8,.8],np.full(42,.5)]
YCC_TO_RGB=np.array([[1,1.402,0],[1,-.7141,-.3441],[1,0,1.772]])
CP_TO_RAW=np.array([3,0,1,2,7,4,5,6])

def interpolate(q,x,y):
    """Linear interpolation, value and derivative; y may have trailing axes."""
    q=np.asarray(q);x=np.asarray(x);y=np.asarray(y)
    idx=np.clip(np.searchsorted(x,q,side='right')-1,0,len(x)-2)
    f=np.clip((q-x[idx])/(x[idx+1]-x[idx]),0,1)
    shape=(...,)+(None,)*(y.ndim-1)
    value=y[idx]+f[shape]*(y[idx+1]-y[idx])
    slope=(y[idx+1]-y[idx])/(x[idx+1]-x[idx])[shape]
    slope=slope*((q>x[0])&(q<x[-1]))[shape]
    return value,slope

def gamma_basis(r):
    y=np.r_[0,r.gamma_y].astype(float);t=y/y[-1]
    return np.r_[0,r.gamma_x].astype(float)/128, np.stack([y,y*(1-t)**2,y*t*(1-t)],-1)/16

def matrix_design(r,x):
    n=len(x);d=np.zeros((n,3,18))
    for e,w in enumerate(r.wb_weights):
        for c,(a,b) in enumerate(((1,2),(0,2),(0,1))):
            d[:,c,e*6+c*2]=w*(x[:,c]-x[:,a])
            d[:,c,e*6+c*2+1]=w*(x[:,c]-x[:,b])
    return d

def forward_jacobian(r,source,theta,prepared=None):
    theta=np.asarray(theta,dtype=float)
    x,base,h,thresholds=r.prepare(source) if prepared is None else prepared
    if x.ndim!=2:raise ValueError('N x 3 collocation points required')
    n=len(x);D=np.zeros((n,3,45));D[:,:,3:21]=matrix_design(r,x)
    v=base.astype(float)+np.einsum('nij,j->ni',D,theta)
    D*=((v>0)&(v<16383))[:,:,None];v=np.clip(v,0,16383)
    knots,basis=gamma_basis(r);gy=np.r_[0,r.gamma_y].astype(float)/16+basis@theta[:3]
    active=(gy>0)&(gy<0x3ffff/16);gy=np.clip(gy,0,0x3ffff/16)
    if np.any(np.diff(gy)<0):raise ValueError('Nonmonotonic native Gamma')
    enc,slope=interpolate(v,knots,gy)
    D*=slope[:,:,None]
    dg,_=interpolate(v,knots,basis*active[:,None]);D[:,:,:3]+=dg
    p=r.cc1.astype(float)+cp_to_raw_cc1(r.wb_weights@theta[21:].reshape(3,8)*512)
    dp=np.zeros((8,45))
    for e,w in enumerate(r.wb_weights):
        dp[np.arange(8),21+e*8+CP_TO_RAW]=512*w
    y=enc@(r.luma/64);dy=np.einsum('ncp,c->np',D,r.luma/64)
    u=enc[:,0]-enc[:,1];v=enc[:,2]-enc[:,1]
    du=D[:,0]-D[:,1];dv=D[:,2]-D[:,1]
    ci=np.where(u<0,3,1);ri=np.where(v<0,2,0)
    cb=v+p[ci]/1024*u;dcb=dv+p[ci,None]/1024*du+u[:,None]*dp[ci]/1024
    cr=u+p[ri]/1024*v;dcr=du+p[ri,None]/1024*dv+v[:,None]*dp[ri]/1024
    gi=np.where(cr<0,7,5);bi=np.where(cb<0,6,4)
    dcr=dcr*p[gi,None]/1024+cr[:,None]*dp[gi]/1024
    dcb=dcb*p[bi,None]/1024+cb[:,None]*dp[bi]/1024
    cr=cr*p[gi]/1024;cb=cb*p[bi]/1024
    dcr*=((cr>-8192)&(cr<8191))[:,None];dcb*=((cb>-8192)&(cb<8191))[:,None]
    cr=np.clip(cr,-8192,8191);cb=np.clip(cb,-8192,8191)
    a,b,c,d=r.cs
    f0=np.where(y<a,255-(a-y)*c/4096,np.where(y>b,255-(y-b)*d/4096,255))
    f=np.clip(f0,0,255)/256
    df=np.where(y<a,c/4096,np.where(y>b,-d/4096,0))/256
    df*=((f0>0)&(f0<255))
    dcr=f[:,None]*dcr+(cr*df)[:,None]*dy
    dcb=f[:,None]*dcb+(cb*df)[:,None]*dy
    cr*=f;cb*=f
    yy,slope=interpolate(y,np.arange(32768),r.ylut[32768:].view(np.uint16).astype(float))
    yy=(yy-r.yoff)*r.ygain+r.yoff
    dyy=dy*(slope*r.ygain*((yy>0)&(yy<16383)))[:,None];yy=np.clip(yy,0,16383)
    rgb=np.stack([yy+1.402*cr,yy-.7141*cr-.3441*cb,yy+1.772*cb],-1)/16383
    jac=np.stack([dyy+1.402*dcr,dyy-.7141*dcr-.3441*dcb,dyy+1.772*dcb],1)/16383
    jac*=((rgb>0)&(rgb<1))[:,:,None]
    return np.clip(rgb,0,1),jac

def lab_jacobian(rgb):
    x=np.clip(np.asarray(rgb),0,1)
    lin=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    dl=np.where(x<=.04045,1/12.92,2.4/1.055*((x+.055)/1.055)**1.4)
    m=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.072175],[.0193339,.119192,.9503041]])/np.array([.95047,1,1.08883])[:,None]
    q=lin@m.T;d=6/29
    df=np.where(q>d**3,1/(3*np.maximum(q,1e-20)**(2/3)),1/(3*d*d))
    f=np.where(q>d**3,np.cbrt(q),q/(3*d*d)+4/29)
    L=np.array([[0,116,0],[500,-500,0],[0,200,-200]])
    val=f@L.T+[-16,0,0]
    jac=np.einsum('ab,nb,bc,nc->nac',L,df,m,dl)
    return val,jac

def de00_local_metric(labs):
    """K such that ||K dLab||² is the local CIEDE2000 quadratic form.

    Derived by taking the coincident-color limit of Sharma/Wu/Dalal 2005.
    It is a tangent metric, NOT a globally exact DeltaE00 replacement.
    """
    L,a,b=np.asarray(labs).T;C=np.hypot(a,b)
    G=.5*(1-np.sqrt(C**7/(C**7+25**7)))
    ap=(1+G)*a;Cp=np.hypot(ap,b);h=np.mod(np.arctan2(b,ap),2*np.pi)
    T=1-.17*np.cos(h-np.pi/6)+.24*np.cos(2*h)+.32*np.cos(3*h+np.pi/30)-.20*np.cos(4*h-63*np.pi/180)
    SL=1+.015*(L-50)**2/np.sqrt(20+(L-50)**2);SC=1+.045*Cp;SH=1+.015*Cp*T
    RT=-2*np.sqrt(Cp**7/(Cp**7+25**7))*np.sin(2*np.pi/6*np.exp(-((h*180/np.pi-275)/25)**2))
    den=np.maximum(Cp,1e-20);B=np.zeros((len(L),3,3));B[:,0,0]=1/SL
    B[:,1,1]=ap*(1+G)/den/SC;B[:,1,2]=b/den/SC
    B[:,2,1]=-b*(1+G)/den/SH;B[:,2,2]=ap/den/SH
    neutral=Cp<1e-8;B[neutral,1,1]=1.5;B[neutral,2,2]=1
    M=np.tile(np.eye(3),(len(L),1,1));M[:,1,2]=M[:,2,1]=RT/2
    return np.swapaxes(np.linalg.cholesky(M),1,2)@B

def inverse_monotone(query,values):
    """Invert rising linear segments; explicitly flag non-invertible plateaus."""
    q=np.asarray(query);values=np.asarray(values)
    if np.any(np.diff(values)<0):raise ValueError('Nonmonotone inverse requested')
    lo=np.clip(np.searchsorted(values,q,side='right')-1,0,len(values)-2);hi=lo+1
    gap=values[hi]-values[lo]
    x=lo+(q-values[lo])/np.maximum(gap,1e-15)
    valid=(q>values[0])&(q<values[-1])&(gap>0)
    plateaus=np.unique(values[:-1][np.diff(values)==0])
    if len(plateaus):
        i=np.clip(np.searchsorted(plateaus,q),0,len(plateaus)-1)
        distance=np.minimum(np.abs(q-plateaus[i]),np.abs(q-plateaus[np.maximum(i-1,0)]))
        valid&=distance>1e-7
    return np.clip(x,0,len(values)-1),valid

def inverse_tail(r,target):
    """Invert YCC2RGB, monotone YGamma and ChromaSuppress where attainable.

    Returned validity excludes output-clipped pixels and singular suppression.
    Such points remain present in the subsequent full-domain projection/report.
    """
    target=np.asarray(target);out=target*16383@np.linalg.inv(YCC_TO_RGB).T
    vals=np.clip((r.ylut[32768:].view(np.uint16).astype(float)-r.yoff)*r.ygain+r.yoff,0,16383)
    y,invertible=inverse_monotone(out[:,0],vals)
    a,b,c,d=r.cs
    f=np.where(y<a,np.clip(255-(a-y)*c/4096,0,255),np.where(y>b,np.clip(255-(y-b)*d/4096,0,255),255))/256
    valid=(target.min(1)>.001)&(target.max(1)<.999)&(f>.1)&invertible
    return np.stack([y,out[:,1]/np.maximum(f,1e-10),out[:,2]/np.maximum(f,1e-10)],-1),valid

def inverse_cc1(r,ycc,p=None):
    p=r.cc1.astype(float) if p is None else np.asarray(p)
    y,cr,cb=ycc.T
    A=cr*1024/np.where(cr<0,p[7],p[5]);B=cb*1024/np.where(cb<0,p[6],p[4])
    result=np.full_like(ycc,np.nan);found=np.zeros(len(y),bool)
    for su in (-1,1):
        for sv in (-1,1):
            k=p[3 if su<0 else 1]/1024;l=p[2 if sv<0 else 0]/1024;det=1-k*l
            if abs(det)<1e-4:continue
            u=(A-l*B)/det;v=(B-k*A)/det
            valid=(u*su>=-1e-7)&(v*sv>=-1e-7)&(~found)
            g=(y-r.luma[0]/64*u-r.luma[2]/64*v)/(r.luma.sum()/64)
            result[valid]=np.stack([g+u,g,g+v],-1)[valid];found|=valid
    return result,found

def solve_gamma(states,teacher):
    A=[];b=[];retained=[]
    for state in states:
        r=state['renderer'];x=np.repeat(np.geomspace(8,12500,240)[:,None],3,1)
        target=teacher.render(x,state['kelvin'],neutral=state['neutral'])
        yc,ok=inverse_tail(r,target);prepared=r.prepare(x);base=prepared[1]
        # Neutral invariance: every allowed matrix increment has zero row sum.
        knots,B=gamma_basis(r);a,_=interpolate(base.mean(1),knots,B)
        g,_=interpolate(base.mean(1),knots,np.r_[0,r.gamma_y]/16)
        ok&=(base.min(1)>0)&(base.max(1)<16383)
        A.append(a[ok]/16383);b.append((yc[ok,0]-g[ok])/16383)
        retained.append(int(ok.sum()))
    A=np.vstack(A);b=np.concatenate(b)
    fit=lsq_linear(A,b,bounds=(LOW[:3],HIGH[:3]),tol=1e-12)
    theta=np.zeros(45);theta[:3]=fit.x
    from_vector(theta).validate()
    return theta,{'method':'bounded linear least squares on analytically inverted neutral axis',
       'gray_points':len(b),'retained_per_state':retained,'rank':int(np.linalg.matrix_rank(A)),
       'singular_values':np.linalg.svd(A,compute_uv=False).tolist(),'rms_normalized_gamma':float(np.sqrt(np.mean((A@fit.x-b)**2)))}

def solve_matrix(batches,theta,ridge=1e-5):
    A=[];b=[];retained=[]
    for batch in batches:
        r,x,target,weight=batch
        yc,ok=inverse_tail(r,target);enc,ccok=inverse_cc1(r,yc);ok&=ccok
        knots,B=gamma_basis(r);g=np.r_[0,r.gamma_y]/16+B@theta[:3]
        indices,gvalid=inverse_monotone(np.nan_to_num(enc),g)
        z=np.interp(indices,np.arange(len(knots)),knots)
        ok&=gvalid.all(1)&(yc[:,1:].min(1)>-8192)&(yc[:,1:].max(1)<8191)
        xx,base,_,_=r.prepare(x);design=matrix_design(r,xx)/16383
        w=np.sqrt(weight[ok])[:,None,None]
        A.append((design[ok]*w).reshape(-1,18));b.append(((z-base)[ok]/16383*w[:,:,0]).ravel())
        retained.append({'total':len(x),'inverse_valid':int(ok.sum())})
    A=np.vstack(A);b=np.concatenate(b);s=np.linalg.svd(A,compute_uv=False)
    solution=lsq_linear(np.vstack([A,np.eye(18)*np.sqrt(ridge)]),np.r_[b,np.zeros(18)],bounds=(LOW[3:21],HIGH[3:21]),tol=1e-12)
    result=theta.copy();result[3:21]=solution.x
    return result,{'method':'piecewise tail/CC1/Gamma inverse then bounded linear matrix projection',
       'retained':retained,'rank':int(np.linalg.matrix_rank(A)),'singular_values':s.tolist(),
       'ridge':ridge,'rms_normalized_linear_rgb':float(np.sqrt(np.mean((A@solution.x-b)**2)))}
