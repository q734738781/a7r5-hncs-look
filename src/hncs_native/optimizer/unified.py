"""Joint projection onto the existing 45 native controls and 405 RGB grid nodes.

Analytic chain rule includes movement through the grid as native controls change.
All image statistics are fixed mathematical constraints, not learned features.
"""
import itertools
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from hncs_native.grid import smoothness
from .operator_math import forward_jacobian, lab_jacobian
try:
    from .de00_compiled import residual_jacobian,residual_values
except ImportError:
    from .de00_differential import residual_jacobian
    def residual_values(predicted,target):return residual_jacobian(predicted,target)[0]
from .reverse import forward_pullback,lab_pullback

KR, KB = .299, .114
KG = 1-KR-KB
COORD = np.array([[KR,KG,KB],[-KR/(2*(1-KB)),-KG/(2*(1-KB)),.5],
                  [.5,-KG/(2*(1-KR)),-KB/(2*(1-KR))]])
COORD = COORD * (255/np.array([219,224,224])*[4,8,8])[:,None]
OFFSET = np.array([-16/219*4,4,4])

def grid_forward_reference(rgb, grid):
    """Return clipped output, d(output)/d(input), and sparse d(raw)/d(grid)."""
    at0 = rgb @ COORD.T + OFFSET
    at = np.clip(at0,0,[4,8,8]); lo = np.minimum(at.astype(int),[3,7,7]); f = at-lo
    gate = (at0>0)&(at0<[4,8,8])
    ids=[]; weights=[]; deriv=np.zeros((len(rgb),3,3))
    for bits in itertools.product((0,1),repeat=3):
        bits=np.array(bits); t=np.where(bits,f,1-f)
        ix=(lo[:,0]+bits[0])*81+(lo[:,1]+bits[1])*9+lo[:,2]+bits[2]
        w=t.prod(1); dw=np.empty_like(t)
        for axis in range(3):
            other=[c for c in range(3) if c!=axis]
            dw[:,axis]=(1 if bits[axis] else -1)*t[:,other].prod(1)*gate[:,axis]
        deriv += np.einsum('no,ni->noi',grid[ix],dw@COORD)
        ids.append(ix); weights.append(w)
    ids=np.stack(ids,1); weights=np.stack(weights,1)
    A=coo_matrix((weights.ravel(),(np.repeat(np.arange(len(rgb)),8),ids.ravel())),shape=(len(rgb),405)).tocsr()
    raw=rgb+A@grid; active=(raw>0)&(raw<1)
    return np.clip(raw,0,1),(deriv+np.eye(3))*active[:,:,None],A,active

try:
    from .grid_compiled import kernel as grid_kernel
except ImportError:
    grid_kernel=None

def grid_forward(rgb,grid):
    if grid_kernel is None:return grid_forward_reference(rgb,grid)
    rgb=np.asarray(rgb,float)
    out,j,ids,w,active=grid_kernel(rgb,np.asarray(grid,float),COORD,rgb@COORD.T+OFFSET)
    A=csr_matrix((w,ids,np.arange(0,len(w)+1,8,dtype=np.int32)),shape=(len(rgb),405))
    return out,j,A,active

def pseudo(res, weight, delta=4.):
    q=np.sum(res*res,axis=1); den=np.sqrt(1+q/delta**2)
    return float(np.sum(weight*2*delta**2*(den-1))), 2*res*(weight/den)[:,None]

class AppearanceLoss:
    def __init__(self,target,weights,P,pweights,reference=None):
        self.use_lab_pullback=True
        self.target=lab_jacobian(target)[0]
        self.w=weights/weights.sum();self.P=P;self.pw=pweights/pweights.sum()
        self.mean=P@self.target
        self.std=np.sqrt(np.maximum(P@(self.target**2)-self.mean**2,0)+.25)
        self.scale=np.array([1.,1.,1.])
        self.reference_distance=None
        if reference is not None:
            ref=lab_jacobian(reference)[0]
            self.reference_distance=np.linalg.norm((P@ref-self.mean)/self.scale,axis=1)

    def __call__(self,rgb,gradient=True):
        if not gradient:
            from .fit import lab
            lv=lab(rgb);res=residual_values(lv,self.target)
            cost=.2*pseudo(res,self.w)[0]
            cost+=.3*pseudo((lv-self.target)/self.scale,self.w)[0]
            mean=self.P@lv;pr=(mean-self.mean)/self.scale
            cost+=.35*pseudo(pr,self.pw)[0]
            std=np.sqrt(np.maximum(self.P@(lv*lv)-mean*mean,0)+.25)
            cost+=.15*pseudo((std-self.std)/self.scale,self.pw)[0]
            if self.reference_distance is not None:
                excess=np.maximum(np.linalg.norm(pr,axis=1)-self.reference_distance-.3,0)
                cost+=.5*(self.pw@(excess*excess))
            return float(cost)
        if self.use_lab_pullback:lv,back=lab_pullback(rgb)
        else:lv,lj=lab_jacobian(rgb)
        res,rj=residual_jacobian(lv,self.target)
        cost,gr=pseudo(res,self.w)
        cost*=.2; gl=.2*np.einsum('ni,nij->nj',gr,rj)
        # Absolute a/b differences avoid discounting saturation errors by chroma.
        c,g=pseudo((lv-self.target)/self.scale,self.w)
        cost+=.3*c;gl+=.3*g/self.scale
        mean=self.P@lv;pr=(mean-self.mean)/self.scale
        c,g=pseudo(pr,self.pw)
        cost+=.35*c;gl+=self.P.T@(.35*g/self.scale)
        # Contrast and colour spread cannot cancel inside regional averages.
        std=np.sqrt(np.maximum(self.P@(lv*lv)-mean*mean,0)+.25)
        c,g=pseudo((std-self.std)/self.scale,self.pw)
        cost+=.15*c;gs=.15*g/self.scale/std
        gl+=lv*(self.P.T@gs)-self.P.T@(gs*mean)
        # Soft guard against worsening already-good palette regions; no frozen coefficients.
        if self.reference_distance is not None:
            dist=np.linalg.norm(pr,axis=1)
            excess=np.maximum(dist-self.reference_distance-.3,0)
            c=float(self.pw@(excess**2));cost+=.5*c
            gl+=self.P.T@((self.pw*excess/np.maximum(dist,1e-12))[:,None]*pr/self.scale)
        if not gradient:return float(cost)
        return float(cost),back(gl) if self.use_lab_pullback else np.einsum('ni,nij->nj',gl,lj)

class JointObjective:
    def __init__(self,batches,loss,initial,reverse=True,*,prior_strength=.002,grid_smooth=.30,grid_ridge=.005,endpoint_smooth=.01):
        self.batches=batches;self.loss=loss;self.initial=initial
        self.L=smoothness();self.calls=0
        self.scale=np.r_[np.full(45,.1),np.full(1215,.03)]
        self.reverse=reverse
        self.last_z=None;self.last_result=None
        self.prior_strength=prior_strength;self.grid_smooth=grid_smooth
        self.grid_ridge=grid_ridge;self.endpoint_smooth=endpoint_smooth

    def unpack(self,z):
        p=self.initial+z*self.scale
        return p[:45],p[45:].reshape(405,3)

    def __call__(self,z):
        if self.last_z is not None and np.array_equal(z,self.last_z):
            return self.last_result
        theta,grid=self.unpack(z);out=[];jac=[]
        try:
            for r,x,prepared in self.batches:
                forward=forward_pullback if self.reverse else forward_jacobian
                a,j=forward(r,x,theta,prepared)
                out.append(a);jac.append(j)
        except ValueError:
            return 1e12+float(z@z),2*z
        src=np.concatenate(out)
        rgb,G,A,active=grid_forward(src,grid)
        cost,g=self.loss(rgb)
        if self.reverse:
            upstream=np.einsum('no,noi->ni',g,G)
            native=np.zeros(45);offset=0
            for a,pullback in zip(out,jac):
                native+=pullback(upstream[offset:offset+len(a)])
                offset+=len(a)
        else:
            native=np.einsum('no,noi,nip->p',g,G,np.concatenate(jac),optimize=True)
        gg=A.T@(g*active)
        # Smooth grid and endpoint controls, with a weak minimum-change prior.
        smooth=self.grid_smooth
        cost+=smooth*np.sum(grid*(self.L@grid))+self.grid_ridge*np.sum(grid*grid)
        gg+=2*smooth*(self.L@grid)+2*self.grid_ridge*grid
        change=theta-self.initial[:45]
        cost+=self.prior_strength*np.sum(change*change);native+=2*self.prior_strength*change
        for start,width in ((3,6),(21,8)):
            a=theta[start:start+3*width].reshape(3,width)
            d=np.diff(a,axis=0);cost+=self.endpoint_smooth*np.sum(d*d)
            q=np.zeros_like(a);q[:-1]-=2*self.endpoint_smooth*d;q[1:]+=2*self.endpoint_smooth*d
            native[start:start+3*width]+=q.ravel()
        self.calls+=1
        self.last_z=np.array(z,copy=True)
        self.last_result=(float(cost),np.r_[native,gg.ravel()]*self.scale)
        return self.last_result
