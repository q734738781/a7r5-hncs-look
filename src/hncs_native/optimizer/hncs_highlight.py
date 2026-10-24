"""Target-free constraints on the fitted operator's own exposure response.

No ST outputs, old Look outputs, preferred hue, or image-dependent output mask.
Only unreliable input rays are checked for sudden chroma growth/curvature and
brightness reversal. Exact neutral ramps additionally preserve the gray axis.
"""
import numpy as np
from .highlight import LUMA,CENTRE,project_neutral_grid
from .integer_grid import IntegerGridProblem

class OperatorStabilityLoss:
    def __init__(self,appearance,rays,ray_weights,neutrals,strength=.2):
        self.appearance=appearance;self.rays=np.asarray(rays,int).reshape(-1,5)
        self.rw=np.asarray(ray_weights,float);self.rw/=self.rw.sum()
        self.neutrals=np.asarray(neutrals,int).reshape(-1,80);self.strength=float(strength)
        self.ids=np.r_[self.rays.ravel(),self.neutrals.ravel()]
        assert len(np.unique(self.ids))==len(self.ids)
        self.nray=self.rays.size

    def guard_compact(self,a,gradient=True):
        c=a[:self.nray].reshape(-1,5,3);gray=a[self.nray:].reshape(-1,80,3)
        dc=np.zeros_like(c);dg=np.zeros_like(gray)
        # Neutrality is symmetric in RGB, with no Sony colour reference.
        centered=gray-gray.mean(-1,keepdims=True)
        cost=1000*np.mean(centered**2);dg+=2000*centered/centered.size
        white=np.maximum(.98-gray[:,-1],0)
        cost+=1000*np.mean(white**2);dg[:,-1]-=2000*white/white.size
        gray_y=gray@LUMA;rev=np.maximum(gray_y[:,:-1]-gray_y[:,1:]-1/1024,0)
        cost+=1000*np.mean(rev**2)
        d=2000*rev/rev.size;dg[:,:-1]+=d[...,None]*LUMA;dg[:,1:]-=d[...,None]*LUMA
        chroma=c-c.mean(-1,keepdims=True)
        norm=np.sqrt(np.sum(chroma**2,axis=-1)+1e-10)
        # Above the observed clipped exposure, allow modest chroma growth but
        # penalize a sudden increase; there is no absolute saturation target.
        growth=np.maximum(norm[:,3:]-norm[:,2:4]-.025,0)
        cost+=1000*np.sum(self.rw[:,None]*growth**2)/2
        d=1000*self.rw[:,None]*growth
        dc[:,3:]+=d[...,None]*chroma[:,3:]/norm[:,3:,None]
        dc[:,2:4]-=d[...,None]*chroma[:,2:4]/norm[:,2:4,None]
        # Equal log-exposure spacing, central second difference. A dead zone
        # allows the intended film shoulder instead of fitting a straight line.
        curvature=chroma[:,:-2]-2*chroma[:,1:-1]+chroma[:,2:]
        norm2=np.sqrt(np.sum(curvature**2,axis=-1)+1e-10)
        excess=np.maximum(norm2-.04,0)
        cost+=300*np.sum(self.rw[:,None]*excess**2)/3
        d=200*self.rw[:,None,None]*excess[...,None]*curvature/norm2[...,None]
        dc[:,:-2]+=d;dc[:,1:-1]-=2*d;dc[:,2:]+=d
        y=c@LUMA;rev=np.maximum(y[:,:-1]-y[:,1:]-.004,0)
        cost+=1000*np.sum(self.rw[:,None]*rev**2)/4
        d=500*self.rw[:,None]*rev;dc[:,:-1]+=d[...,None]*LUMA;dc[:,1:]-=d[...,None]*LUMA
        if not gradient:return float(self.strength*cost)
        return float(self.strength*cost),self.strength*np.concatenate([dc.reshape(-1,3),dg.reshape(-1,3)])

    def guard(self,rgb,gradient=True):
        if not gradient:return self.guard_compact(rgb[self.ids],False)
        value,g=self.guard_compact(rgb[self.ids]);out=np.zeros_like(rgb);out[self.ids]=g
        return value,out

    def __call__(self,rgb,gradient=True):
        if not gradient:return self.appearance(rgb,False)+self.guard(rgb,False)
        a,g=self.appearance(rgb);b,h=self.guard(rgb);return a+b,g+h

class HncsIntegerGridProblem(IntegerGridProblem):
    def __init__(self,src,loss,L,grid,*,grid_smooth=.3):
        np.testing.assert_array_equal(project_neutral_grid(grid),grid)
        self.stability=loss;self.lookup=np.full(len(src),-1,int);self.lookup[loss.ids]=np.arange(len(loss.ids))
        super().__init__(src,loss.appearance,L,grid,grid_smooth=grid_smooth,grid_ridge=0.)
        self.compact=np.clip(self.raw[loss.ids],0,1)
        self.guard_value=loss.guard_compact(self.compact,False);self.value+=self.guard_value
        np.testing.assert_allclose(self.value,self.full_value(),atol=1e-8,rtol=1e-11)

    def trial_vector(self,node,steps,accept=False):
        change=np.asarray(steps)/256
        if node in CENTRE[[0,4]] and np.any(change):return np.inf
        if node in CENTRE and np.ptp(change)>0:return np.inf
        if np.any(self.g[node]+change<-.25) or np.any(self.g[node]+change>63/256):return np.inf
        idx,w,_,_=self.support[node];take=self.lookup[idx]>=0
        if np.any(take):
            compact=self.compact.copy();compact[self.lookup[idx[take]]]=np.clip(self.raw[idx[take]]+w[take,None]*change,0,1)
            guard=self.stability.guard_compact(compact,False)
        else:compact=self.compact;guard=self.guard_value
        delta=guard-self.guard_value;value=super().trial_vector(node,steps,accept)
        if accept:self.compact=compact;self.guard_value=guard;self.value+=delta;return self.value
        return value+delta

    def full_value(self):return super().full_value()+self.stability.guard(np.clip(self.raw,0,1),False)

    def trial_block(self,*args,**kwargs):raise NotImplementedError('Only constrained single-node moves are implemented')

    def sweep(self,step=1,tolerance=1e-11):
        accepted=0;start=self.value
        for node in range(405):
            if node in CENTRE[[0,4]]:continue
            directions=[np.ones(3,dtype=int)*step] if node in CENTRE else list(np.eye(3,dtype=int)*step)
            for d in directions:
                a=self.trial_vector(node,-d);b=self.trial_vector(node,d)
                if min(a,b)<self.value-tolerance:
                    self.trial_vector(node,-d if a<=b else d,True);accepted+=1
        exact=self.full_value();np.testing.assert_allclose(self.value,exact,rtol=1e-10,atol=1e-8);self.value=exact
        return dict(accepted=accepted,before=start,after=exact,improvement=start-exact)
