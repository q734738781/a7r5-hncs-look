"""Allocation-light evaluation of the audited HSMs; reference remains untouched.

No reduced-resolution LUT approximation or fast-math is used. This backend is
only for full-photo review and is independently compared to the reference.
"""
import numpy as np
from numba import njit,prange
from .reference import HNCSReference
from .phocus import HNCSTeacher
from hncs_core.adobe_triple_illuminant import temperature_tint_to_xy

@njit(cache=True,parallel=True)
def hsm_kernel(rgb,grid,hdr):
    nv,nh,ns,_=grid.shape;out=np.empty_like(rgb)
    for i in prange(len(rgb)):
        r=max(rgb[i,0],0.);g=max(rgb[i,1],0.);b=max(rgb[i,2],0.)
        if hdr and nv>1:
            r=r*(256.+r)/(256.*(1.+r));g=g*(256.+g)/(256.*(1.+g));b=b*(256.+b)/(256.*(1.+b))
        v=max(r,g,b);mn=min(r,g,b);gap=v-mn;h=0.;s=0.
        if gap>0:
            s=gap/v
            if r==v:
                h=(g-b)/gap
                if h<0:h+=6
            elif g==v:h=2+(b-r)/gap
            else:h=4+(r-g)/gap
        hp=h*nh/6.;sp=min(max(s,0.),1.)*(ns-1)
        hb=np.floor(hp);h0=int(hb)%nh;hf=hp-hb;s0=min(int(np.floor(sp)),ns-2);sf=sp-s0
        v0=0;vf=0.
        if nv>1:
            vp=min(max(v,0.),1.)*(nv-1);v0=min(int(np.floor(vp)),nv-2);vf=vp-v0
        d0=0.;d1=0.;d2=0.
        for vi in range(2):
            vv=min(v0+vi,nv-1);vw=vf if vi else 1-vf
            for hi in range(2):
                hh=(h0+hi)%nh;hw=hf if hi else 1-hf
                for si in range(2):
                    ss=s0+si;sw=sf if si else 1-sf;weight=(vw*hw)*sw
                    d0+=grid[vv,hh,ss,0]*weight;d1+=grid[vv,hh,ss,1]*weight;d2+=grid[vv,hh,ss,2]*weight
        h=(h+d0/60.)%6.;s=min(s*d1,1.);v=min(max(v*d2,0.),1.)
        if s<=0:ar=ag=ab=v
        else:
            sector=int(h);f=h-sector;p=v*(1-s);q=v*(1-s*f);t=v*(1-s*(1-f))
            if sector==0:ar,ag,ab=v,t,p
            elif sector==1:ar,ag,ab=q,v,p
            elif sector==2:ar,ag,ab=p,v,t
            elif sector==3:ar,ag,ab=p,q,v
            elif sector==4:ar,ag,ab=t,p,v
            else:ar,ag,ab=v,p,q
        if hdr and nv>1:
            z=min(max(ar,0.),1.);ar=16.*((8.*z)-8.+np.sqrt(max(64.*z*z-127.*z+64.,0.)))
            z=min(max(ag,0.),1.);ag=16.*((8.*z)-8.+np.sqrt(max(64.*z*z-127.*z+64.,0.)))
            z=min(max(ab,0.),1.);ab=16.*((8.*z)-8.+np.sqrt(max(64.*z*z-127.*z+64.,0.)))
        out[i,0]=ar;out[i,1]=ag;out[i,2]=ab
    return out

def apply(spec,payload,rgb,hdr=False):
    grid=np.asarray(payload,dtype=np.float64).reshape(spec.value,spec.hue,spec.saturation,3)
    return hsm_kernel(np.asarray(rgb,dtype=np.float64).reshape(-1,3),grid,hdr)

class FastHNCSTeacher(HNCSTeacher):
    def color_stages(self,camera_rgb,kelvin=5550,*,white_scale=8192,neutral=None):
        shape=np.shape(camera_rgb);x=np.asarray(camera_rgb).reshape(-1,3)/white_scale
        white=temperature_tint_to_xy(float(kelvin),0) if neutral is None else self.spec.neutral_to_xy(neutral)
        state=self.spec.set_white_xy(white)
        pro=(x*state.camera_white)@state.camera_to_pcs.T@self.pro_from_pcs.T
        stages={'camera_to_prophoto':pro.reshape(shape)};w=state.weights.astype(np.float32)
        payload=(w[0]*self.bases[0]+w[1]*self.bases[1]+w[2]*self.bases[2]).astype(np.float32)
        pro=apply(self.hsm_spec,payload,np.clip(pro,0,1),self.hdr);stages['wb_carrier']=pro.reshape(shape)
        scale=np.maximum(pro.max(axis=1),1);pro=apply(self.adobe_spec,self.adobe,pro/scale[:,None])*scale[:,None];stages['adobe_look']=pro.reshape(shape)
        scale=np.maximum(pro.max(axis=1),1);pro=apply(self.fixed_spec,self.fixed,pro/scale[:,None])*scale[:,None];stages['hncs_color']=pro.reshape(shape)
        stages['hasselblad_linear']=(pro@self.h_from_pro.T).reshape(shape)
        return stages

class FastHNCSReference(HNCSReference):
    def __init__(self,main,st):
        super().__init__(main,st);self.colour=FastHNCSTeacher()
