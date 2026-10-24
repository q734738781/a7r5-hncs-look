"""Same limited-range BT.601 trilinear grid; compiled scalar arithmetic."""
import numpy as np
from numba import njit,prange

@njit(cache=True,parallel=True)
def kernel(rgb,grid,coord,at0):
    n=len(rgb);out=np.empty((n,3));deriv=np.empty((n,3,3))
    ids=np.empty(n*8,np.int32);weights=np.empty(n*8)
    active=np.empty((n,3),np.bool_)
    for row in prange(n):
        f=np.empty(3);lo=np.empty(3,np.int32);gate=np.empty(3)
        for a in range(3):
            maximum=4 if a==0 else 8
            value=at0[row,a]
            gate[a]=1. if 0<value<maximum else 0.
            value=min(max(value,0.),maximum)
            lo[a]=min(int(value),maximum-1);f[a]=value-lo[a]
        correction=np.zeros(3);d=np.zeros((3,3))
        for corner in range(8):
            b0=corner//4;b1=(corner//2)%2;b2=corner%2
            t0=f[0] if b0 else 1-f[0]
            t1=f[1] if b1 else 1-f[1]
            t2=f[2] if b2 else 1-f[2]
            index=(lo[0]+b0)*81+(lo[1]+b1)*9+lo[2]+b2
            w=t0*t1*t2
            ids[row*8+corner]=index;weights[row*8+corner]=w
            dw0=(1 if b0 else -1)*t1*t2*gate[0]
            dw1=(1 if b1 else -1)*t0*t2*gate[1]
            dw2=(1 if b2 else -1)*t0*t1*gate[2]
            for o in range(3):
                v=grid[index,o];correction[o]+=w*v
                for k in range(3):
                    d[o,k]+=v*(dw0*coord[0,k]+dw1*coord[1,k]+dw2*coord[2,k])
        for o in range(3):
            raw=rgb[row,o]+correction[o]
            active[row,o]=0<raw<1
            out[row,o]=min(max(raw,0.),1.)
            for k in range(3):deriv[row,o,k]=(d[o,k]+(1. if o==k else 0.))*active[row,o]
    return out,deriv,ids,weights,active
