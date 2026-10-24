"""Same trilinear grid, without the full-image N x 8 x 3 allocation."""
import numpy as np
from numba import njit,prange
from hncs_native.grid import coordinates

@njit(cache=True,parallel=True)
def kernel(rgb,grid,at):
    out=np.empty_like(rgb)
    for i in prange(len(rgb)):
        f=np.empty(3);lo=np.empty(3,np.int32)
        for k in range(3):
            top=4 if k==0 else 8;v=min(max(at[i,k],0.),top);lo[k]=min(int(v),top-1);f[k]=v-lo[k]
        d=np.zeros(3)
        for corner in range(8):
            b0=corner//4;b1=(corner//2)%2;b2=corner%2
            w=((f[0] if b0 else 1-f[0])*(f[1] if b1 else 1-f[1]))*(f[2] if b2 else 1-f[2])
            ix=(lo[0]+b0)*81+(lo[1]+b1)*9+lo[2]+b2
            for k in range(3):d[k]+=grid[ix,k]*w
        for k in range(3):out[i,k]=min(max(rgb[i,k]+d[k]/256,0.),1.)
    return out

def render_fast(renderer,x,look=None,*,prepared=None):
    if look is None or look.grid_model=='disabled':return renderer.render(x,look,prepared=prepared)
    core=look.quantized();grid=core.grid.copy();core.grid[:]=0;core.grid_model='disabled'
    source=renderer.render(x,core,prepared=prepared);shape=source.shape;rgb=source.reshape(-1,3)
    at=coordinates(rgb)*[4,8,8]+[0,4,4]
    return kernel(np.asarray(rgb,np.float64),np.asarray(grid,np.float64),np.asarray(at,np.float64)).reshape(shape)
