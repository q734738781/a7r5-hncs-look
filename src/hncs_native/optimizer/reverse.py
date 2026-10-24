"""Exact reverse chain rule for the established piecewise Sony operator.

No change to its mathematics or parameters. Retain only N x 3 intermediate
arrays instead of materializing the N x 3 x 45 forward Jacobian.
"""
import numpy as np
from .operator_math import interpolate, gamma_basis, CP_TO_RAW
from hncs_native.sony.rea_adapter import cp_to_raw_cc1

def lab_pullback(rgb):
    """The same Lab conversion with a reverse product instead of N x 3 x 3."""
    x=np.clip(np.asarray(rgb),0,1)
    lin=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    dl=np.where(x<=.04045,1/12.92,2.4/1.055*((x+.055)/1.055)**1.4)
    m=np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.072175],[.0193339,.119192,.9503041]])/np.array([.95047,1,1.08883])[:,None]
    q=lin@m.T;d=6/29
    df=np.where(q>d**3,1/(3*np.maximum(q,1e-20)**(2/3)),1/(3*d*d))
    f=np.where(q>d**3,np.cbrt(q),q/(3*d*d)+4/29)
    L=np.array([[0,116,0],[500,-500,0],[0,200,-200]])
    def back(g):return ((g@L)*df)@m*dl
    return f@L.T+[-16,0,0],back


def forward_pullback(r, source, theta, prepared=None):
    x, base, _, _ = r.prepare(source) if prepared is None else prepared
    theta = np.asarray(theta, float)
    pairs = ((1, 2), (0, 2), (0, 1))
    dm = np.stack([np.stack([x[:, c]-x[:, a], x[:, c]-x[:, b]], 1)
                   for c, (a, b) in enumerate(pairs)], 1)
    m = np.tensordot(r.wb_weights, theta[3:21].reshape(3, 3, 2), axes=1)
    v0 = base.astype(float) + np.sum(dm*m, axis=2)
    mv = (v0 > 0) & (v0 < 16383)
    v = np.clip(v0, 0, 16383)
    knots, basis = gamma_basis(r)
    gy = np.r_[0, r.gamma_y].astype(float)/16 + basis @ theta[:3]
    ag = (gy > 0) & (gy < 0x3ffff/16)
    gy = np.clip(gy, 0, 0x3ffff/16)
    if np.any(np.diff(gy) < 0):
        raise ValueError('Nonmonotonic native Gamma')
    enc, slope = interpolate(v, knots, gy)
    dg, _ = interpolate(v, knots, basis*ag[:, None])
    p = r.cc1.astype(float)+cp_to_raw_cc1(r.wb_weights @ theta[21:].reshape(3, 8)*512)
    y = enc @ (r.luma/64)
    u, v = enc[:, 0]-enc[:, 1], enc[:, 2]-enc[:, 1]
    ci, ri = np.where(u < 0, 3, 1), np.where(v < 0, 2, 0)
    cb0, cr0 = v+p[ci]/1024*u, u+p[ri]/1024*v
    gi, bi = np.where(cr0 < 0, 7, 5), np.where(cb0 < 0, 6, 4)
    cr1, cb1 = cr0*p[gi]/1024, cb0*p[bi]/1024
    mr, mb = (cr1 > -8192) & (cr1 < 8191), (cb1 > -8192) & (cb1 < 8191)
    cr1, cb1 = np.clip(cr1, -8192, 8191), np.clip(cb1, -8192, 8191)
    a, b, c, d = r.cs
    f0 = np.where(y < a, 255-(a-y)*c/4096, np.where(y > b, 255-(y-b)*d/4096, 255))
    f = np.clip(f0, 0, 255)/256
    df = np.where(y < a, c/4096, np.where(y > b, -d/4096, 0))/256
    df *= (f0 > 0) & (f0 < 255)
    yy, sy = interpolate(y, np.arange(32768), r.ylut[32768:].view(np.uint16).astype(float))
    yy = (yy-r.yoff)*r.ygain+r.yoff
    sy *= r.ygain*((yy > 0) & (yy < 16383))
    yy = np.clip(yy, 0, 16383)
    cr, cb = cr1*f, cb1*f
    rgb = np.stack([yy+1.402*cr, yy-.7141*cr-.3441*cb, yy+1.772*cb], 1)/16383
    active = (rgb > 0) & (rgb < 1)

    def pullback(g):
        g = g*active/16383
        gr, gb = 1.402*g[:, 0]-.7141*g[:, 1], -.3441*g[:, 1]+1.772*g[:, 2]
        dy = g.sum(1)*sy+(gr*cr1+gb*cb1)*df
        gr, gb = gr*f*mr, gb*f*mb
        dp = np.bincount(gi, gr*cr0/1024, minlength=8)
        dp += np.bincount(bi, gb*cb0/1024, minlength=8)
        gr, gb = gr*p[gi]/1024, gb*p[bi]/1024
        dp += np.bincount(ri, gr*v/1024, minlength=8)
        dp += np.bincount(ci, gb*u/1024, minlength=8)
        du, dv = gr+gb*p[ci]/1024, gb+gr*p[ri]/1024
        de = dy[:, None]*(r.luma/64)
        de[:, 0] += du
        de[:, 1] -= du+dv
        de[:, 2] += dv
        gamma = np.einsum('nc,ncp->p', de, dg)
        dv0 = de*slope*mv
        matrix = np.einsum('nc,nck->ck', dv0, dm)
        matrix = r.wb_weights[:, None, None]*matrix
        cp = np.empty(8)
        cp[CP_TO_RAW] = dp
        chroma = r.wb_weights[:, None]*cp*512
        return np.r_[gamma, matrix.ravel(), chroma.ravel()]

    return np.clip(rgb, 0, 1), pullback
