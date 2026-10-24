"""Parameter coordinates and display Lab; no learned colour model."""
import numpy as np
from colour.difference import delta_E_CIE2000
from hncs_native.sony.rea_adapter import NativeLook

def lab(rgb):
    rgb=np.clip(np.asarray(rgb),0,1)
    lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    xyz=lin @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.072175],[.0193339,.119192,.9503041]]).T
    q=xyz/np.array([.95047,1,1.08883]); d=6/29
    f=np.where(q>d**3,np.cbrt(q),q/(3*d*d)+4/29)
    return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])),-1)

def metrics(a,b):
    a=np.asarray(a).reshape(-1,3); b=np.asarray(b).reshape(-1,3)
    de=delta_E_CIE2000(lab(a),lab(b))
    return dict(n=len(a),mean_de00=float(de.mean()),median_de00=float(np.median(de)),
                p95_de00=float(np.percentile(de,95)),max_de00=float(de.max()),
                rgb_mae_255=float(np.abs(a-b).mean()*255))

def from_vector(p,name='HNCS'):
    p=np.asarray(p)
    if len(p)==17:
        return NativeLook(name,p[:3]*1024,np.tile(p[3:9]*1024,(3,1)),np.tile(p[9:]*512,(3,1)))
    if len(p)==45:
        return NativeLook(name,p[:3]*1024,p[3:21].reshape(3,6)*1024,p[21:].reshape(3,8)*512)
    raise ValueError('Expected shared 17-control or endpoint 45-control vector')

def vector_of(look,endpoint=False):
    return np.r_[look.tone_q10/1024,look.matrix_planes.ravel()/1024 if endpoint else look.matrix_planes[0]/1024,
                 look.chroma_planes.ravel()/512 if endpoint else look.chroma_planes[0]/512]
