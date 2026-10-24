"""REA joint native parameter basis, including its integer tone arithmetic.

The published joint patch adds one Q10 row-sum-preserving matrix correction
per illuminant to all 16 hue sectors. Independent 16-sector edits are NOT
supported by that patch; the Sony baseline still has all 16 sectors.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class NativeLook:
    name: str = 'ST'
    tone_q10: np.ndarray = field(default_factory=lambda: np.zeros(3))
    matrix_planes: np.ndarray = field(default_factory=lambda: np.zeros((3, 6)))
    chroma_planes: np.ndarray = field(default_factory=lambda: np.zeros((3, 8)))
    grid: np.ndarray = field(default_factory=lambda: np.zeros((405, 3)))
    grid_model: str = 'disabled'

    def quantized(self):
        self.validate()
        return NativeLook(self.name, np.rint(self.tone_q10), np.rint(self.matrix_planes),
                          np.rint(self.chroma_planes), np.rint(self.grid), self.grid_model)

    def validate(self):
        for a, shape, lo, hi in ((self.tone_q10, (3,), -1024, 1024),
                               (self.matrix_planes, (3, 6), -1024, 1024),
                               (self.chroma_planes, (3, 8), -1024, 1024),
                               (self.grid, (405, 3), -64, 63)):
            a=np.asarray(a)
            if a.shape != shape or not np.isfinite(a).all() or np.any((a<lo)|(a>hi)):
                raise ValueError(f'Invalid native values: expected {shape} in [{lo},{hi}]')
        if self.grid_model not in ('disabled', 'rea-bt601-limited-reference-v1'):
            raise ValueError('Unknown grid coordinate model')
        if self.grid_model == 'disabled' and np.any(self.grid):
            raise ValueError('Nonzero grid needs an explicit experimental coordinate model')

def expand_matrix(p):
    a,b,c,d,e,f = np.asarray(p).T
    return np.stack((a+b,-a,-b,-c,c+d,-d,-e,-f,e+f),-1).reshape(-1,3,3).squeeze()

def cp_to_raw_cc1(delta):
    # CP coefficient order from REA look_chroma.forward versus 751K rgb2ycc.
    return np.asarray(delta)[[3,0,1,2,7,4,5,6]]

def tone_curve(values, tone, integer=True):
    if integer:
        y=np.asarray(values,dtype=np.int64); gain,toe,shoulder=np.rint(tone).astype(np.int64)
        x=(y<<12)//int(y[-1]); v=4096-x
        delta=(y*gain)>>10
        delta+=(((((y*toe)>>10)*v)>>12)*v)>>12
        delta+=(((((y*shoulder)>>10)*x)>>12)*v)>>12
        out=np.clip(y+delta,0,0x3ffff)
    else:
        y=np.asarray(values,dtype=float); gain,toe,shoulder=np.asarray(tone)/1024
        x=y/y[-1]; out=np.clip(y*(1+gain+toe*(1-x)**2+shoulder*x*(1-x)),0,0x3ffff)
    if np.any(np.diff(out)<0):
        raise ValueError('Nonmonotonic native Gamma')
    return out
