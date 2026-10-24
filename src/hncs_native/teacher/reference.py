"""Versioned WB-aware colour reference with an explicit physical input contract.

This combines the published A7R V HNCS colour profile with V-Log-Alchemy's
verified display tail. It is not a Phocus RAW export or Lightroom PV emulation.
Never apply it to Sony Edit JPEGs, gamma-encoded values, or averaged RAW.
"""
import numpy as np
from hncs_native.teacher.phocus import HNCSTeacher
from hncs_native.sony.reference import SonySTReference
from hncs_native.sony.forward import ints

TEACHER_ID='a7rv-hncs-public-colour-vlog-tail-physical-v1'
PIPELINE_CONTRACT={
    'id':TEACHER_ID,
    'input':'full-resolution post-GeoCorr uint16 camera RGB; Sony RAW DRO already applied once',
    'pre_colour':'Sony SSCS once, common to native baseline and target',
    'colour':'published HNCS-Lightroom A7R V DCP carrier + original Adobe LookTable + fixed HNCS Color XMP',
    'tone':'V-Log-Alchemy published Reinhard shoulder knee 0.5, Standard 65536-entry Film Curve, ICC TRC, D50/D65 adaptation, sRGB',
    'white_scale':8192,
    'sampling':'render physical pixels first; downsample display outputs only',
    'fit_domain':'exclude sensor/WB-clipped support and out-of-domain target values; keep separate neutral-highlight constraints',
    'not_claimed':['Phocus proprietary export ICC intent','Lightroom PV','complete multi-WB raw Phocus oracle','camera firmware grid equivalence'],
    'validation':'upstream operator/LUT conformance; physical RAW/colour boundary audit',
}

class HNCSReference:
    id=TEACHER_ID
    def __init__(self,main,st):
        self.sony=SonySTReference(main,st);self.colour=HNCSTeacher()
        levels=ints(main['0x7313'])[[0,1,3]];self.neutral=levels[1]/levels

    def render(self,camera_rgb):
        return self.colour.render_reference_tail(self.sony.prepare(camera_rgb),neutral=self.neutral)

    def fitting_target(self,camera_rgb,raw_colour_valid):
        x=np.asarray(camera_rgb);valid=np.asarray(raw_colour_valid)
        if valid.dtype!=np.bool_ or valid.shape!=x.shape[:-1]:
            raise ValueError('A physical RAW clipping-support validity mask is required')
        y=self.render(x)
        # Explicit robust domain, fixed before parameter fitting. Sensor-clipped
        # colours are not recovered simply because a forward operator returns RGB.
        keep=valid&(y.max(-1)>.035)&(y.max(-1)<.985)&(np.ptp(y,axis=-1)<.82)
        return y,keep
