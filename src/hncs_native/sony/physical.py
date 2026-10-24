"""Native Look operator on audited physical pixels, with no premature clipping."""
import numpy as np
from .forward import SonyNativeLookRenderer,SC

BASELINE_ID='sony-edit-a7rv-physical-u16-v2'

class SonyPhysicalLookRenderer(SonyNativeLookRenderer):
    baseline_id=BASELINE_ID

    def prepare(self,camera_rgb):
        x=np.asarray(camera_rgb)
        if x.dtype!=np.uint16 or x.shape[-1]!=3:
            raise ValueError('Physical post-GeoCorr uint16 RGB required')
        if self.sscs is not None:
            x=np.stack(SC.sscs_int(*(x[...,c] for c in range(3)),self.sscs),-1)
        return self.prepare_after_sscs(x.astype(np.float32))
