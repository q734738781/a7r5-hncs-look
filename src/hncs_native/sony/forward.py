"""A7R V RAW-calibrated Sony look path, using the 751K stage implementations.

camera_rgb is white-balanced camera RGB on Sony's 14-bit working scale.
Output is display encoded sRGB in [0,1]. Spatial NR/DRO/lens/sharpening are
excluded. Exact here refers to the implemented software arithmetic, not hardware.
"""
from pathlib import Path
from functools import lru_cache
import base64, hashlib, json
import numpy as np
from hncs_native.bootstrap import ROOT, portable_import
import arw_looks as AL
import sony_look as SL
import sony_ycc as SY
import sony_frontend as SF
import decode_780f as D
import sony_sscs as SC
from .rea_adapter import NativeLook, expand_matrix, cp_to_raw_cc1, tone_curve
from .baseline import BASELINE_ID

BT=portable_import('build_color_table')
SS=portable_import('sony_ycc_stages')

@lru_cache(maxsize=128)
def cached_main_gamma(x_knots,y_knots):
    """Cache exact upstream integer LUTs, shared across identical calibrations."""
    result=SL.build_main_gamma(x_knots,y_knots,len(y_knots))
    result.flags.writeable=False
    return result

def ints(value):
    return np.asarray([int(x) for x in str(value).split()],dtype=np.int64)

class SonyNativeLookRenderer:
    baseline_id = BASELINE_ID

    def __init__(self, main, st, *, sscs=True):
        self.main=main; self.st=st
        tags={int(k,16):ints(v).tolist() for k,v in main.items()
              if k in ('0x7882','0x7883','0x7884','0x7885','0x78f7')}
        self.sscs=SC.params_from_tags(tags) if sscs else None
        if self.sscs is not None and any(self.sscs['T']<=self.sscs[k] for k in ('tR','tG','tB')):
            raise ValueError('Missing or invalid RAW SSCS threshold calibration')
        self.decoded=D.decode_blob(base64.b64decode(st['0x780f'].removeprefix('base64:')))
        self.table=np.asarray(BT.build_table(self.decoded),dtype=np.float32)
        self.gamma_x=ints(st['0x7805']); self.gamma_y=ints(st['0x7806'])
        self.luma=ints(st.get('0x7801','19 38 7'))
        arrays={t:ints(st[t]) for t in ('0x7842','0x7843','0x7844','0x7845','0x7846')}
        arrays.update({t:ints(main[t]) for t in ('0x7841','0x7847','0x7848')})
        self.cc1=np.asarray(SY.select_params(arrays,1))
        # analytic atan table: no unprovided proprietary macOS binary required.
        self.atan=(np.arctan(np.arange(512)/512)*512/np.pi).astype(np.float32)
        self.ylut=SS.build_ygamma([i<<2 for i in range(512)],False,0,0,0,
                                  int(st.get('0x780c',0)),int(st.get('0x780d',0)),0,[0]*512,[0]*512,0,0)[1]
        self.ygain=SS.ygamma_gain_raw(1,ints(st['0x780e']),0)/16384
        self.yoff=SS.ygamma_offset(1,ints(main['0x780b']),0)
        weights=ints(main['0x7848'])[:3]/1024
        # Keep extrapolation if Sony encoded it (some sample sums exceed 1).
        self.wb_weights=weights
        self.base_gamma=cached_main_gamma(tuple(self.gamma_x),tuple(self.gamma_y))
        g=np.concatenate([ints(main[t]) for t in ('0x787e','0x787f','0x7880','0x7881')])
        look=dict(zip((0xf00,0xf02,0xf04,0xf06,0xf08,0xf0a,0xf0c,0xf0e),g),fmt=0)
        self.cs=SS.chroma_suppress_params(look,0,64,SS.inverse_lut_positive(self.ylut),
             SS.inverse_lut_centered(self.base_gamma),self.base_gamma[32768:],self.ylut)

    @classmethod
    def from_arw(cls,path):
        main=AL.main_tags(str(path)); st=AL.look_ifds(str(path))['Standard']
        return cls(main,st)

    def prepare(self,camera_rgb):
        x=np.asarray(camera_rgb,dtype=np.float32)
        if x.shape[-1]!=3 or not np.isfinite(x).all(): raise ValueError('Finite camera RGB required')
        x=np.clip(x,0,16383)
        if self.sscs is not None:
            # RAW/ITP feeds u16 planes. Apply exactly the same upstream integer
            # stage before hue indexing and LM16; precompute for optimization.
            z=np.floor(x).astype(np.uint16)
            x=np.stack(SC.sscs_int(*(z[...,c] for c in range(3)),self.sscs),-1).astype(np.float32)
        return self.prepare_after_sscs(x)

    def prepare_after_sscs(self,x):
        """Shared LM16 preparation; caller owns the single SSCS application."""
        planes=[x[...,c] for c in range(3)]
        controls=self.decoded['writes_by_destination']
        h=SL.hue_index(planes,planes,self.atan,controls['0x9199c']['value'])
        thresholds=tuple(controls[k]['value'] for k in ('0x91bac','0x91bb0','0x91bb4'))
        # Preserve pre-clipping values so fitted coefficients can recover a
        # channel that the unmodified ST matrix would otherwise have clipped.
        table=self.table[np.minimum(np.trunc(h).astype(int),1023)]
        mixed=[]
        for row in range(3):
            b=table[...,row*3:row*3+3]
            lin=SL.fma32(b[...,2],planes[2],SL.fma32(b[...,1],planes[1],
                         (b[...,0]*planes[0]).astype(np.float32)))
            if controls['0x91bba']['value']:
                nn=table[...,9+row*3:12+row*3]
                q=SL.fma32(nn[...,2],planes[2]**2,SL.fma32(nn[...,1],planes[1]**2,
                          (nn[...,0]*planes[0]**2).astype(np.float32)))*np.float32(1/16384)
                thr=np.float32(thresholds[row]); sm=(q+lin).astype(np.float32)
                lin=np.where(q<0,lin,np.where(q>=thr,sm-thr,np.where(q<=-thr,sm+thr,lin)))
            mixed.append(lin)
        return x,np.stack(mixed,-1),h,thresholds

    def _encode(self,x,base,h,thresholds,look,weights,integer):
        matrix=expand_matrix(weights @ look.matrix_planes)/1024
        if integer and np.any(matrix):
            native_table=self.table.copy()
            native_table[:,:9]+=matrix.reshape(1,9).astype(np.float32)
            mixed=np.stack(SL.apply_lm16([x[...,c] for c in range(3)],h,native_table,thresholds,
                 bool(self.decoded['writes_by_destination']['0x91bba']['value'])),-1)
        else:
            mixed=np.clip(base+x @ matrix.T,0,16383)
        gamma_y=tone_curve(self.gamma_y,look.tone_q10,integer)
        # Fast differentiable Gamma for optimization; final evaluator uses Sony LUT.
        if integer:
            gamma=self.base_gamma if not np.any(look.tone_q10) else cached_main_gamma(tuple(self.gamma_x),tuple(gamma_y))
            enc=np.stack([SL.apply_main_gamma(mixed[...,c],gamma) for c in range(3)],-1)
        else:
            enc=np.interp(mixed*128,np.r_[0,self.gamma_x],np.r_[0,gamma_y])/16
        return enc

    def render(self,camera_rgb,native_look=None,wb=None,*,integer=True,prepared=None,cache_encode=False):
        look=native_look or NativeLook(); look.validate()
        if integer: look=look.quantized()
        weights=self.wb_weights if wb is None else np.asarray(wb,dtype=float)
        if weights.shape!=(3,): raise ValueError('wb is three native endpoint weights')
        x,base,h,thresholds=self.prepare(camera_rgb) if prepared is None else prepared
        # Opt-in for immutable collocation batches. Keep strong input references,
        # so recycled object ids cannot produce a false cache hit.
        key=((weights@look.matrix_planes).tobytes(),tuple(look.tone_q10),tuple(thresholds))
        cached=getattr(self,'_encoded_cache',None) if cache_encode and integer else None
        if cached is not None and cached[0] is x and cached[1] is base and cached[2] is h and cached[3]==key:
            enc=cached[4]
        else:
            enc=self._encode(x,base,h,thresholds,look,weights,integer)
            if cache_encode and integer:self._encoded_cache=(x,base,h,key,enc)
        p=self.cc1.astype(float)+cp_to_raw_cc1(weights @ look.chroma_planes)
        if integer:
            y,cr,cb=SY.rgb2ycc(*[enc[...,c] for c in range(3)],np.rint(p).astype(int),self.luma)
            y,cr,cb=SS.chroma_suppress_apply(y,cr,cb,self.cs)
            y=SS.ygamma_apply(y,self.ylut,self.yoff,self.ygain)
            rgb=np.stack(SY.ycc2rgb(y,cr,cb),-1)/16383
        else:
            y=enc @ (self.luma/64)
            rg=enc[...,0]-enc[...,1]; bg=enc[...,2]-enc[...,1]
            cb=bg+np.where(rg<0,p[3],p[1])/1024*rg
            cr=rg+np.where(bg<0,p[2],p[0])/1024*bg
            cr=np.clip(cr*.5*np.where(cr<0,p[7],p[5])/512,-8192,8191)
            cb=np.clip(cb*.5*np.where(cb<0,p[6],p[4])/512,-8192,8191)
            w9,w12,w8,w11=self.cs
            factor=np.where(y<w9,np.clip(255-(w9-y)*w8/4096,0,255),
                   np.where(y>w12,np.clip(255-(y-w12)*w11/4096,0,255),255))/256
            cr*=factor; cb*=factor
            yy=np.interp(y,np.arange(32768),self.ylut[32768:].view(np.uint16))
            yy=np.clip((yy-self.yoff)*self.ygain+self.yoff,0,16383)
            rgb=np.stack([yy+1.402*cr,yy-.7141*cr-.3441*cb,yy+1.772*cb],-1)/16383
        rgb=np.clip(rgb,0,1)
        if look.grid_model!='disabled':
            from hncs_native.grid import predict
            # REA's provisional output-space coordinate convention, clearly labeled.
            shape=rgb.shape
            rgb=predict(rgb.reshape(-1,3),look.grid,np.eye(3)).reshape(shape)
        return rgb
