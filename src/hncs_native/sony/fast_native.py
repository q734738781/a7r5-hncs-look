"""Compiled evaluation of the existing integer colour stages for review only.

Uses true float32 FMA, the original integer Gamma LUTs and original parameter
builders. No fast-math or lower-resolution transfer table is introduced.
"""
import numpy as np
from numba import njit,prange,types
from numba.extending import intrinsic
from llvmlite import ir
from .forward import cached_main_gamma,SY,SL,SC
from .rea_adapter import NativeLook,expand_matrix,tone_curve,cp_to_raw_cc1
from .fast_grid import kernel as grid_kernel
from hncs_native.grid import coordinates

@intrinsic
def fma(typingctx,a,b,c):
    sig=types.float32(types.float32,types.float32,types.float32)
    def codegen(context,builder,signature,args):
        fn=builder.module.declare_intrinsic('llvm.fma',[ir.FloatType()],ir.FunctionType(ir.FloatType(),[ir.FloatType()]*3))
        return builder.call(fn,args)
    return sig,codegen

@njit(cache=True,inline='always')
def i16(v):
    v=v&65535
    return v-65536 if v>=32768 else v

@njit(cache=True,inline='always')
def truncdiv(v,d):return -(abs(v)//d) if v<0 else v//d

@njit(cache=True,parallel=True)
def colour_kernel(x,base,h,table,gamma,thresholds,quadratic,base_only,luma,shears,gains,cs,ylut,yoff,ygain):
    n=len(x);out=np.empty((n,3));F=np.float32
    for j in prange(n):
        hh=int(h[j]);hh=hh if 0<=hh<=1022 else 1023
        xx=x[j];enc=np.empty(3,np.int64)
        for row in range(3):
            if base_only:v=base[j,row]
            else:
                k=row*3;lin=fma(table[hh,k+2],xx[2],fma(table[hh,k+1],xx[1],F(table[hh,k]*xx[0])))
                if quadratic:
                    k+=9;q=fma(table[hh,k+2],F(xx[2]*xx[2]),fma(table[hh,k+1],F(xx[1]*xx[1]),F(table[hh,k]*F(xx[0]*xx[0]))))
                    q=F(q*F(6.1035156e-05));thr=thresholds[row];s=F(q+lin)
                    if q<0:v=lin
                    elif q>=thr:v=F(s-thr)
                    elif q<=-thr:v=F(s+thr)
                    else:v=lin
                else:v=lin
            v=F(min(max(v,F(0)),F(16383)));idx=int(v);frac=F(v-F(np.floor(v)))
            lo=F(gamma[32768+idx]);hi=F(gamma[32768+idx+1])
            enc[row]=i16(int(fma(F(hi-lo),frac,lo)))
        rr,gg,bb=enc[0],enc[1],enc[2]
        yy=(((rr*luma[0]+gg*luma[1]+bb*luma[2])&0xffffffff)>>13)&65535
        rg=F(rr-gg);bg=F(bb-gg)
        cb=fma(shears[3] if rg<0 else shears[1],rg,bg)
        cr=fma(shears[2] if bg<0 else shears[0],bg,rg)
        cr=F(F(cr*F(.5))*(gains[3] if cr<0 else gains[1]));cb=F(F(cb*F(.5))*(gains[2] if cb<0 else gains[0]))
        cr=F(min(max(cr,F(-8192)),F(8191)));cb=F(min(max(cb,F(-8192)),F(8191)))
        cr0=int(F(cr+F(32768)))-32768;cb0=int(F(cb+F(32768)))-32768
        y=i16(yy);lo=np.int32(cs[0]-y);hi=np.int32(y-cs[1]);m=np.int32(lo*cs[2] if lo>0 else hi*cs[3])
        factor=min(max(255-(m>>12),0),255) if lo>0 or hi>0 else 255
        cr1=truncdiv(np.int32(cr0*factor),256);cb1=truncdiv(np.int32(cb0*factor),256)
        y=max(y,0);v=F(ylut[32768+y]&65535)
        gy=fma(F(v-yoff),ygain,yoff);y=int(min(max(gy,F(0)),F(16383)))
        out[j,0]=min(max(truncdiv(cr1*14020+y*10000,10000),0),16383)/16383.
        out[j,1]=min(max(truncdiv(cr1*-7141+y*10000+cb1*-3441,10000),0),16383)/16383.
        out[j,2]=min(max(truncdiv(cb1*17720+y*10000,10000),0),16383)/16383.
    return out

def prepare_fast(renderer,x):
    """Share SSCS/hue only; evaluate each LM16 once inside the scalar kernel."""
    x=np.asarray(x)
    if x.dtype!=np.uint16 or x.shape[-1]!=3:raise ValueError('Physical uint16 required')
    r=renderer
    if r.sscs is not None:x=np.stack(SC.sscs_int(*(x[...,c] for c in range(3)),r.sscs),-1)
    x=x.astype(np.float32);planes=[x[...,c] for c in range(3)];ctl=r.decoded['writes_by_destination']
    h=SL.hue_index(planes,planes,r.atan,ctl['0x9199c']['value'])
    thresholds=tuple(ctl[k]['value'] for k in ('0x91bac','0x91bb0','0x91bb4'))
    return x,np.empty((0,3),np.float32),h,thresholds

def render_fast(renderer,x,look=None,*,prepared=None):
    look=(look or NativeLook()).quantized();look.validate();r=renderer
    xx,base,h,thresholds=prepare_fast(r,x) if prepared is None else prepared
    matrix=expand_matrix(r.wb_weights@look.matrix_planes)/1024;table=r.table.copy();table[:,:9]+=matrix.reshape(1,9).astype(np.float32)
    gamma=r.base_gamma if not np.any(look.tone_q10) else cached_main_gamma(tuple(r.gamma_x),tuple(tone_curve(r.gamma_y,look.tone_q10,True)))
    p=np.rint(r.cc1.astype(float)+cp_to_raw_cc1(r.wb_weights@look.chroma_planes)).astype(int)
    shears=np.array([SY._shear(v)/256 for v in p[:4]],np.float32);gains=np.array([((v&65535)>>3&255)/64 for v in p[4:]],np.float32)
    luma=np.array([SY.i16(int(v)<<7) for v in r.luma],np.int64)
    result=colour_kernel(xx.reshape(-1,3),base.reshape(-1,3),h.ravel(),table,gamma,np.array(thresholds,np.float32),
        bool(r.decoded['writes_by_destination']['0x91bba']['value']),not np.any(matrix) and base.size>0,luma,shears,gains,np.array(r.cs,np.int64),r.ylut,np.float32(SY.i16(r.yoff)),np.float32(r.ygain))
    if look.grid_model!='disabled':result=grid_kernel(result,np.asarray(look.grid,np.float64),coordinates(result)*[4,8,8]+[0,4,4])
    return result.reshape(np.shape(x))
