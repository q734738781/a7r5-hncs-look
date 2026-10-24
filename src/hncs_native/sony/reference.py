"""Audited colour/spatial reference; independent of the fitting surrogate.

Uses the published Sony stage functions. atan is the analytic approximation
already used by this project; no assertion of original-binary bit identity.
RawNR, sharpening and Marble are excluded from this colour audit explicitly.
"""
from pathlib import Path
import time
import numpy as np
from .forward import SonyNativeLookRenderer,SL,SY,SS,SC,AL,SF,ints

class SonySTReference:
    def __init__(self,main,st):
        self.calibration=SonyNativeLookRenderer(main,st)
        # ILCE-7RM5 samples have ARW 5.x: format 0x3500, has LM16, look id 2.
        self.area=SS.area_comp_params(0x3500,True,2)

    def prepare(self,rgb):
        x=np.asarray(rgb)
        if x.dtype!=np.uint16 or x.shape[-1]!=3:
            raise ValueError('Post-GeoCorr uint16 camera RGB required; do not rescale or preclip DRO output')
        return np.stack(SC.sscs_int(*(x[...,c] for c in range(3)),self.calibration.sscs),-1)

    def render(self,rgb,*,spatial=True,area=True,sscs=True):
        r=self.calibration;x=self.prepare(rgb) if sscs else np.asarray(rgb)
        planes=[np.asarray(x[...,c],np.float32) for c in range(3)]
        c=r.decoded['writes_by_destination'];v=lambda k:c[k]['value']
        ctl=dict(blur_luma=v('0x91998') if spatial else 0,
                 blur_chroma=v('0x9199a') if spatial else 0,scale=v('0x9199c'),
                 quadratic=v('0x91bba'),thresholds=tuple(v(k) for k in ('0x91bac','0x91bb0','0x91bb4')))
        enc=SL.render_look(planes,r.table,r.base_gamma,r.atan,ctl)
        y,cr,cb=SY.rgb2ycc(*enc,r.cc1,r.luma,clamp=True)
        if area:cr,cb=SS.area_comp_apply(cr,cb,self.area)
        y,cr,cb=SS.chroma_suppress_apply(y,cr,cb,r.cs)
        y=SS.ygamma_apply(y,r.ylut,r.yoff,r.ygain)
        return np.clip(np.stack(SY.ycc2rgb(y,cr,cb),-1)/16383,0,1)

def reference_frontend(path,cache_path,*,dro=True,progress=print):
    """Full-resolution upstream ITP -> optional DRO -> GeoCorr, stored as u16.

    Cache must be an explicit new path. Caller owns manifest/hash validation.
    Downsampling belongs AFTER rendering, never in this function.
    """
    import rawpy,sony_itp as ITP,sony_dro as DRO,sony_lens as LN,decoder_params as DP
    path=Path(path);cache_path=Path(cache_path);start=time.perf_counter()
    main=AL.main_tags(str(path));st=AL.look_ifds(str(path))['Standard']
    black=ints(main['0x7310']);gains=SF.wb_gains(ints(main['0x7313']),int(main['0x7809']))
    x0,y0,x1,y1=map(int,DP.crop_rect(path));rect=(x0,y0,x1,y1)
    with rawpy.imread(str(path)) as f:
        if f.raw_pattern.tolist()!=[[0,1],[3,2]]:raise ValueError('RGGB required')
        raw=f.raw_image_visible.copy()
    hr,wr=raw.shape;ex0,ey0=max(x0-32,0),max(y0-32,0);ex1,ey1=min(x1+32,wr),min(y1+32,hr)
    ox,oy=ex0-x0,ey0-y0;raw_crop=raw[y0:y1,x0:x1]
    G=None
    if dro:
        G=DRO.build_grid(raw_crop,ints(main['0x781b'])[:10].astype(np.float32)/1024,
                        ints(main['0x781c'])[:28].astype(np.float32)/1024,black=tuple(black[[0,1,3]]))
    raw=np.ascontiguousarray(raw[ey0:ey1,ex0:ex1]);h,w=raw.shape
    # Same tile and border rules as upstream itp_full_image, with progress only.
    halo=12;pad=raw[np.ix_(ITP._edge_indices(h,halo),ITP._edge_indices(w,halo))]
    planes=[np.empty((h,w),np.uint16) for _ in range(3)]
    for ya in range(0,h,1024):
        yb=min(ya+1024,h)
        for xa in range(0,w,1024):
            xb=min(xa+1024,w)
            p,_=ITP.itp_process(pad[ya:yb+2*halo,xa:xb+2*halo],black,gains,int(main['0x7311']))
            for k in range(3):planes[k][ya:yb,xa:xb]=p[k][halo:halo+yb-ya,halo:halo+xb-xa]
        progress(f'ITP rows {yb}/{h}; {time.perf_counter()-start:.1f}s',flush=True)
    del pad,raw
    if dro:
        for ya in range(0,h,128):
            yb=min(ya+128,h);X,Y=np.meshgrid(np.arange(ox,ox+w),np.arange(oy+ya,oy+yb))
            p=DRO.vatr([q[ya:yb] for q in planes],X,Y,G)
            for k in range(3):planes[k][ya:yb]=p[k]
        progress(f'DRO complete; {time.perf_counter()-start:.1f}s',flush=True)
    L=LN.lens_from_arw(path,rect=rect,dist_setting=1,ca_setting=1)
    cache_path.parent.mkdir(exist_ok=True,parents=True)
    out=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.uint16,shape=(y1-y0,x1-x0,3))
    for ya in range(0,y1-y0,96):
        yb=min(ya+96,y1-y0)
        p=LN.geocorr_u16(planes,(ox,oy),(0,ya,x1-x0,yb),L)
        out[ya:yb]=np.stack(p,-1)
    out.flush()
    progress(f'GeoCorr complete; {time.perf_counter()-start:.1f}s',flush=True)
    return out,main,st,{'pipeline':'upstream tiled ITP -> DRO once on RAW -> GeoCorr -> uint16',
        'dro':dro,'shape':list(out.shape),'seconds':time.perf_counter()-start,
        'excluded':['RawNR','Sharpness','Spica','Marble'],'crop_rect':rect}

class ReferencePatchSource:
    """Full-resolution patches with whole-frame DRO statistics and lens geometry.

    Halo is real neighbouring RAW, not mean/resize samples. Equivalence to the
    full-frame reference is a separately checked regression requirement.
    """
    def __init__(self,path,*,dro=True,nr=False):
        import rawpy,sony_dro as DRO,sony_lens as LN,decoder_params as DP
        self.path=Path(path);self.main=AL.main_tags(str(path));self.st=AL.look_ifds(str(path))['Standard']
        self.black=ints(self.main['0x7310']);self.gains=SF.wb_gains(ints(self.main['0x7313']),int(self.main['0x7809']))
        self.rect=tuple(map(int,DP.crop_rect(path)));x0,y0,x1,y1=self.rect
        with rawpy.imread(str(path)) as f:
            if f.raw_pattern.tolist()!=[[0,1],[3,2]]:raise ValueError('RGGB required')
            self.raw=f.raw_image_visible.copy()
            self.white_level=int(f.white_level)
        self.lens=LN.lens_from_arw(path,rect=self.rect,dist_setting=1,ca_setting=1)
        self.grid=None
        self.nr_look=None
        if nr:
            import sony_rawnr as NR,struct
            tags={int(k,16):(None,ints(v).tolist(),None) for k,v in self.main.items() if k.startswith('0x78c') or k.startswith('0x78d')}
            if 0x78c0 not in tags:raise ValueError('Missing RAW noise calibration')
            self.nr_look=NR.look_from_sr2(tags)
            buf=self.path.read_bytes();ifd=DP._ifd(buf,struct.unpack_from('<I',buf,4)[0]);exif=DP._ifd(buf,DP._get(buf,ifd,0x8769))
            self.iso=DP._get(buf,exif,0x8827,0)
            if not self.iso or self.iso==65535:raise ValueError('Unsupported ISO metadata for RawNR audit')
        if dro:
            self.grid=DRO.build_grid(self.raw[y0:y1,x0:x1],ints(self.main['0x781b'])[:10].astype(np.float32)/1024,
                      ints(self.main['0x781c'])[:28].astype(np.float32)/1024,black=tuple(self.black[[0,1,3]]))

    def read(self,region,*,with_validity=False):
        import sony_itp as ITP,sony_dro as DRO,sony_lens as LN
        xa,ya,xb,yb=map(int,region)
        X,Y=np.meshgrid(np.arange(xa,xb,dtype=np.float32),np.arange(ya,yb,dtype=np.float32))
        coords=[LN.src_coords(X,Y,self.lens,k) for k in range(3)]
        cx,cy,_,_=self.rect;hr,wr=self.raw.shape
        x0=max(0,(int(np.floor(min(p[0].min() for p in coords)))+cx-32)//2*2)
        y0=max(0,(int(np.floor(min(p[1].min() for p in coords)))+cy-32)//2*2)
        x1=min(wr,(int(np.ceil(max(p[0].max() for p in coords)))+cx+35)//2*2)
        y1=min(hr,(int(np.ceil(max(p[1].max() for p in coords)))+cy+35)//2*2)
        raw=np.ascontiguousarray(self.raw[y0:y1,x0:x1]);ox,oy=x0-cx,y0-cy
        valid=None
        if with_validity:
            from scipy.ndimage import maximum_filter
            # Both sensor saturation and WB working-range clipping lose colour.
            # Propagate conservatively through RawNR (10), ITP (12) and GeoCorr
            # support. This mask controls fitting, not a display desaturator.
            scaled=SF.itp_scale(raw,self.black,self.gains,int(self.main['0x7311']))
            clipped=(raw>=self.white_level-2)|(scaled>=16382)
            influence=maximum_filter(clipped.astype(np.uint8),size=49,mode='nearest')
            valid=np.ones(X.shape,bool)
            for sx,sy in coords:
                ix=np.floor(sx-ox).astype(int);iy=np.floor(sy-oy).astype(int)
                for dy in (0,1):
                    for dx in (0,1):valid &= influence[np.clip(iy+dy,0,len(influence)-1),np.clip(ix+dx,0,influence.shape[1]-1)]==0
        if self.nr_look is not None:
            import sony_rawnr as NR
            raw=NR.rawnr(raw,self.nr_look,iso=self.iso)
        planes=ITP.itp_full_image(raw,self.black,self.gains,int(self.main['0x7311']),rust=False)
        if self.grid is not None:
            xx,yy=np.meshgrid(np.arange(ox,ox+raw.shape[1]),np.arange(oy,oy+raw.shape[0]))
            planes=DRO.vatr(planes,xx,yy,self.grid)
        result=np.stack(LN.geocorr_u16(planes,(ox,oy),(xa,ya,xb,yb),self.lens),-1)
        return (result,valid) if with_validity else result
