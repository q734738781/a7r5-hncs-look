"""Explicit local inputs for inspection, sampling, rendering, fitting and checks."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from .bootstrap import ROOT,check_sources
from .format import read_look,pack_native
from .io import dump,sha

def require_sources():
    rows=check_sources()
    if not all(row['ok'] for row in rows):
        raise RuntimeError('Pinned sources are missing/changed. Run python tools/fetch_upstream.py --fetch, or check HNCS_UPSTREAM_DIR.')

def render_raw(args):
    from PIL import Image,ImageOps
    from .dataset import validate_raw
    from .sony.reference import ReferencePatchSource
    from .sony.physical import SonyPhysicalLookRenderer,BASELINE_ID
    from .runtime import render_fast,teacher_renderer
    from .teacher.reference import TEACHER_ID
    if args.output.exists():raise FileExistsError('Use a new output image')
    if args.scale<1:raise ValueError('scale must be positive')
    if args.output.suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff'):raise ValueError('Choose JPG, PNG or TIFF')
    orientation=validate_raw(args.raw);started=time.perf_counter()
    source=ReferencePatchSource(args.raw,dro=True,nr=True)
    cx,cy,x1,y1=source.rect;w,h=x1-cx,y1-cy
    xa,ya,xb,yb=args.crop or (0,0,w,h)
    if not 0<=xa<xb<=w or not 0<=ya<yb<=h:raise ValueError('Crop outside physical output coordinates')
    ow,oh=(xb-xa)//args.scale,(yb-ya)//args.scale
    if min(ow,oh)<1:raise ValueError('Scale exceeds crop')
    xb=xa+ow*args.scale;yb=ya+oh*args.scale
    out=np.empty((oh,ow,3),np.float32);step=args.scale*max(1,240//args.scale)
    look=None if args.mode!='look' else read_look(args.look)
    renderer=teacher_renderer(source.main,source.st) if args.mode=='teacher' else SonyPhysicalLookRenderer(source.main,source.st)
    for y in range(ya,yb,step):
        end=min(y+step,yb);physical=source.read((xa,y,xb,end))
        rgb=renderer.render(physical) if args.mode=='teacher' else render_fast(renderer,physical,look)
        out[(y-ya)//args.scale:(end-ya)//args.scale]=rgb.reshape((end-y)//args.scale,args.scale,ow,args.scale,3).mean((1,3))
        print(f'colour rows {end-ya}/{yb-ya}',flush=True)
    image=Image.fromarray(np.rint(np.clip(out,0,1)*255).astype('uint8'))
    image.getexif()[274]=orientation;image=ImageOps.exif_transpose(image)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    options={'quality':95} if args.output.suffix.lower() in ('.jpg','.jpeg') else {}
    image.save(args.output,**options)
    receipt=dict(raw_sha256=sha(args.raw),baseline=BASELINE_ID,mode=args.mode,
        teacher=TEACHER_ID if args.mode=='teacher' else None,look_sha256=sha(args.look) if look else None,
        rawnr=True,dro_applications=1,extra_DRO=False,crop=[xa,ya,xb,yb],
        scale_after_colour=args.scale,size=list(image.size),output=args.output.name,seconds=time.perf_counter()-started)
    dump(args.output.with_suffix('.json'),receipt);print(json.dumps(receipt))

def evaluate(args):
    from .dataset import load_manifest
    from .sony.physical import SonyPhysicalLookRenderer
    from .runtime import render_fast
    from .optimizer.fit import metrics
    _,frames=load_manifest(args.dataset);look=read_look(args.look);rows=[]
    for row in frames:
        m,z=row['metadata'],row['data'];r=SonyPhysicalLookRenderer(m['main'],m['st'])
        pred=render_fast(r,z['rgb'],look);v=z['valid']
        rows.append({'id':row['id'],'split':row['split'],'all_samples':metrics(pred,z['target']),
                     'valid_samples':metrics(pred[v],z['target'][v]) if v.any() else None})
    report={'reference':'published composite teacher, not actual Adobe output','validation_is_blind':False,'photos':rows}
    if args.output:
        if args.output.exists():raise FileExistsError('Use a new evaluation receipt')
        args.output.parent.mkdir(exist_ok=True,parents=True);dump(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='fit':
        if '--help' not in argv and '-h' not in argv:require_sources()
        from .solver import main as fit
        return fit(argv[1:])
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('fit',help='Joint analytic fitting and exact integer polishing; use fit --help')
    sub.add_parser('doctor',help='Check pinned numerical sources and optional acceleration')
    p=sub.add_parser('inspect',help='Inspect a Look without upstream assets');p.add_argument('--look',type=Path,default=ROOT/'looks/HNCS_Unified_Only.a7look')
    p=sub.add_parser('prepare',help='Sample physical RAW pixels for a private dataset');p.add_argument('--raw',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seed',type=int,default=241024);p.add_argument('--samples-per-patch',type=int,default=64)
    p=sub.add_parser('render',help='Render own RAW; reduce RGB only after colour');p.add_argument('--raw',required=True,type=Path);p.add_argument('--mode',choices=('look','teacher','sony'),default='look');p.add_argument('--look',type=Path,default=ROOT/'looks/HNCS_Unified_Only.a7look');p.add_argument('--output',required=True,type=Path);p.add_argument('--scale',type=int,default=12);p.add_argument('--crop',nargs=4,type=int,metavar=('X0','Y0','X1','Y1'))
    p=sub.add_parser('evaluate',help='Report sample errors against the declared composite teacher');p.add_argument('--dataset',required=True,type=Path);p.add_argument('--look',required=True,type=Path);p.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.command=='inspect':
        look=read_look(args.look)
        print(json.dumps({'name':look.name,'sha256':sha(args.look),'native_bytes':len(pack_native(look)),'grid_bytes':1215,
            'grid_model':look.grid_model,'grid_nonzero_nodes':int(np.count_nonzero(np.any(look.grid,axis=1))),
            'calibration_model':'ILCE-7RM5','camera_verified':False},ensure_ascii=False,indent=2));return
    if args.command=='doctor':
        rows=check_sources()
        print(json.dumps({'sources':rows,'Python':sys.version.split()[0]},indent=2))
        if not all(row['ok'] for row in rows):raise SystemExit(1)
        from .runtime import teacher_renderer,render_fast
        print('Numerical runtime imports OK.');return
    require_sources()
    if args.command=='prepare':
        from .dataset import prepare
        print(json.dumps(prepare(args.raw,args.output,seed=args.seed,samples_per_patch=args.samples_per_patch)));return
    if args.command=='render':return render_raw(args)
    if args.command=='evaluate':return evaluate(args)

if __name__=='__main__':main()
