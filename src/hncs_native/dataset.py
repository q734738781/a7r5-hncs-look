"""Prepare private physical-pixel samples and read an explicit local manifest.

Original RAWs are never copied; technical tags and numerical samples stay in the
user-selected local output directory, which is ignored by the repository.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .io import dump,sha

def teacher_signature():
    directory=Path(__file__).parent/'teacher'
    return hashlib.sha256(b''.join((directory/name).read_bytes() for name in ('reference.py','phocus.py','vlog_reference.py'))).hexdigest()

def validate_raw(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        tags=tf.pages[0].tags
        model=str(tags[272].value).strip('\x00 ') if 272 in tags else ''
        orientation=int(tags[274].value) if 274 in tags else 1
    if model!='ILCE-7RM5':raise ValueError('This pipeline requires an ILCE-7RM5 RAW')
    return orientation

def prepare(raw,output,*,seed=241024,samples_per_patch=64):
    from .sony.reference import ReferencePatchSource
    from .teacher.reference import HNCSReference,TEACHER_ID
    from .sony.physical import BASELINE_ID
    raw=Path(raw);output=Path(output)
    data_path=output.with_suffix('.npz');meta_path=output.with_suffix('.json')
    if data_path.exists() or meta_path.exists():raise FileExistsError('Use a new sample output prefix')
    validate_raw(raw)
    if not 1<=samples_per_patch<=4096:raise ValueError('samples_per_patch must be 1..4096')
    source=ReferencePatchSource(raw,dro=True,nr=True)
    teacher=HNCSReference(source.main,source.st)
    x0,y0,x1,y1=source.rect;w,h=x1-x0,y1-y0
    if min(w,h)<768:raise ValueError('RAW crop too small for 12x8 physical patches')
    rng=np.random.default_rng(seed)
    arrays={k:[] for k in ('rgb','target','valid','raw_valid','patch')}
    for py in range(8):
        for px in range(12):
            xa=int((px+.5)*w/12)-32;ya=int((py+.5)*h/8)-32
            x,raw_valid=source.read((xa,ya,xa+64,ya+64),with_validity=True)
            if samples_per_patch==64:
                yy,xx=np.mgrid[0:8,0:8]
                yy=yy*8+rng.integers(0,8,(8,8));xx=xx*8+rng.integers(0,8,(8,8))
                ids=(yy*64+xx).ravel()
            else:ids=rng.choice(4096,samples_per_patch,replace=False)
            x=x.reshape(-1,3)[ids];v=raw_valid.ravel()[ids]
            y,keep=teacher.fitting_target(x,v)
            for key,value in zip(arrays,(x,y,keep,v,np.full(len(x),py*12+px))):arrays[key].append(value)
        print(f'physical patch rows {py+1}/8',flush=True)
    data={k:np.concatenate(v) for k,v in arrays.items()}
    ids=np.flatnonzero(~data['raw_valid'])[:768]
    data['stress_rgb']=data['rgb'][ids];data['stress_valid']=data['raw_valid'][ids]
    data_path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(data_path,**data)
    metadata=dict(schema='hncs-physical-samples/1',model='ILCE-7RM5',raw_sha256=sha(raw),
        main=source.main,st=source.st,baseline=BASELINE_ID,teacher=TEACHER_ID,teacher_hash=teacher_signature(),
        nr=True,dro_applications=1,sampling='12x8 physical patches, stratified pixels; no RAW averaging',
        seed=seed,samples_per_patch=samples_per_patch,n=len(data['rgb']),valid=int(data['valid'].sum()))
    dump(meta_path,metadata)
    return {'samples':data_path.name,'metadata':meta_path.name,'n':len(data['rgb']),'valid':metadata['valid']}

def load_manifest(path):
    from .teacher.reference import TEACHER_ID
    from .sony.physical import BASELINE_ID
    path=Path(path);doc=json.loads(path.read_text(encoding='utf-8'))
    if doc.get('schema')!='hncs-local-fit/1' or not doc.get('frames'):raise ValueError('Expected a nonempty hncs-local-fit/1 manifest')
    rows=[];ids=set()
    for item in doc['frames']:
        name=item['id']
        if name in ids:raise ValueError('Duplicate frame ID')
        ids.add(name)
        if item['split'] not in ('construction','validation'):raise ValueError('Invalid split')
        sp=Path(item['samples']);mp=Path(item['metadata'])
        if sp.is_absolute() or mp.is_absolute():raise ValueError('Use manifest-relative local data paths')
        metadata=json.loads((path.parent/mp).read_text(encoding='utf-8'))
        if (metadata.get('teacher'),metadata.get('baseline'),metadata.get('teacher_hash'))!=(TEACHER_ID,BASELINE_ID,teacher_signature()):
            raise ValueError('Sample teacher/baseline signature mismatch; prepare with the current sources')
        if metadata.get('dro_applications')!=1 or metadata.get('nr') is not True:raise ValueError('Expected RawNR and one native DRO pass')
        with np.load(path.parent/sp,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        n=len(data['rgb'])
        if data['rgb'].dtype!=np.uint16 or data['rgb'].shape!=(n,3) or n==0:raise ValueError('Physical RGB must be nonempty Nx3 uint16')
        if data['target'].shape!=(n,3) or not np.isfinite(data['target']).all() or np.any((data['target']<0)|(data['target']>1)):raise ValueError('Invalid target RGB')
        for field in ('valid','raw_valid'):
            if data[field].shape!=(n,) or data[field].dtype!=np.bool_:raise ValueError('Invalid validity mask')
        if data['patch'].shape!=(n,) or np.any((data['patch']<0)|(data['patch']>=96)):raise ValueError('Invalid spatial patch IDs')
        rows.append(dict(id=name,split=item['split'],metadata=metadata,data=data))
    if not any(row['split']=='construction' for row in rows):raise ValueError('At least one construction frame is required')
    return doc,rows
