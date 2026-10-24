"""Read/write the documented REA carrier format; no firmware operations.

REA is the format reference only; this module does not execute upstream patch
builders or import firmware images. The legacy target remains a wire marker.
"""
from pathlib import Path
import base64
import hashlib
import json
import struct
import numpy as np
from .sony.rea_adapter import NativeLook,expand_matrix

def document(look,metadata=None):
    look=look.quantized();look.validate()
    return dict(schema='a7c2-creative-look/1',name=look.name,base='ST',
        target={'model':'ILCE-7CM2','firmware':'2.01'},table_format='native-5x9x9-s7-planar',
        table_base64=base64.b64encode(look.grid.astype('i1').T.copy().tobytes()).decode(),
        native_photo={'runtime_basis':dict(schema='a7c2-photo-q10-endpoint-uniform-matrix/1',
          tone_q10=look.tone_q10.astype(int).tolist(),matrix_planes=look.matrix_planes.astype(int).tolist(),chroma_planes=look.chroma_planes.astype(int).tolist())},
        hncs_native={'calibration_model':'ILCE-7RM5','grid_model':look.grid_model,
          'carrier_target_is_legacy_wire_contract_only':True,'software_candidate':True,
          'camera_verified':False,'extra_dro_layer':False,**(metadata or {})})

def decode(doc):
    if (doc.get('schema'),doc.get('base'),doc.get('table_format'))!=('a7c2-creative-look/1','ST','native-5x9x9-s7-planar'):
        raise ValueError('Unsupported Look carrier')
    if doc.get('target')!={'model':'ILCE-7CM2','firmware':'2.01'}:
        raise ValueError('Unexpected legacy carrier marker')
    if doc.get('hncs_native',{}).get('calibration_model')!='ILCE-7RM5':
        raise ValueError('A7R V calibration declaration required')
    spec=doc['native_photo']['runtime_basis']
    if spec['schema']!='a7c2-photo-q10-endpoint-uniform-matrix/1':raise ValueError('Unsupported native basis')
    for field,shape in [('tone_q10',(3,)),('matrix_planes',(3,6)),('chroma_planes',(3,8))]:
        arr=np.asarray(spec[field],dtype=object);values=arr.ravel().tolist()
        if arr.shape!=shape or any(type(x) is not int or not -1024<=x<=1024 for x in values):
            raise ValueError('Invalid integer native parameters')
    raw=base64.b64decode(doc['table_base64'],validate=True)
    if len(raw)!=1215:raise ValueError('Expected 1215 grid bytes')
    look=NativeLook(doc['name'],np.array(spec['tone_q10']),np.array(spec['matrix_planes']),
        np.array(spec['chroma_planes']),np.frombuffer(raw,'i1').reshape(3,405).T.copy(),doc['hncs_native']['grid_model'])
    look.validate();return look

def read_look(path):return decode(json.loads(Path(path).read_text(encoding='utf-8')))

def pack_native(look):
    q=look.quantized();q.validate()
    return struct.pack('<54h',*np.r_[q.tone_q10,expand_matrix(q.matrix_planes).ravel(),q.chroma_planes.ravel()].astype(int))

def write_look(look,path,metadata=None):
    path=Path(path)
    if path.exists():raise FileExistsError('Refusing to replace an existing Look')
    doc=document(look,metadata);decode(doc)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_bytes':len(pack_native(look)),'grid_bytes':1215}
