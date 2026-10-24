"""Check the published Look and comparison files. Standard library only."""
from pathlib import Path
import base64,hashlib,json,struct

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='78ac328c0e9a073b35080200fd41d3feec1ec55942931a6b6bd56270c6bcf787'
def sha(data):return hashlib.sha256(data).hexdigest()
def jpeg_metadata(data):
    assert data[:2]==b'\xff\xd8'
    offset=2;exif=False
    while offset<len(data):
        assert data[offset]==255
        while data[offset]==255:offset+=1
        marker=data[offset];offset+=1
        if marker in (0xda,0xd9):break
        if marker in range(0xd0,0xd8) or marker==1:continue
        length=struct.unpack('>H',data[offset:offset+2])[0]
        payload=data[offset+2:offset+length]
        if marker==0xe1 and payload.startswith(b'Exif\0\0'):exif=True
        offset+=length
    return exif
def main():
    look=(ROOT/'looks/HNCS_Unified_Only.a7look').read_bytes()
    assert sha(look)==EXPECTED
    assert look==(ROOT/'docs/downloads/HNCS_Unified_Only.a7look').read_bytes()
    d=json.loads(look);assert d['hncs_native']['calibration_model']=='ILCE-7RM5'
    assert d['hncs_native']['camera_verified'] is False
    assert d['hncs_native']['extra_dro_layer'] is False
    assert d['hncs_native']['grid_model']!='disabled'
    grid=base64.b64decode(d['table_base64'],validate=True)
    assert len(grid)==1215 and all(-64<=x<=63 for x in struct.unpack('1215b',grid))
    spec=d['native_photo']['runtime_basis']
    assert len(spec['tone_q10'])==3
    assert len(spec['matrix_planes'])==3 and all(len(r)==6 for r in spec['matrix_planes'])
    assert len(spec['chroma_planes'])==3 and all(len(r)==8 for r in spec['chroma_planes'])
    manifest=json.loads((ROOT/'docs/comparison.json').read_text(encoding='utf-8'))
    assert manifest['look']['sha256']==EXPECTED and len(manifest['photos'])==9
    assert len({p['id'] for p in manifest['photos']})==9
    total=0
    for p in manifest['photos']:
        assert set(p['images'])=={'st','adobe','a7look'}
        for v in p['images'].values():
            target=(ROOT/'docs'/v['path']).resolve();assert target.is_relative_to(ROOT/'docs')
            data=target.read_bytes();assert sha(data)==v['sha256'] and len(data)==v['bytes']
            assert not jpeg_metadata(data)
            assert v['width']==792 and v['height']==528
            total+=1
    for file in ('docs/index.html','docs/assets/app.js','docs/assets/style.css','README.md','LICENSE','ASSETS.md'):
        text=(ROOT/file).read_text(encoding='utf-8')
        assert not any(s in text for s in ('F:\\','C:\\Users\\','Z:\\','PRIVATE KEY'))
    assert not list(ROOT.rglob('*.ARW')) and not list(ROOT.rglob('*.DAT'))
    print(f'PASS: unchanged Look; full grid; {total} previews; no EXIF; local assets verified.')
if __name__=='__main__':main()
