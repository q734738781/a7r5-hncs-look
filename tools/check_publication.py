"""Audit Git's tracked/staged publication set; report only, never delete files."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.arw','.dng','.nef','.cr2','.cr3','.raf','.rw2','.npy','.npz','.dat','.dcp','.xmp','.cube','.bin','.dll','.exe','.pdb','.pem','.key','.p12','.pfx'}
PERSONAL_PATH=re.compile(r'(?i)(?:\b[a-z]:[\\/](?:Users|A7R5|RAW)[\\/]|/(?:Users|home)/[^/\s]+/)')
SECRET=re.compile(r'(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})')

def check():
    result=subprocess.run(['git','ls-files','-z','--cached'],cwd=ROOT,capture_output=True,check=True)
    paths=[p for p in result.stdout.decode('utf-8').split('\0') if p]
    errors=[]
    for rel in paths:
        path=ROOT/rel
        if path.suffix.lower() in FORBIDDEN:errors.append(rel+': forbidden source asset/binary/cache')
        if any(part in {'.external','.venv','__pycache__','runs','local'} for part in Path(rel).parts):errors.append(rel+': private/dependency directory')
        if path.suffix.lower() in {'.py','.md','.json','.txt','.toml','.js','.css','.html','.a7look'} or path.name in {'.gitignore','.gitattributes','LICENSE'}:
            content=path.read_text(encoding='utf-8')
            if PERSONAL_PATH.search(content):errors.append(rel+': personal absolute path')
            if SECRET.search(content):errors.append(rel+': secret marker')
    return {'pass':not errors,'tracked_files':len(paths),'errors':errors,'scope':'Git publication allow-set content and file types; no image-content or legal guarantee'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    report=check();print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['pass']:raise SystemExit(1)
