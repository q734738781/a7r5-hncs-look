"""Acquire pinned public sources only when explicitly requested; no camera IO."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def git(*args,cwd=None):
    result=subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True,text=True)
    return result.stdout.strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true',help='Explicitly clone the three pinned public repositories')
    args=parser.parse_args()
    from hncs_native.bootstrap import VENDOR,check_sources
    lock=json.loads((ROOT/'upstream.lock.json').read_text(encoding='utf-8'))
    if args.fetch:
        VENDOR.mkdir(exist_ok=True,parents=True)
        for item in lock['upstreams']:
            target=VENDOR/item['directory']
            if not target.exists():
                git('clone','--no-checkout',item['url'],str(target))
                git('checkout','--detach',item['commit'],cwd=target)
            elif not (target/'.git').exists():
                raise RuntimeError('Existing non-Git source directory: '+item['directory'])
            current=git('rev-parse','HEAD',cwd=target)
            if current!=item['commit']:
                raise RuntimeError('Existing checkout is at another commit; use a separate HNCS_UPSTREAM_DIR. No reset performed.')
    rows=check_sources()
    for row in rows:print(json.dumps(row,ensure_ascii=False))
    if not all(row['ok'] for row in rows):raise SystemExit('Pinned sources incomplete or modified.')

if __name__=='__main__':main()
