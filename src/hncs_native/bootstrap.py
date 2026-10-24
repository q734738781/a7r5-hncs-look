"""Explicit source roots and an in-memory C99 math-library portability adapter.

No commercial binary is loaded by this adapter. The source modules come from
the three repositories pinned in upstream.lock.json, acquired separately.
"""
from pathlib import Path
import ctypes.util
import hashlib
import json
import os
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(os.environ.get('HNCS_UPSTREAM_DIR', ROOT / '.external')).resolve()
for relative in ('sony-a7m5-looks/sonylook', 'sony-a7m5-looks/camera_raw', 'HNCS-Lightroom'):
    directory = str(VENDOR / relative)
    if directory not in sys.path:
        sys.path.insert(0, directory)

def portable_import(name):
    if name in sys.modules:
        return sys.modules[name]
    path = VENDOR / 'sony-a7m5-looks/sonylook' / (name + '.py')
    if not path.is_file():
        raise ImportError('Missing pinned Sony source. Run tools/fetch_upstream.py --fetch, or set HNCS_UPSTREAM_DIR.')
    source = path.read_text(encoding='utf-8')
    if sys.platform != 'darwin':
        library = 'ucrtbase.dll' if os.name == 'nt' else ctypes.util.find_library('m')
        if not library:
            raise RuntimeError('A C99 math runtime with fma is required.')
        source = source.replace('/usr/lib/libSystem.B.dylib', library)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module

def check_sources():
    lock = json.loads((ROOT / 'upstream.lock.json').read_text(encoding='utf-8'))
    rows = []
    for upstream in lock['upstreams']:
        missing, changed = [], []
        for item in upstream['required_files']:
            path = VENDOR / upstream['directory'] / item['path']
            if not path.is_file():
                missing.append(item['path']); continue
            content = path.read_bytes()
            if item['mode'] == 'text-lf':
                content = content.replace(b'\r\n', b'\n')
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                changed.append(item['path'])
        rows.append({'repository': upstream['repository'], 'commit': upstream['commit'],
                     'checked_files': len(upstream['required_files']), 'missing': missing, 'changed': changed,
                     'ok': not missing and not changed})
    return rows
