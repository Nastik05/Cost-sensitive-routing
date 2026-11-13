"""Portable paths for the published CLINC cache and original experiment code."""
from pathlib import Path
import hashlib
import json
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
OLD_CODE = ROOT.parent
sys.path.insert(0, str(OLD_CODE))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item()) + '\n')
    temporary.replace(path)


def verify():
    """Verify the published immutable cache, rather than unrelated local files."""
    metadata = json.loads((ROOT / 'artifacts/complete.json').read_text())
    assert sha(ROOT / 'artifacts/router_cache.npz') == metadata['cache_sha256']
    return 1


def log(message, **fields):
    value = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'message': message, **fields}
    with (ROOT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)
