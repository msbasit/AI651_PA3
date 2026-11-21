"""Download mirrored LargeST raw files, with resumable ranges and digest checks.

The official Kaggle API returned empty HTTP 404 responses on this machine.
LFS SHA256 digests verify transfer integrity, not independent mirror authenticity.
"""
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'ttydsc30/TrafficForecast-LargeST-Project-Dataset'
REVISION = 'fcdd669cb606db00ddb42d526fdde4fb482b2d32'
FILES = {'2020.h5': 'ca_his_raw_2020.h5', '2021.h5': 'ca_his_raw_2021.h5',
         'adjacency.npy': 'ca_rn_adj.npy', 'metadata.csv': 'ca_meta.csv'}

def main():
    revision = REVISION
    r = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=100', timeout=60)
    r.raise_for_status()
    tree = {x['path'].split('/')[-1]: x for x in r.json() if x['type'] == 'file'}
    records = []
    for source, dest in FILES.items():
        item = tree[source]
        size = item['size']
        path = ROOT/'data'/dest
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/raw/{source}'
        expected = item.get('lfs', {}).get('oid')
        block = 32 * 1024**2
        marker = path.with_suffix(path.suffix+'.ranges.json')
        done = set(json.loads(marker.read_text())) if marker.exists() else set()
        if not path.exists():
            with path.open('wb') as f:
                f.truncate(size)
        assert path.stat().st_size == size
        count = (size+block-1)//block
        start = time.perf_counter()
        def fetch(i):
            lo, hi = i*block, min(size, (i+1)*block)-1
            for attempt in range(5):
                try:
                    q = requests.get(url, headers={'Range': f'bytes={lo}-{hi}'}, timeout=180)
                    q.raise_for_status()
                    assert len(q.content) == hi-lo+1
                    if size > block:
                        assert q.status_code == 206
                    with path.open('r+b') as f:
                        f.seek(lo)
                        f.write(q.content)
                    return i
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2*(attempt+1))
        with cf.ThreadPoolExecutor(max_workers=12) as pool:
            for i in pool.map(fetch, [i for i in range(count) if i not in done]):
                done.add(i)
                marker.write_text(json.dumps(sorted(done)))
                if len(done)%16 == 0 or len(done)==count:
                    print(f'{source}: {len(done)}/{count} chunks, elapsed {time.perf_counter()-start:.1f}s', flush=True)
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if expected:
            assert digest == expected, (source, digest, expected)
        records.append(dict(file=dest, url=url, bytes=size, sha256=digest,
                            mirror_revision=revision, verified_lfs_digest=bool(expected)))
        (ROOT/'results'/'download_provenance.json').write_text(json.dumps(records, indent=2))
        print('VERIFIED', dest, digest, flush=True)

if __name__ == '__main__':
    main()
