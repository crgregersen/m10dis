#!/usr/bin/env python3
"""Extract the PLE byte ranges listed in a ple_shards.tsv into standalone shard files.

ple_shards.tsv rows (7 tab-separated columns, no header):
    shard_id  host_gpu_index  source_path  byte_offset  byte_count  rows  row_width
The PLE loaders (cuda_load_stage9_ple*.cu) open source_path, seek to byte_offset and
read byte_count bytes. This script copies each range into OUT_DIR/ple_shard_<id>.bin and
writes OUT_DIR/ple_shards.tsv with the new path and offset 0 (all other columns unchanged),
so the loaders read identical bytes from the new location.

Verification: SHA-256 of each source range is computed while copying; the written file is
then re-read and hashed independently, and both must match. Sizes must equal byte_count
and rows * row_width. Writes OUT_DIR/SHA256SUMS and OUT_DIR/extract-report.json.
Usage: extract_qwen38_ple_shards.py SOURCE_PLE_SHARDS.tsv OUT_DIR
"""
import hashlib, json, os, sys, time
from pathlib import Path

src_tsv, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = [l.rstrip('\n').split('\t') for l in src_tsv.read_text().splitlines() if l.strip()]
assert rows and all(len(r) == 7 for r in rows), 'expected 7-column ple_shards.tsv'
ids = [int(r[0]) for r in rows]
assert len(set(ids)) == len(ids), 'duplicate shard ids'
CHUNK = 64 << 20
started = time.monotonic(); total = 0; report = []; new_rows = []
for sid, host, path, off, nbytes, nrows, width in rows:
    off, nbytes, nrows, width = int(off), int(nbytes), int(nrows), int(width)
    assert nbytes == nrows * width, f'shard {sid}: bytes != rows*width'
    assert os.path.getsize(path) >= off + nbytes, f'shard {sid}: range beyond end of {path}'
    dst = out / f'ple_shard_{int(sid):03d}.bin'
    h_src = hashlib.sha256(); left = nbytes
    with open(path, 'rb') as fi, open(dst.with_suffix('.tmp'), 'wb') as fo:
        fi.seek(off)
        while left:
            b = fi.read(min(CHUNK, left))
            assert b, f'shard {sid}: short read'
            h_src.update(b); fo.write(b); left -= len(b)
        fo.flush(); os.fsync(fo.fileno())
    os.replace(dst.with_suffix('.tmp'), dst)
    h_dst = hashlib.sha256()
    with open(dst, 'rb') as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # re-read from the NVMe, not the page cache
        while b := f.read(CHUNK):
            h_dst.update(b)
    ok = h_src.hexdigest() == h_dst.hexdigest() and dst.stat().st_size == nbytes
    report.append({'id': int(sid), 'source': path, 'offset': off, 'bytes': nbytes,
                   'sha256': h_dst.hexdigest(), 'match': ok})
    assert ok, f'shard {sid}: verification failed'
    new_rows.append('\t'.join([sid, host, str(dst), '0', str(nbytes), str(nrows), str(width)]))
    total += nbytes
    print(f'PLE_SHARD,{sid},bytes,{nbytes},sha256,{h_dst.hexdigest()[:16]},match,1', flush=True)
(out / 'ple_shards.tsv').write_text('\n'.join(new_rows) + '\n')
(out / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  ple_shard_{r['id']:03d}.bin\n" for r in report))
secs = time.monotonic() - started
(out / 'extract-report.json').write_text(json.dumps({'source_tsv': str(src_tsv), 'shards': len(report),
    'total_bytes': total, 'seconds': secs, 'all_match': all(r['match'] for r in report), 'shards_detail': report}, indent=2) + '\n')
print(f'PLE_EXTRACT_OK,shards,{len(report)},bytes,{total},seconds,{secs:.0f},tsv,{out / "ple_shards.tsv"}', flush=True)
