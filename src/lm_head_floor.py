# Jitter floor of the LM-head logprobs across repeated identical runs (and between two run sets).
# Per pair of runs and per generated position with an identical token prefix: max |delta logprob| over the
# token ids present in both top-N lists. Logprobs are normalized over the M10 LM-head candidates (exact differences).
import csv, itertools, json, sys
def load(d):
    rows = list(csv.DictReader(open(d + '/runs.tsv'), delimiter='\t'))
    out = []
    for r in rows:
        g = json.load(open(r['run_dir'].rstrip('/') + '/client/vllm/resident_vllm_client_result.json'))['generation'][0]
        if 'top_logprobs' in g: out.append((g['generated_token_ids'], g['top_logprobs']))
    return out
def pair(a, b):
    ta, la = a; tb, lb = b
    worst = []
    for k in range(min(len(ta), len(tb))):
        if ta[:k] != tb[:k]: break
        da, db = dict(map(tuple, la[k])), dict(map(tuple, lb[k]))
        common = set(da) & set(db)
        worst.append(max(abs(da[t] - db[t]) for t in common) if common else float('nan'))
    return worst
sets = [load(d) for d in sys.argv[1:]]
for i, s in enumerate(sets):
    per = [pair(a, b) for a, b in itertools.combinations(s, 2)]
    allv = [v for p in per for v in p]
    pos = [max(p[k] for p in per if len(p) > k) for k in range(max(map(len, per)))]
    print(f'set {i}: {len(s)} runs, floor max {max(allv):.4f}, median pair-position {sorted(allv)[len(allv)//2]:.4f}')
    print('  per-position floor:', ' '.join(f'{v:.3f}' for v in pos))
if len(sets) == 2:
    cross = [v for a in sets[0] for b in sets[1] for v in pair(a, b)]
    print(f'cross set0 x set1: max {max(cross):.4f}, median {sorted(cross)[len(cross)//2]:.4f}')
