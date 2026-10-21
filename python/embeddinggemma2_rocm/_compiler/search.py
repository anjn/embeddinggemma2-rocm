"""Finite, architecture-aware kernel candidate sets (CPU-only import)."""
from itertools import product

BLOCKS=(64,128,256,512)
VECTORS=(1,2,4,8,16)
LANES=(4,8,16,32,64)


def candidates(family,device):
    blocks=[b for b in BLOCKS if b<=device['max_threads_per_block']]
    if family in ('gelu','preprocess'):
        return [dict(block=b,vector=v) for b,v in product(blocks,VECTORS)]
    lanes=[l for l in LANES if l<=device['wave_size']]
    if family=='qk':
        return [dict(block=b,lanes=l,packed=False) for b,l in product(blocks,lanes) if b%l==0]+[dict(block=b,lanes=16,packed=True) for b in blocks if device['wave_size']>=16]
    if family=='norm':
        return [dict(block=b,lanes=l) for b,l in product(blocks,lanes) if b%l==0 and 768%(l*4)==0]
    raise ValueError('Unknown kernel family')


def combinations(best):
    families=[f for f in ('gelu','qk','norm','preprocess') if best.get(f)]
    for mask in product((False,True),repeat=len(families)):
        result={f:best[f] if on else None for f,on in zip(families,mask)}
        # Test both the upstream compiled attention and the segmented fast path.
        for segmented in (False,True):
            yield dict(result,segmented=segmented)
