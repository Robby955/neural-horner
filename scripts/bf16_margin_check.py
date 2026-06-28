"""Is bf16 decision-equivalent to fp32 for the per-step threshold?

Resolves the determinism/hardware risk of bf16 inference: runs N tier-10-scale
problems through the full reduce/reduce/multiply pipeline in BOTH fp32 and bf16,
compares the final integer outputs (any flipped answer?), and records the minimum
|logit| over every step (the safety margin: if min|logit| >> bf16 rounding error,
no hardware can flip the sigmoid>0.5 decision, so bf16 == fp32 decisions).

Usage: python bf16_margin_check.py <submission_dir> [--n 200]
"""
from __future__ import annotations
import argparse, importlib.util, random, sys
from pathlib import Path
import torch

_SMALL = (2,3,5,7,11,13,17,19,23,29,31,37)
def is_prime(n, rng):
    if n < 2: return False
    for p in _SMALL:
        if n % p == 0: return n == p
    d,r = n-1,0
    while d%2==0: d//=2; r+=1
    for a in list(_SMALL)+[rng.randrange(2,n-1) for _ in range(16)]:
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(r-1):
            x=(x*x)%n
            if x==n-1: break
        else: return False
    return True

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("submission"); ap.add_argument("--n",type=int,default=200)
    args=ap.parse_args()
    sub=Path(args.submission).resolve()
    spec=importlib.util.spec_from_file_location("m",sub/"model.py"); m=importlib.util.module_from_spec(spec); sys.modules["m"]=m; spec.loader.exec_module(m)
    dev=torch.device("cuda")
    ck=torch.load(sub/"weights.pt",map_location=dev,weights_only=True); L=int(ck["L"])
    cell=m.Cell(**ck.get("config",{})); cell.load_state_dict(ck["state_dict"]); cell.to(dev).eval()
    to_bits=m.to_bits_limbs
    rng=random.Random(31337)
    margin={"min":1e9}

    @torch.no_grad()
    def step(s,x,p,d,dtype):
        feat=torch.stack([s,x,p],dim=-1)
        if dtype==torch.bfloat16:
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                lg=cell(feat,d)
            lg=lg.float()
        else:
            lg=cell(feat,d)
            margin["min"]=min(margin["min"], lg.abs().min().item())
        return (torch.sigmoid(lg)>0.5).float()

    @torch.no_grad()
    def reduce_(bitmat,p_bits,dtype):
        n,width=bitmat.shape
        s=torch.zeros((n,L),device=dev); ones=to_bits([1]*n,dev,L).float()
        for pos in range(width):
            s=step(s,ones,p_bits,bitmat[:,pos],dtype)
        return s
    @torch.no_grad()
    def mul_(ra,rb,p_bits,dtype):
        n=ra.shape[0]; s=torch.zeros((n,L),device=dev); rbl=rb.long()
        for k in range(L):
            s=step(s,ra,p_bits,rbl[:,k],dtype)
        return s
    def decode(bits):
        out=[]
        for row in bits.long().tolist():
            v=0
            for b in row: v=v*2+b
            out.append(v)
        return out

    # tier-10-scale problems: p ~ L-bit prime, a,b ~ 2L-bit
    lo,hi=1<<(L-2),(1<<L)-1
    A,B,P=[],[],[]
    while len(P)<args.n:
        p=rng.randint(lo,hi)|1
        if is_prime(p,rng): P.append(p); A.append(rng.randrange(0,1<<(2*L))); B.append(rng.randrange(0,1<<(2*L)))
    truth=[(a*b)%p for a,b,p in zip(A,B,P)]
    def to_mat(vals,width):
        mat=torch.zeros((len(vals),width),dtype=torch.long,device=dev)
        for r,v in enumerate(vals):
            bl=[int(c) for c in bin(v)[2:]] if v>0 else [0]
            mat[r,width-len(bl):]=torch.tensor(bl,device=dev)
        return mat
    w=2*L
    amat=to_mat(A,w); bmat=to_mat(B,w)
    def full(dtype):
        p_bits=to_bits(P,dev,L).float()
        ra=reduce_(amat,p_bits,dtype); rb=reduce_(bmat,p_bits,dtype)
        return decode(mul_(ra,rb,p_bits,dtype))
    out32=full(torch.float32); outbf=full(torch.bfloat16)
    match_fp32=sum(o==t for o,t in zip(out32,truth))
    match_bf16=sum(o==t for o,t in zip(outbf,truth))
    agree=sum(a==b for a,b in zip(out32,outbf))
    print(f"submission={sub.name} L={L} n={args.n}")
    print(f"fp32 vs ground-truth: {match_fp32}/{args.n}")
    print(f"bf16 vs ground-truth: {match_bf16}/{args.n}")
    print(f"bf16 vs fp32 agreement (flipped answers?): {agree}/{args.n}  -> {args.n-agree} flips")
    print(f"min |logit| over all steps (fp32): {margin['min']:.3f}   (bf16 rel-eps ~ 0.008; safe if min|logit| >> a few)")

if __name__=="__main__":
    main()
