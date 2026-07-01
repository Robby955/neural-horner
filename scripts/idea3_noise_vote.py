"""Idea 3: test-time Gaussian noise + majority vote at the F_11 failing step.

The Cell has no dropout (MC-dropout unavailable), so the only compliant test-time
stochastic lever is Gaussian perturbation of the hidden activations. This injects
i.i.d. Gaussian noise into the GRU input embedding (uniform, operand-agnostic -- it
does NOT inspect operand structure), runs K samples at the s=2^2047,d=1,x=1 step,
majority-votes each output bit, and checks whether the vote recovers the exact
transition. Also reports how many of the deterministically-wrong bits the vote flips
to correct. Swept over several noise scales.

Usage: PYTHONPATH=<mc-src> python3 scripts/idea3_noise_vote.py model [--K 129]
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
    d,r=n-1,0
    while d%2==0: d//=2; r+=1
    for a in list(_SMALL)+[rng.randrange(2,n-1) for _ in range(16)]:
        x=pow(a,d,n)
        if x in (1,n-1): continue
        for _ in range(r-1):
            x=(x*x)%n
            if x==n-1: break
        else: return False
    return True

def battery_prime(L):
    rng=random.Random(20260627); lo,hi=1<<(L-2),(1<<L)-1
    return next(c for c in iter(lambda: rng.randint(lo,hi),None) if is_prime(c,rng))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--K",type=int,default=129)
    args=ap.parse_args()
    sub=Path(args.model_dir).resolve()
    spec=importlib.util.spec_from_file_location("m",sub/"model.py"); m=importlib.util.module_from_spec(spec)
    sys.modules["m"]=m; spec.loader.exec_module(m)
    dev=torch.device("cpu")
    ck=torch.load(sub/"weights.pt",map_location=dev,weights_only=True); L=int(ck["L"])
    cell=m.Cell(**ck["config"]); cell.load_state_dict(ck["state_dict"]); cell.to(dev).eval()
    to_bits=m.to_bits_limbs
    p=battery_prime(L); Leff=min(L,max(32,((p.bit_length()+31)//32)*32))
    s_state=1<<2047; d_bit=1
    exact_next=(2*s_state+d_bit)%p
    exact_bits=to_bits([exact_next],dev,Leff).float()[0]
    s_bits=to_bits([s_state],dev,Leff).float(); x_bits=to_bits([1],dev,Leff).float()
    p_bits=to_bits([p],dev,Leff).float(); d=torch.tensor([d_bit],dtype=torch.long,device=dev)
    feat=torch.stack([s_bits,x_bits,p_bits],dim=-1)

    @torch.no_grad()
    def noisy_logits(sigma):
        # replicate Cell.forward but inject Gaussian noise into the GRU input embedding
        x=cell.in_proj(feat)+cell.d_emb(d)[:,None,:]
        if sigma>0: x=x+sigma*torch.randn_like(x)
        h,_=cell.gru(x)
        return cell.head(h).squeeze(-1)[0]

    # deterministic baseline
    det=(torch.sigmoid(noisy_logits(0.0))>0.5).float()
    det_wrong=(det!=exact_bits)
    n_det_wrong=int(det_wrong.sum().item())
    print(f"L={L} step s=2^2047 d=1  deterministic wrong_bits={n_det_wrong}  det transition correct={(int(''.join(str(int(b)) for b in det.tolist()),2)==exact_next)}")
    print(f"K={args.K} votes per bit; noise on GRU input embedding (uniform, operand-agnostic)")
    torch.manual_seed(0)
    for sigma in [0.02,0.05,0.1,0.2,0.5,1.0]:
        votes=torch.zeros(Leff)
        for _ in range(args.K):
            votes+=(torch.sigmoid(noisy_logits(sigma))>0.5).float()
        voted=(votes>(args.K/2)).float()
        voted_int=int("".join(str(int(b)) for b in voted.tolist()),2)
        vote_wrong=int((voted!=exact_bits).sum().item())
        # of the deterministically-wrong bits, how many did the vote fix?
        fixed=int(((det_wrong)&(voted==exact_bits)).sum().item())
        broke=int(((~det_wrong)&(voted!=exact_bits)).sum().item())
        print(f"  sigma={sigma:<4}: voted transition correct={voted_int==exact_next}  "
              f"vote_wrong_bits={vote_wrong}  fixed_of_{n_det_wrong}={fixed}  newly_broken={broke}")

if __name__=="__main__":
    main()
