"""Exhaustive per-step exactness of the trained cell on small primes.

The bridge question for provable exactness: does the learned cell, read as a discrete
map (threshold its output bits), compute EXACTLY s' = (2s + d*x) mod p for every state
(s, x in [0,p), d in {0,1}) at small primes p? If yes on a verifiable small-width domain,
the cell is already a discrete exact transition there (and translation-invariance + the
bounded wrap, 2s+dx<3p, lift it). If no, the soft cell must be quantized/distilled first.

Loads a submission's Cell + weights, runs ONE per-step transition exhaustively per prime.
Usage: python per_step_exhaustive.py <submission_dir> [--pmax 64]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--pmax", type=int, default=64)
    args = ap.parse_args()
    sub = Path(args.submission).resolve()
    spec = importlib.util.spec_from_file_location("m", sub / "model.py")
    m = importlib.util.module_from_spec(spec); sys.modules["m"] = m; spec.loader.exec_module(m)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(sub / "weights.pt", map_location=dev, weights_only=True)
    L = int(ck.get("L", 32))
    cell = m.Cell(**ck.get("config", {})); cell.load_state_dict(ck["state_dict"]); cell.to(dev).eval()
    to_bits = m.to_bits_limbs if hasattr(m, "to_bits_limbs") else m.to_bits

    @torch.no_grad()
    def step_bits(s_i, x_i, p_i, d_i):
        s_b = to_bits(s_i, dev, L).float(); x_b = to_bits(x_i, dev, L).float(); p_b = to_bits(p_i, dev, L).float()
        d = torch.tensor(d_i, dtype=torch.long, device=dev)
        feat = torch.stack([s_b, x_b, p_b], dim=-1)
        bits = (torch.sigmoid(cell(feat, d)) > 0.5).long().tolist()
        out = []
        for row in bits:
            v = 0
            for b in row:
                v = v * 2 + b
            out.append(v)
        return out

    primes = [p for p in range(2, args.pmax) if is_prime(p)]
    print(f"submission={sub.name} L={L} primes<{args.pmax}: {len(primes)} device={dev}")
    total_ok = total = 0
    worst = []
    for p in primes:
        s_i, x_i, p_i, d_i, truth = [], [], [], [], []
        for s in range(p):
            for x in range(p):
                for d in (0, 1):
                    s_i.append(s); x_i.append(x); p_i.append(p); d_i.append(d)
                    truth.append((2 * s + d * x) % p)
        ok = 0
        bs = 4096
        for i in range(0, len(s_i), bs):
            out = step_bits(s_i[i:i+bs], x_i[i:i+bs], p_i[i:i+bs], d_i[i:i+bs])
            for j, v in enumerate(out):
                ok += (v == truth[i + j])
        rate = ok / len(s_i)
        total_ok += ok; total += len(s_i)
        if rate < 1.0:
            worst.append((p, round(rate, 4)))
    print(f"per-step exact-match (all states, all primes < {args.pmax}): {total_ok}/{total} = {total_ok/total:.5f}")
    print("primes with ANY per-step error:", worst if worst else "NONE (cell is the exact discrete step on this domain)")


if __name__ == "__main__":
    main()
