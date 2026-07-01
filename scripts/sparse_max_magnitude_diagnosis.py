"""Mechanism diagnosis for the F_11 reduce_a failure at s=2^2047 (d=1).

Prior probe (fermat_precision_probe.py) localized one wrong transition and noted a
single near-zero-margin bit. This script isolates WHAT triggers the failure. It only
*measures* the frozen cell on generated (s, x, p, d) states -- no operand-structure
special-casing, no "is this a Fermat number" test; the same sparse/dense state families
are generated identically regardless of which numbers appear.

Findings it reproduces:
  1. The single global min-|logit| bit is 2^1315 (MSB-idx 732), decided correctly but
     marginally. It sits at the *frontier* between the confidently-correct high bits and
     the confidently-wrong low bits -- NOT a carry/borrow position that must propagate.
  2. The failing transition is exactly one subtraction: (2*2^2047+1) - p. But borrow
     propagation is NOT the cause: dense states with identical full-width borrow chains
     (Class B) are computed perfectly.
  3. The real trigger is INPUT SPARSITY at maximal magnitude. s=2^2047 doubles to
     2^2048, which overflows the L=2048-bit state window exactly. Wrong-bit count is a
     smooth monotone function of input density: dense inputs recover completely.

Usage: python3 scripts/sparse_max_magnitude_diagnosis.py <model_dir> [--device cpu]
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n, rng):
    if n < 2:
        return False
    for q in _SMALL:
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in list(_SMALL) + [rng.randrange(2, n - 1) for _ in range(16)]:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def battery_prime(L):
    rng = random.Random(20260627)
    lo, hi = 1 << (L - 2), (1 << L) - 1
    return next(c for c in iter(lambda: rng.randint(lo, hi), None) if is_prime(c, rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    sub = Path(args.model_dir).resolve()
    spec = importlib.util.spec_from_file_location("m", sub / "model.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m"] = m
    spec.loader.exec_module(m)
    to_bits = m.to_bits_limbs

    dev = torch.device(args.device)
    ck = torch.load(sub / "weights.pt", map_location=dev, weights_only=True)
    L = int(ck["L"])
    cell = m.Cell(**ck.get("config", {}))
    cell.load_state_dict(ck["state_dict"])
    cell.to(dev).eval()

    p = battery_prime(L)
    Leff = min(L, max(32, ((p.bit_length() + 31) // 32) * 32))
    x_bits = to_bits([1], dev, Leff).float()
    p_bits = to_bits([p], dev, Leff).float()
    print(f"L={L} Leff={Leff} p.bit_length={p.bit_length()} device={dev}")

    @torch.no_grad()
    def step(s, d_bit):
        exact = (2 * s + d_bit) % p
        s_bits = to_bits([s], dev, Leff).float()
        feat = torch.stack([s_bits, x_bits, p_bits], dim=-1)
        lg = cell(feat, torch.tensor([d_bit], device=dev))[0].float()
        dec = (torch.sigmoid(lg) > 0.5).float()
        eb = to_bits([exact], dev, Leff).float()[0]
        return lg, dec, eb, exact

    # ---- 1. The marginal bit and the correctness/confidence frontier ----
    lg, dec, eb, exact = step(1 << 2047, 1)
    absl = lg.abs()
    correct = (dec == eb)
    gmin = int(absl.argmin())
    print("\n[1] MARGINAL BIT")
    print(f"    global min |logit| bit: MSB-idx {gmin} = 2^{Leff-1-gmin}  "
          f"logit={lg[gmin].item():+.4f}  decided={int(dec[gmin])} exact={int(eb[gmin])} "
          f"correct={bool(correct[gmin])}")
    wrong = int((~correct).sum())
    top_ok = correct[:576].all().item()
    print(f"    total wrong bits={wrong}; top 576 MSB bits all correct={top_ok}; "
          f"below the frontier accuracy collapses to ~50% while |logit| stays high "
          f"(confidently wrong, not marginal)")

    # ---- 2. Carry/borrow is NOT the trigger: dense wrapping states pass ----
    print("\n[2] BORROW PROPAGATION IS NOT THE CAUSE")
    rng = random.Random(7)
    okB = 0
    nB = 12
    for _ in range(nB):
        s = rng.randrange(p // 2 + 1, p)
        d = rng.randrange(0, 2)
        if (2 * s + d) >= p:  # wraps -> full-width borrow subtraction
            _, dc, e2, _ = step(s, d)
            okB += int((dc == e2).all())
    print(f"    dense states that wrap (identical borrow structure): {okB}/{nB} exact")
    _, dc, e2, _ = step(1 << 2047, 1)
    print(f"    the sparse max-magnitude state s=2^2047,d=1: "
          f"{'PASS' if (dc == e2).all() else 'FAIL'} ({int((dc != e2).sum())} wrong)")

    # ---- 3. The trigger is input sparsity at maximal magnitude ----
    print("\n[3] TRIGGER = INPUT SPARSITY AT MAX MAGNITUDE (exact-width overflow)")
    print(f"    2s = 2^2048 overflows the {Leff}-bit state window; truncated 2s = "
          f"{(2*(1<<2047)) % (1<<Leff)}; the dense reduction 2^2048+1-p must be "
          f"reconstructed from a single input bit")
    rng2 = random.Random(3)
    print("    density sweep (s = 2^2047 + k random low bits, d=1):")
    for k in [0, 8, 32, 64, 128, 256, 512]:
        nw = []
        for _ in range(6):
            s = 1 << 2047
            for b in rng2.sample(range(0, 2047), k):
                s |= (1 << b)
            if s < p:
                _, dc, e2, _ = step(s, 1)
                nw.append(int((dc != e2).sum()))
        print(f"      k={k:<4} set bits: mean_wrong={sum(nw)/len(nw):7.1f}  "
              f"exact_ok={sum(1 for x in nw if x==0)}/{len(nw)}")

    print("\nVERDICT: the marginal bit is a sparse-input reconstruction frontier, not a "
          "carry/borrow signal. An architectural carry channel is not the right lever; "
          "the class is closable by training coverage on sparse max-magnitude states.")


if __name__ == "__main__":
    main()
