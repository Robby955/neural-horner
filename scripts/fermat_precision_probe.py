"""Fermat F_11 precision probe: fp32 vs bf16 at the known failing transition.

Reconstructs the held_out_battery prime (seed 20260627), then runs reduce_a of
F_11 = 2^2048+1 mod p under several precision modes on the loaded cell, tracing
every step against the exact recurrence s' = (2s + d*x) mod p. Reports:
  - whether reduce_a is exact in each mode,
  - the first divergent step,
  - the raw per-bit logits at the s=2^2047 step (the documented failure point),
    so we can see whether the wrong bit(s) are CONFIDENT (|logit| large) or
    MARGINAL (|logit| near 0).

No operand-structure special-casing anywhere: this only *measures* the existing
frozen cell on a fixed input. Usage:
    PYTHONPATH=<mc-src> python3 scripts/fermat_precision_probe.py <model_dir> [--device cpu|mps]
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
    for p in _SMALL:
        if n % p == 0:
            return n == p
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
    """Exactly reproduce held_out_battery.py's prime selection."""
    rng = random.Random(20260627)
    lo, hi = 1 << (L - 2), (1 << L) - 1
    return next(c for c in iter(lambda: rng.randint(lo, hi), None) if is_prime(c, rng))


def bits_to_int(row):
    return int("".join("1" if b > 0.5 else "0" for b in row.tolist()) or "0", 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--operand", default="F11",
                    help="F11 (=2^2048+1) or an integer")
    ap.add_argument("--full", action="store_true",
                    help="also run the full fp32 reduce_a rollout end-to-end")
    args = ap.parse_args()

    sub = Path(args.model_dir).resolve()
    spec = importlib.util.spec_from_file_location("m", sub / "model.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["m"] = m
    spec.loader.exec_module(m)

    dev = torch.device(args.device)
    ck = torch.load(sub / "weights.pt", map_location=dev, weights_only=True)
    L = int(ck["L"])
    cell = m.Cell(**ck.get("config", {}))
    cell.load_state_dict(ck["state_dict"])
    cell.to(dev).eval()
    to_bits = m.to_bits_limbs

    p = battery_prime(L)
    print(f"L={L} device={dev} battery-prime bit_length={p.bit_length()}")
    print(f"p = {p}")

    if args.operand == "F11":
        a = (1 << 2048) + 1
    else:
        a = int(args.operand)
    a_bits = [int(c) for c in bin(a)[2:]]
    print(f"operand a bit_length={a.bit_length()} (F_11=2^2048+1)  width={len(a_bits)}")

    # p_bits sized as in the model: _Leff = min(L, max(32, ceil(bits/32)*32))
    Leff = min(L, max(32, ((p.bit_length() + 31) // 32) * 32))
    p_bits = to_bits([p], dev, Leff).float()
    x_bits = to_bits([1], dev, Leff).float()  # reduce uses multiplicand 1

    import copy
    cell_bf16 = copy.deepcopy(cell).to(torch.bfloat16)

    @torch.no_grad()
    def logits_of(s_bits, x_b, p_b, d, mode):
        feat = torch.stack([s_bits, x_b, p_b], dim=-1)
        if mode == "fp32":
            return cell(feat, d).float()
        if mode == "autocast_bf16":
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                lg = cell(feat, d)
            return lg.float()
        if mode == "hard_bf16":
            lg = cell_bf16(feat.to(torch.bfloat16), d).float()
            return lg
        raise ValueError(mode)

    def run_reduce(mode):
        """Trace reduce_a of `a` mod p under `mode`. Returns (final_int, ok, first_div, n_bad)."""
        s_bits = torch.zeros((1, Leff), device=dev)
        d_seq = torch.tensor(a_bits, dtype=torch.long, device=dev)
        first_div = None
        n_bad = 0
        for pos in range(len(a_bits)):
            d = d_seq[pos:pos + 1]
            s_in_int = bits_to_int(s_bits[0])
            lg = logits_of(s_bits, x_bits, p_bits, d, mode)
            s_bits = (torch.sigmoid(lg) > 0.5).float()
            neural = bits_to_int(s_bits[0])
            exact_s = (2 * s_in_int + int(d.item()) * 1) % p
            if neural != exact_s:
                n_bad += 1
                if first_div is None:
                    first_div = (pos, s_in_int, int(d.item()), neural, exact_s)
        final = bits_to_int(s_bits[0])
        return final, (final == (a % p)), first_div, n_bad

    target_true = a % p
    print(f"exact (a mod p) bit_length={target_true.bit_length()}")
    print("=" * 70)

    # ---- Fast single-step diagnostic at the directly-constructed s=2^2047 ----
    print("SINGLE-STEP DIAGNOSTIC at s=2^2047, d=1, x=1  (the documented failing transition)")
    s_state = 1 << 2047
    d_bit = a_bits[-1]  # last bit of F_11 = 1
    s_bits = to_bits([s_state], dev, Leff).float()
    d = torch.tensor([d_bit], dtype=torch.long, device=dev)
    exact_next = (2 * s_state + d_bit * 1) % p
    exact_bits = to_bits([exact_next], torch.device("cpu"), Leff).float()[0]
    print(f"   exact next state (2*2^2047+{d_bit}) mod p has bit_length {exact_next.bit_length()}")
    for mode in ["fp32", "autocast_bf16", "hard_bf16"]:
        try:
            lg = logits_of(s_bits, x_bits, p_bits, d, mode)[0].detach().cpu()
        except Exception as e:  # noqa
            print(f"   [{mode}] ERROR: {type(e).__name__}: {e}")
            continue
        decided = (torch.sigmoid(lg) > 0.5).float()
        neural_int = int("".join(str(int(b)) for b in decided.tolist()), 2)
        wrong_mask = (decided != exact_bits)
        n_wrong = int(wrong_mask.sum().item())
        absl = lg.abs()
        step_correct = (neural_int == exact_next)
        print(f"   [{mode}] transition correct={step_correct}  wrong_bits={n_wrong}  "
              f"min|logit|(all {Leff})={absl.min().item():.4f}")
        if n_wrong:
            widx = torch.nonzero(wrong_mask).flatten().tolist()
            wrong_logit_abs = absl[wrong_mask]
            print(f"        |logit| at wrong bits: min={wrong_logit_abs.min().item():.4f} "
                  f"max={wrong_logit_abs.max().item():.4f} mean={wrong_logit_abs.mean().item():.4f}")
            for bi in widx[:8]:
                print(f"        bit[msb-idx {bi} = 2^{Leff-1-bi}]: logit={lg[bi].item():+.4f}  "
                      f"decided={int(decided[bi].item())} exact={int(exact_bits[bi].item())}")
    print("=" * 70)

    if args.full:
        print("FULL reduce_a rollout of F_11 (end-to-end, each step fed prev thresholded output)")
        for mode in ["fp32", "autocast_bf16", "hard_bf16"]:
            final, ok, fd, n_bad = run_reduce(mode)
            print(f"   [{mode}] reduce_a final==(a mod p): {ok}   diverging steps: {n_bad}")
            if fd:
                pos, s_i, d_i, neu, exa = fd
                is2047 = (s_i == (1 << 2047))
                print(f"      first divergence at step {pos}: s bit_length={s_i.bit_length()} "
                      f"d={d_i}  (s==2^2047: {is2047})")
            else:
                print("      every step matched the exact recurrence (reduce_a exact)")


if __name__ == "__main__":
    main()
