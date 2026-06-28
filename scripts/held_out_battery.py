"""Genuinely HELD-OUT adversarial battery.

The earlier adversarial_stress.py families (powers of two, sparse, near-multiple,
all-ones, symmetric, multiply-control) are exactly the families used to refine the
DAgger training data, so passing them is in-distribution fit, not robustness. This
battery uses families DISJOINT from training: Fibonacci-valued operands, Fermat
numbers, alternating bit patterns, fixed-Hamming-weight operands, operands whose
PRODUCT straddles a multiple of p (a*b ~ k*p +/- small, a reduction-boundary stress
distinct from the training "operand near k*p"), and a structurally-chosen prime
(p = 3 mod 4). Reports exact-match per family vs an independent Python ground truth.

Usage: python held_out_battery.py <submission_dir> [--n 128]
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n, rng):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
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


def prime_3mod4(lo, hi, rng):
    for _ in range(100000):
        c = rng.randint(lo, hi) | 3  # ensures odd and == 3 mod 4
        if c % 4 == 3 and is_prime(c, rng):
            return c
    return None


def run_cases(model, p, cases):
    inputs = [(model.preprocess_a(a), model.preprocess_b(b), model.preprocess_p(p)) for a, b in cases]
    outs = model.predict_digits_batch(inputs)
    ok, fails = 0, []
    for (a, b), digits in zip(cases, outs):
        val = 0
        for dd in digits:
            val = val * 2 + dd
        truth = (a * b) % p
        if val == truth:
            ok += 1
        elif len(fails) < 3:
            fails.append((a.bit_length(), b.bit_length(), val, truth))
    return ok, len(cases), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    sub = Path(args.submission).resolve()
    spec = importlib.util.spec_from_file_location("submodel", sub / "model.py")
    m = importlib.util.module_from_spec(spec); sys.modules["submodel"] = m
    spec.loader.exec_module(m)
    model = m.BitSerialReducer(); model.load(str(sub))
    L = getattr(model, "L", None) or getattr(m, "L", 32)
    rng = random.Random(20260627)
    lo, hi = 1 << (L - 2), (1 << L) - 1
    p = next(c for c in iter(lambda: rng.randint(lo, hi), None) if is_prime(c, rng))
    W = 2 * L
    N = args.n
    print(f"submission={sub.name} L={L} prime={p.bit_length()}b operand_width={W}b n={N}  [HELD-OUT families]")

    cats = {}
    # Fibonacci-valued operands up to ~2^W
    fib = [1, 2]
    while fib[-1] < (1 << W):
        fib.append(fib[-1] + fib[-2])
    cats["fibonacci values"] = [(rng.choice(fib), rng.choice(fib)) for _ in range(N)]
    # Fermat numbers 2^(2^k)+1
    fermat = [(1 << (1 << k)) + 1 for k in range(0, 12) if (1 << (1 << k)) + 1 < (1 << W)]
    cats["fermat numbers"] = [(rng.choice(fermat), rng.randrange(0, 1 << W)) for _ in range(N)]
    # alternating bit patterns
    a5 = int("01" * (W // 2), 2); aA = int("10" * (W // 2), 2)
    cats["alternating bits"] = [(rng.choice([a5, aA]), rng.choice([a5, aA, rng.randrange(0, 1 << W)])) for _ in range(N)]
    # fixed Hamming weight W/2, random positions (distinct from sparse / all-ones)
    def hw(half):
        bits = rng.sample(range(W), half)
        v = 0
        for b in bits:
            v |= (1 << b)
        return v
    cats["fixed Hamming weight W/2"] = [(hw(W // 2), hw(W // 2)) for _ in range(N)]
    # PRODUCT straddles a multiple of p: a*b ~ k*p +/- small
    straddle = []
    for _ in range(N):
        a = rng.randrange(1 << (W // 2 - 1), 1 << (W // 2))
        k = rng.randrange(1, 1 << (W // 2))
        target = k * p + rng.randrange(-3, 4)
        b = max(1, target // a)
        straddle.append((a, b))
    cats["product straddles k*p"] = straddle
    # structured prime p = 3 mod 4, random operands
    p2 = prime_3mod4(lo, hi, rng) or p
    cats[f"prime=3mod4 (random ops)"] = [(rng.randrange(0, 1 << W), rng.randrange(0, 1 << W)) for _ in range(N)]

    total_ok = total = 0
    for name, cases in cats.items():
        pp = p2 if name.startswith("prime=3mod4") else p
        ok, tot, fails = run_cases(model, pp, cases)
        total_ok += ok; total += tot
        flag = "OK" if ok == tot else "FAIL"
        print(f"  [{flag}] {name}: {ok}/{tot}" + ("" if ok == tot else f"  e.g.{fails}"))
    print(f"TOTAL held-out exact-match: {total_ok}/{total} = {total_ok/total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
