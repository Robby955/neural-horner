"""Independent check that a bit-serial submission genuinely computes (a*b) mod p.

Loads a submission directory, picks a prime inside the model's trained width, feeds
operands far LARGER than p (so real reduction is exercised), and compares the model's
output to an independent Python ground truth. Also shows the answer is not any trivial
shortcut (a%p, b%p, or the un-reduced product). With --randomize it reloads with fresh
random weights to confirm the score collapses (the answer is in the weights, not the loop).

Usage (from this directory, with the modchallenge venv):
    python verify_no_shortcut.py bit-serial-v2
    python verify_no_shortcut.py bit-serial-v4 --n 16
    python verify_no_shortcut.py bit-serial-v4 --randomize
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int, rng: random.Random) -> bool:
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    bases = list(_SMALL) + [rng.randrange(2, n - 1) for _ in range(16)]
    for a in bases:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--randomize", action="store_true",
                    help="reload with random weights to confirm the score collapses")
    args = ap.parse_args()

    sub = Path(args.submission).resolve()
    spec = importlib.util.spec_from_file_location("submodel", sub / "model.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["submodel"] = m
    spec.loader.exec_module(m)

    model = m.BitSerialReducer()
    model.load(str(sub))
    if args.randomize:
        torch.manual_seed(12345)
        fresh = m.Cell(**{"dmodel": 96, "hidden": 128})
        model.model.load_state_dict(fresh.state_dict())
        print("[randomized weights -- expecting collapse to 0]")

    L = getattr(model, "L", None) or getattr(m, "L", 32)
    rng = random.Random(2026)
    # a prime near the top of the model's width, and operands a few times larger
    lo, hi = 1 << (L - 2), (1 << L) - 1
    p = next(c for c in iter(lambda: rng.randint(lo, hi), None) if is_prime(c, rng))
    op_bits = min(4 * L, L + 16) if L <= 16 else 3 * L  # operands >> p
    print(f"submission={sub.name}  L={L}  prime p has {p.bit_length()} bits  operand width {op_bits} bits")

    ok = 0
    for i in range(args.n):
        a = rng.randrange(0, 1 << op_bits)
        b = rng.randrange(0, 1 << op_bits)
        digits = model.predict_digits(model.preprocess_a(a), model.preprocess_b(b), model.preprocess_p(p))
        val = 0
        for dbit in digits:
            val = val * 2 + dbit
        truth = (a * b) % p
        match = (val == truth)
        ok += match
        if i < 4:
            print(f"  a>{('p' if a>p else '?')} b>{('p' if b>p else '?')}  model={val}  truth={truth}  match={match}")
    print(f"exact-match: {ok}/{args.n}")

    # show it is not a trivial shortcut
    a = rng.randrange(0, 1 << op_bits); b = rng.randrange(0, 1 << op_bits)
    digits = model.predict_digits(model.preprocess_a(a), model.preprocess_b(b), model.preprocess_p(p))
    val = 0
    for dbit in digits:
        val = val * 2 + dbit
    print(f"not a shortcut: val!=a%p={val != a % p}  val!=b%p={val != b % p}  "
          f"val!=(a%p)*(b%p)={val != (a % p) * (b % p)}  val==(a*b)%p={val == (a * b) % p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
