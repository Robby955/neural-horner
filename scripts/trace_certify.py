"""Trace-level certification for the bit-serial reducer.

For each evaluated problem, logs every per-step transition the model actually takes
along the reduce/reduce/multiply rollout and checks it against the exact symbolic
transition s' = (2 s + d x) mod p. A passed rollout then carries a concrete
certificate -- not just "the final answer matched" but "every learned step matched the
exact recurrence." For a failed rollout it localizes the FIRST divergent step.

This is Level 3 of the verification stack (Level 1 = Lean proof of the integer
algorithm; Level 2 = exhaustive small-prime transition check; Level 3 = trace
certification on the actual evaluated rollouts). It certifies the evaluated rollouts,
not all inputs.

Usage:
    python3 trace_certify.py <model_dir> [--small] [--cases N] [--bits LO HI]
Reusable for any step-decomposable learned reducer: subclass the model and override
the per-step call to log + verify.
"""
import argparse
import random
import sys

import torch


def _bits_to_int(row) -> int:
    # row: 1-D tensor of 0/1, MSB-first
    return int("".join("1" if b > 0.5 else "0" for b in row.tolist()) or "0", 2)


def make_tracer(model_module):
    BitSerialReducer = model_module.BitSerialReducer

    class TracingReducer(BitSerialReducer):
        def predict_digits_batch(self, inputs):
            self._pvals = [int(p) for (_, _, p) in inputs]
            self._phase = "?"
            self.trace_ok = 0
            self.trace_bad = 0
            self.first_div = {}  # row -> (phase, step, s, x, d, neural, exact)
            self._step_idx = {}
            return super().predict_digits_batch(inputs)

        def _reduce(self, bit_lists, p_bits, dev):
            self._phase = "reduce_a" if self._phase in ("?",) else ("reduce_b" if self._phase == "reduce_a" else self._phase)
            return super()._reduce(bit_lists, p_bits, dev)

        def _mul(self, ra, rb, p_bits):
            self._phase = "multiply"
            return super()._mul(ra, rb, p_bits)

        def _step(self, s_bits, x_bits, p_bits, d):
            s_prime = super()._step(s_bits, x_bits, p_bits, d)
            n = s_bits.shape[0]
            for r in range(n):
                p = self._pvals[r]
                s_i = _bits_to_int(s_bits[r])
                x_i = _bits_to_int(x_bits[r])
                d_i = int(d[r].item() if hasattr(d[r], "item") else d[r])
                neural = _bits_to_int(s_prime[r])
                exact = (2 * s_i + d_i * x_i) % p
                key = (self._phase, r)
                self._step_idx[key] = self._step_idx.get(key, -1) + 1
                if neural == exact:
                    self.trace_ok += 1
                else:
                    self.trace_bad += 1
                    if r not in self.first_div:
                        self.first_div[r] = (self._phase, self._step_idx[key], s_i, x_i, d_i, neural, exact)
            return s_prime

    return TracingReducer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--small", action="store_true", help="local CPU validation on small primes")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--bits", type=int, nargs=2, default=[6, 12])
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    sys.path.insert(0, args.model_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", args.model_dir.rstrip("/") + "/model.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)

    Tracer = make_tracer(mm)
    m = Tracer()
    m.load(args.model_dir)

    rng = random.Random(args.seed)

    def randprime(lo, hi):
        while True:
            c = rng.randrange(1 << (lo - 1), 1 << hi)
            if c > 2 and all(c % q for q in range(2, int(c ** 0.5) + 1)):
                return c

    inputs, truths = [], []
    for _ in range(args.cases):
        lo, hi = args.bits
        p = randprime(lo, hi)
        a = rng.randrange(0, 1 << (hi + 2))
        b = rng.randrange(0, 1 << (hi + 2))
        inputs.append((m.preprocess_a(a), m.preprocess_b(b), m.preprocess_p(p)))
        truths.append((a * b) % p)

    digits = m.predict_digits_batch(inputs)
    finals = [int("".join(str(int(x)) for x in d) or "0", 2) for d in digits]
    final_ok = sum(1 for f, t in zip(finals, truths) if f == t)

    print(f"TRACE CERTIFICATION  model={args.model_dir}  cases={args.cases}  prime bits={args.bits}")
    print(f"  final-answer correct : {final_ok}/{len(truths)}")
    print(f"  transitions verified : {m.trace_ok}/{m.trace_ok + m.trace_bad}  (against s' = (2s + d x) mod p)")
    if m.first_div:
        print(f"  first divergences ({len(m.first_div)} rollouts):")
        for r, (phase, step, s_i, x_i, d_i, neu, exa) in list(m.first_div.items())[:5]:
            print(f"    case {r}: phase={phase} step={step}  s={s_i} x={x_i} d={d_i}  neural={neu} exact={exa}")
    else:
        print("  every transition on every evaluated rollout matched the exact recurrence.")
    return 0 if m.trace_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
