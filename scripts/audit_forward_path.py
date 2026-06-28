#!/usr/bin/env python3
"""Static forward-path audit for learned-model submissions.

Scans a model's forward path for classical-arithmetic shortcuts that would make a
"purely learned" claim false: modular reduction, 3-arg pow, big-int math libraries,
lookup tables, or comparison against the modulus to correct the output. Reusable across
any model-submission paper; pairs with the weight-randomization collapse test (dynamic
evidence) to make the no-shortcut claim hard to dismiss. This is a provenance/repro
audit, NOT a proof.

Usage:  python3 scripts/audit_forward_path.py <model.py> [more.py ...]
Exit 0 = clean; 1 = banned operation found.
"""
import re
import sys
import os

# (pattern, why) -- matched against code lines with comments/strings stripped.
BANNED = [
    (r"%\s*p\b|%\s*self\.p\b|%\s*mod\b", "modular reduction (% p) in forward path"),
    (r"\bpow\s*\([^)]*,[^)]*,[^)]*\)", "3-arg pow(base, exp, mod) -- built-in modexp"),
    (r"\btorch\.(remainder|fmod)\b|\bnp\.(remainder|mod|fmod)\b", "tensor modular reduction"),
    (r"\bmath\.(fmod|remainder)\b", "math modular reduction"),
    (r"\b(gmpy2|sympy|Crypto|gmpy)\b", "big-integer / symbolic-math library"),
    (r"\bdivmod\s*\(", "divmod() in forward path"),
    (r">=?\s*p\b|>=?\s*self\.p\b", "comparison against the modulus (output correction)"),
    (r"\blookup\b|\btable\[|LUT\b", "lookup table in forward path"),
]
# Methods considered the forward/inference path. Lines outside these are reported as
# context only (e.g. an import or a training helper), not a hard fail.
FORWARD_HINTS = ("def forward", "def _step", "def _reduce", "def _mul", "def predict",
                 "def infer", "def __call__", "def step")


def _strip(line):
    line = re.sub(r"#.*$", "", line)
    line = re.sub(r"(['\"]).*?\1", "", line)  # crude string strip
    return line


def audit_file(path):
    src = open(path, encoding="utf-8", errors="ignore").read()
    lines = src.splitlines()
    # mark which lines are inside a forward-path method (until dedent to col 0 def/class)
    in_fwd = False
    findings = []
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_fwd = any(h in raw for h in FORWARD_HINTS)
        code = _strip(raw)
        for pat, why in BANNED:
            if re.search(pat, code):
                findings.append((i + 1, "FORWARD" if in_fwd else "context", why, stripped[:90]))
    return findings


def run(paths):
    any_forward = False
    print("FORWARD-PATH AUDIT")
    for p in paths:
        print(f"\n  file: {os.path.basename(p)}")
        fs = audit_file(p)
        forward_hits = [f for f in fs if f[1] == "FORWARD"]
        if not fs:
            print("    no banned arithmetic operations detected")
        for ln, where, why, txt in fs:
            tag = "[BANNED-IN-FORWARD]" if where == "FORWARD" else "[context]"
            print(f"    {tag} line {ln}: {why}\n        {txt}")
        any_forward = any_forward or bool(forward_hits)
    print("\nAllowed in a clean forward path: bit extraction, tensor indexing, the learned"
          "\ncell, and output decoding by the official scorer.")
    if any_forward:
        print("\nRESULT: FAIL -- classical arithmetic in the forward path; the 'purely learned' claim is unsupported.")
        return 1
    print("\nRESULT: PASS -- no classical modular arithmetic / lookup / compare-against-p in the forward path.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: audit_forward_path.py <model.py> [...]")
        sys.exit(2)
    sys.exit(run(sys.argv[1:]))
