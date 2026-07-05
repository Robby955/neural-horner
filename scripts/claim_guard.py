#!/usr/bin/env python3
"""Fail-closed claim guard for the NeuralHorner paper.

Codifies the 2026-07-04 catch: the paper claimed the full 1100-problem battery
was "1100/1100" while the SAIR platform's own T0 plain-multiplication diagnostic
scores 40/100, so the true raw all-tier count is 1040/1100. That is an overclaim
contradicted by our own platform data. This guard makes it impossible to
reintroduce silently.

Platform ground truth (SAIR eval 2026-06-27):
  T1-T10 = 100/100 each = 1000/1000  (ranked "overall accuracy 100%")
  T0 (plain-multiplication diagnostic, no reduction) = 40/100
  raw all-tier = 1040/1100

Usage:  python scripts/claim_guard.py [paper/paper_neuralhorner.tex]
Exit 0 = clean; exit 1 = overclaim / inconsistency (CI-ready).
"""
import re
import sys
import pathlib

tex = sys.argv[1] if len(sys.argv) > 1 else "paper/paper_neuralhorner.tex"
src = pathlib.Path(tex).read_text()
low = src.lower()
fails = []

# 1. The specific overclaim must never reappear.
if re.search(r"1100\s*/\s*1100", src):
    fails.append("'1100/1100' perfect-battery claim present -- platform truth is 1040/1100 (T0=40/100).")

# 2. If the T0 diagnostic is discussed at all, its honest 40/100 must be stated (no softening).
mentions_t0 = ("mathrm{t0}" in low) or ("plain-multiplication" in low) or ("plain multiplication" in low)
if mentions_t0 and not re.search(r"40\s*/\s*100", src):
    fails.append("T0 plain-multiplication is discussed but its 40/100 score is missing.")

# 3. If the full 1100-problem battery is referenced, the true 1040/1100 count must be present.
refs_full = bool(re.search(r"1100[- ]problem", low)) or bool(re.search(r"full.{0,40}1100", low))
if refs_full and "1040/1100" not in src:
    fails.append("Full 1100-problem battery referenced but the true raw count 1040/1100 is absent.")

if fails:
    print("CLAIM GUARD FAILED (paper contradicts platform ground truth):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("claim guard OK: no 1100/1100 overclaim; T0=40/100 and 1040/1100 consistent with platform.")
