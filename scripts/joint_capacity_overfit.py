"""Idea 2 test: JOINT one-step capacity ceiling, 128 vs 256, WITH the max-borrow class.

Frozen one-step overfit that isolates the transition function s' = (2s + d*x) mod p
from rollout drift. Distinct from the original frozen_overfit.py in one way that matters:
its corpus EXPLICITLY includes the max-input-magnitude "max-borrow" transitions
(s = 2^(L-1) and a band just below it, d in {0,1}, x=1, over many top-bit-length primes)
-- the exact class the deployed model fails on (Fermat F_11 reduce_a at s=2^2047). The
original boundary sampler only targets *result*-near-boundary states (2s+dx = kp+eps,
small eps) and its family harvest records every 10th step, so it can miss s=2^(L-1).

Decision logic (per the frozen_overfit design):
  128 fits maxborrow held-out ~1.00 -> capacity SUFFICIENT jointly; ceiling is coverage/drift
  128 cannot fit maxborrow but 256 can -> CAPACITY ceiling (Idea 2 is the right lever)
  neither fits -> deeper representational limit; Idea 2 as specified will not close it

Env: L(2048) HID(128) RESUME(path to v8 weights.pt) SCRATCH(0 -> warm if HID==128)
     STEPS(4000) BATCH(192) NPRIME(48) SEED(51) OUT.
No operand-structure special-casing: this only trains/evaluates the frozen transition
map on generated (s,x,p,d) -> y labels.
"""
from __future__ import annotations
import json, math, os, pathlib, random, time
import torch
from torch import nn

L = int(os.environ.get("L", "2048"))
HID = int(os.environ.get("HID", "128"))
RESUME = os.environ.get("RESUME", "")
SCRATCH = os.environ.get("SCRATCH", "0") == "1"
STEPS = int(os.environ.get("STEPS", "4000"))
BATCH = int(os.environ.get("BATCH", "192"))
NPRIME = int(os.environ.get("NPRIME", "48"))
SEED = int(os.environ.get("SEED", "51"))
LR = float(os.environ.get("LR", "5e-4"))
OUT = pathlib.Path(os.environ.get("OUT", f"/tmp/joint_overfit_h{HID}"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
_MASK32 = (1 << 32) - 1
_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _to_bits_small(vals, width):
    sh = torch.arange(width - 1, -1, -1, device=vals.device)
    return (vals[:, None] >> sh[None, :]) & 1


def to_bits(ints, dev, width=L):
    nl = (width + 31) // 32
    cols = [_to_bits_small(torch.tensor([(v >> (32 * k)) & _MASK32 for v in ints],
            dtype=torch.int64, device=dev), 32) for k in range(nl - 1, -1, -1)]
    b = torch.cat(cols, dim=1)
    return b[:, nl * 32 - width:] if width < nl * 32 else b


class Cell(nn.Module):
    def __init__(self, dmodel=96, hidden=128):
        super().__init__()
        self.in_proj = nn.Linear(3, dmodel); self.d_emb = nn.Embedding(2, dmodel)
        self.gru = nn.GRU(dmodel, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, feat, d):
        x = self.in_proj(feat) + self.d_emb(d)[:, None, :]
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def is_prime(n, rng):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in list(_SMALL) + ([rng.randrange(2, n - 1) for _ in range(8)] if n >= (1 << 81) else []):
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


def top_primes(rng, n):
    """Primes with bit_length == L (so s = 2^(L-1) is a valid state < p)."""
    out = []
    while len(out) < n:
        c = rng.randint((1 << (L - 1)) + 1, (1 << L) - 1)
        if is_prime(c, rng):
            out.append(c)
    return out


def boundary_state(p, rng):
    """result-near-boundary: 2s+dx = kp+eps, small eps (the original sampler)."""
    for _ in range(20):
        k = rng.choice([0, 1, 1, 2]); eps = rng.choice([0, 0, 1, -1, 2, -2, rng.randrange(-64, 65)])
        x = 1 if rng.random() < 0.5 else rng.randrange(0, p); d = rng.randrange(0, 2)
        num = k * p + eps - d * x
        if num >= 0 and num % 2 == 0:
            return (num // 2) % p, x, d
    return rng.randrange(0, p), 1, rng.randrange(0, 2)


def maxborrow_state(p, rng):
    """max-INPUT-magnitude: s at/near 2^(L-1) or just below p -> single wrap, long borrow.
    x=1 (as in reduce), d in {0,1}. These are the deployed failure states."""
    half = 1 << (L - 1)
    kind = rng.random()
    if kind < 0.5:
        s = half - rng.randrange(0, 1 << 8)      # s = 2^(L-1) - small (incl. exactly 2^(L-1))
        if rng.random() < 0.25:
            s = half
    else:
        s = p - 1 - rng.randrange(0, max(1, p >> 12))  # near-p states (also long borrow)
    s %= p
    return s, 1, rng.randrange(0, 2)


def build_corpus(pool, rng, n):
    """Mixed corpus. Returns list of (s,x,p,d,type)."""
    rows = []
    n_rand = int(0.30 * n); n_bnd = int(0.30 * n); n_mb = n - n_rand - n_bnd
    for _ in range(n_rand):
        p = rng.choice(pool); s = rng.randrange(0, p)
        x = 1 if rng.random() < 0.5 else (0 if rng.random() < 0.2 else rng.randrange(0, p))
        rows.append((s, x, p, rng.randrange(0, 2), "random"))
    for _ in range(n_bnd):
        p = rng.choice(pool); s, x, d = boundary_state(p, rng)
        rows.append((s, x, p, d, "boundary"))
    for _ in range(n_mb):
        p = rng.choice(pool); s, x, d = maxborrow_state(p, rng)
        rows.append((s, x, p, d, "maxborrow"))
    rng.shuffle(rows)
    return rows


def make_batch(rows, idxs, dev):
    sb = [rows[i][0] for i in idxs]; xb = [rows[i][1] for i in idxs]; pb = [rows[i][2] for i in idxs]
    db = [rows[i][3] for i in idxs]
    yb = [(2 * rows[i][0] + rows[i][3] * rows[i][1]) % rows[i][2] for i in idxs]
    feat = torch.stack([to_bits(sb, dev).float(), to_bits(xb, dev).float(), to_bits(pb, dev).float()], dim=-1)
    return feat, torch.tensor(db, dtype=torch.long, device=dev), to_bits(yb, dev).float()


@torch.no_grad()
def eval_by_type(model, rows, dev):
    by, tot = {}, {}
    for i0 in range(0, len(rows), 256):
        idxs = list(range(i0, min(i0 + 256, len(rows))))
        feat, d, y = make_batch(rows, idxs, dev)
        pred = (torch.sigmoid(model(feat, d)) > 0.5).float()
        exact = (pred == y).all(dim=1)
        for j, i in enumerate(idxs):
            t = rows[i][4]; tot[t] = tot.get(t, 0) + 1; by[t] = by.get(t, 0) + int(exact[j].item())
    return {t: by[t] / tot[t] for t in tot}, sum(by.values()) / sum(tot.values())


def main():
    torch.manual_seed(SEED); rng = random.Random(SEED)
    model = Cell(hidden=HID).to(DEVICE)
    init = "scratch"
    if not SCRATCH and HID == 128 and RESUME:
        sd = torch.load(RESUME, map_location=DEVICE, weights_only=True)["state_dict"]
        model.load_state_dict(sd); init = "warm(v8)"
    print(f"=== joint capacity overfit === HID={HID} init={init} L={L} steps={STEPS} "
          f"batch={BATCH} nprime={NPRIME} device={DEVICE} params={sum(p.numel() for p in model.parameters())}",
          flush=True)
    t = time.time(); pool = top_primes(rng, NPRIME)
    train = build_corpus(pool, rng, int(os.environ.get("CORPUS", "12000")))
    held = build_corpus(pool, rng, 3000)
    print(f"pool={len(pool)} (all bit_length={L}) train={len(train)} held={len(held)} built in {time.time()-t:.0f}s", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = max(1, int(0.05 * STEPS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: (st + 1) / warm if st < warm
        else 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * (st - warm) / max(1, STEPS - warm))))
    bce = nn.BCEWithLogitsLoss(); hist = []; t0 = time.time()
    for step in range(STEPS):
        idxs = [rng.randrange(len(train)) for _ in range(BATCH)]
        feat, d, y = make_batch(train, idxs, DEVICE)
        loss = bce(model(feat, d), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % max(1, STEPS // 8) == 0 or step + 1 == STEPS:
            model.eval()
            tr_by, tr_all = eval_by_type(model, train[:1500], DEVICE)
            he_by, he_all = eval_by_type(model, held, DEVICE)
            model.train()
            row = {"step": step + 1, "loss": float(loss.detach().cpu()),
                   "train_all": tr_all, "held_all": he_all, "held_by": he_by, "train_by": tr_by,
                   "elapsed_s": round(time.time() - t0, 1)}
            hist.append(row); (OUT / "results.json").write_text(json.dumps(hist, indent=2))
            print(f"step {step+1:>5} loss {row['loss']:.5f} train {tr_all:.4f} held {he_all:.4f} | "
                  + " ".join(f"{k}:{v:.3f}" for k, v in sorted(he_by.items())) + f"  [{row['elapsed_s']}s]", flush=True)
    (OUT / "DONE").write_text("done\n")
    print(f"=== done HID={HID} init={init} === held maxborrow={hist[-1]['held_by'].get('maxborrow'):.4f} "
          f"held_all={hist[-1]['held_all']:.4f}", flush=True)


if __name__ == "__main__":
    main()
