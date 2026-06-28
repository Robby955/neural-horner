"""Bit-serial learned reducer (general width) for the Modular Arithmetic Challenge.

Same design as bit-serial-v1/v2: one shared, p-conditioned transition cell that
learned s' = (2*s + d*x) mod p, applied in a fixed bit-serial Horner loop (reduce a,
reduce b, multiply). The arithmetic is in the trained cell; the loop only sequences
bits. Randomising the weights collapses accuracy to chance.

This version generalises the state width to L (read from the checkpoint), so it
covers tiers up to whatever L the weights were trained for. Bit extraction uses
32-bit limbs (`to_bits_limbs`) so a modulus p >= 2^63 never overflows an int64
tensor (needed at L >= 64). State is carried as bits between steps; the harness
decoder reconstructs the integer answer from the emitted base-2 digits.

Regime: primes p < 2^L and operands up to 4*L bits. Outside it the model abstains
and emits [0] -- the honest fallback.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from modchallenge.interface.base_model import ModularMultiplicationModel

_MASK32 = (1 << 32) - 1


def _to_bits_small(vals: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width - 1, -1, -1, device=vals.device)
    return (vals[:, None] >> shifts[None, :]) & 1


def to_bits_limbs(ints, dev, width: int) -> torch.Tensor:
    """List of python ints (< 2^width) -> (N, width) MSB-first bit tensor via 32-bit limbs.

    Overflow-safe for any width: no int64 tensor ever holds a value >= 2^32."""
    nl = (width + 31) // 32
    cols = []
    for k in range(nl - 1, -1, -1):  # most-significant limb first
        limb = torch.tensor([(v >> (32 * k)) & _MASK32 for v in ints],
                            dtype=torch.int64, device=dev)
        cols.append(_to_bits_small(limb, 32))
    bits = torch.cat(cols, dim=1)
    return bits[:, nl * 32 - width:] if width < nl * 32 else bits


class Cell(nn.Module):
    def __init__(self, dmodel: int = 96, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(3, dmodel)
        self.d_emb = nn.Embedding(2, dmodel)
        self.gru = nn.GRU(dmodel, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, feat, d):
        x = self.in_proj(feat) + self.d_emb(d)[:, None, :]
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def _bits_of(n: int) -> list[int]:
    if n <= 0:
        return [0]
    out: list[int] = []
    while n > 0:
        out.append(n & 1)
        n >>= 1
    out.reverse()
    return out


class BitSerialReducer(ModularMultiplicationModel):
    def __init__(self) -> None:
        self.model: Cell | None = None
        self.device: torch.device | None = None
        self.L = 32
        self._Leff = 32

    def load(self, model_dir: str) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        ckpt = torch.load(Path(model_dir) / "weights.pt", map_location=self.device, weights_only=True)
        self.L = int(ckpt.get("L", 32))
        self.model = Cell(**ckpt.get("config", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_a(self, a):
        return _bits_of(int(a))

    def preprocess_b(self, b):
        return _bits_of(int(b))

    def preprocess_p(self, p):
        return int(p)

    @torch.no_grad()
    def predict_digits(self, a_enc, b_enc, p_enc):
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    @torch.no_grad()
    def predict_digits_batch(self, inputs):
        L = self.L
        max_op = 4 * L
        out: list[list[int]] = [[0] for _ in inputs]
        idx, a_lists, b_lists, p_vals = [], [], [], []
        for i, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            a_bits = list(a_enc)
            b_bits = list(b_enc)
            if p < 2 or p >= (1 << L) or len(a_bits) > max_op or len(b_bits) > max_op:
                continue
            idx.append(i)
            a_lists.append(a_bits)
            b_lists.append(b_bits)
            p_vals.append(p)
        if not idx:
            return out
        dev = self.device
        maxp = max(int(p).bit_length() for p in p_vals)
        self._Leff = min(self.L, max(32, ((maxp + 31)//32)*32))
        p_bits = to_bits_limbs(p_vals, dev, self._Leff).float()
        ra = self._reduce(a_lists, p_bits, dev)
        rb = self._reduce(b_lists, p_bits, dev)
        prod = self._mul(ra, rb, p_bits)
        prod_list = prod.long().tolist()
        for j, i in enumerate(idx):
            out[i] = [int(x) for x in prod_list[j]]
        return out

    def max_batch_size(self) -> int:
        return 256

    def _step(self, s_bits, x_bits, p_bits, d):
        feat = torch.stack([s_bits, x_bits, p_bits], dim=-1)
        if self.device is not None and self.device.type == "cuda":
            # bf16 for the GRU (~2x at L=2048); threshold in fp32 so the discrete
            # decision is unchanged (logits are saturated, far from 0).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(feat, d)
            return (torch.sigmoid(logits.float()) > 0.5).float()
        return (torch.sigmoid(self.model(feat, d)) > 0.5).float()

    def _reduce(self, bit_lists, p_bits, dev):
        n = len(bit_lists)
        width = max(len(b) for b in bit_lists)
        padded = torch.zeros((n, width), dtype=torch.long, device=dev)
        for r, bl in enumerate(bit_lists):
            if bl:
                padded[r, width - len(bl):] = torch.tensor(bl, dtype=torch.long, device=dev)
        s_bits = torch.zeros((n, self._Leff), device=dev)
        x_bits = to_bits_limbs([1] * n, dev, self._Leff).float()
        for pos in range(width):
            s_bits = self._step(s_bits, x_bits, p_bits, padded[:, pos])
        return s_bits

    def _mul(self, ra_bits, rb_bits, p_bits):
        n = ra_bits.shape[0]
        s_bits = torch.zeros((n, self._Leff), device=ra_bits.device)
        rb_long = rb_bits.long()
        for k in range(self._Leff):
            s_bits = self._step(s_bits, ra_bits, p_bits, rb_long[:, k])
        return s_bits
