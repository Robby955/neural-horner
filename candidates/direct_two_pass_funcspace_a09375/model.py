"""Two-pass NeuralHorner candidate for the Modular Arithmetic Challenge.

The trained cell models ``s' = (2*s + d*x) mod p``. This fixed schedule
canonically orders the operands, reduces the longer one with ``x = 1``, then
scans the original MSB-first bits of the shorter one with the learned residue as
``x``. For an exact cell, the second scan returns the modular product and removes
the original fixed-width third pass.

Control tokens always come from one preprocessed input.  Model output never
selects the next token.  State is re-quantized to bits after every transition.
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
    """Convert nonnegative Python integers to overflow-safe MSB-first bits."""
    limb_count = (width + 31) // 32
    columns = []
    for limb_index in range(limb_count - 1, -1, -1):
        limb = torch.tensor(
            [(value >> (32 * limb_index)) & _MASK32 for value in ints],
            dtype=torch.int64,
            device=dev,
        )
        columns.append(_to_bits_small(limb, 32))
    bits = torch.cat(columns, dim=1)
    start = limb_count * 32 - width
    return bits[:, start:] if start else bits


class Cell(nn.Module):
    def __init__(self, dmodel: int = 96, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(3, dmodel)
        self.d_emb = nn.Embedding(2, dmodel)
        self.gru = nn.GRU(
            dmodel,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, feat, d):
        projected = self.in_proj(feat) + self.d_emb(d)[:, None, :]
        hidden, _ = self.gru(projected)
        return self.head(hidden).squeeze(-1)


def _bits_of(value: int) -> list[int]:
    if value <= 0:
        return [0]
    bits = []
    while value > 0:
        bits.append(value & 1)
        value >>= 1
    bits.reverse()
    return bits


def _canonical_scan_reduce(
    a_bits: list[int], b_bits: list[int]
) -> tuple[list[int], list[int]]:
    """Return a swap-invariant ``(scan, reduce)`` operand ordering.

    The longer encoded operand is reduced with ``x = 1`` and the shorter is
    scanned with the learned residue. Equal-length operands use lexicographic
    order as a deterministic tie-breaker.
    """
    if (len(a_bits), a_bits) <= (len(b_bits), b_bits):
        return a_bits, b_bits
    return b_bits, a_bits


class DirectBitSerialReducer(ModularMultiplicationModel):
    def __init__(self) -> None:
        self.model: Cell | None = None
        self.device: torch.device | None = None
        self.L = 32

    def load(self, model_dir: str) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        checkpoint = torch.load(
            Path(model_dir) / "weights.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.L = int(checkpoint.get("L", 32))
        self.model = Cell(**checkpoint.get("config", {}))
        self.model.load_state_dict(checkpoint["state_dict"])
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
        if self.device is None:
            raise RuntimeError("load() must be called before prediction")

        # The challenge's largest operand regime is 2*L bits.  Reject anything
        # wider so the manifest and implemented support boundary stay identical.
        max_operand_bits = 2 * self.L
        outputs: list[list[int]] = [[0] for _ in inputs]
        groups: dict[int, list[tuple[int, list[int], list[int], int]]] = {}

        for output_index, (a_enc, b_enc, p_enc) in enumerate(inputs):
            p = int(p_enc)
            a_bits = list(a_enc)
            b_bits = list(b_enc)
            if (
                p < 2
                or p >= (1 << self.L)
                or len(a_bits) > max_operand_bits
                or len(b_bits) > max_operand_bits
            ):
                continue

            effective_width = min(
                self.L,
                max(32, ((p.bit_length() + 31) // 32) * 32),
            )
            groups.setdefault(effective_width, []).append(
                (output_index, a_bits, b_bits, p)
            )

        for effective_width, rows in groups.items():
            output_indices = [row[0] for row in rows]
            ordered = [
                _canonical_scan_reduce(row[1], row[2]) for row in rows
            ]
            scan_lists = [row[0] for row in ordered]
            reduce_lists = [row[1] for row in ordered]
            p_values = [row[3] for row in rows]
            p_bits = to_bits_limbs(
                p_values, self.device, effective_width
            ).float()
            one_bits = to_bits_limbs(
                [1] * len(rows), self.device, effective_width
            ).float()

            reduced_operand = self._scan_bits(
                reduce_lists,
                one_bits,
                p_bits,
                effective_width,
            )
            product = self._scan_bits(
                scan_lists,
                reduced_operand,
                p_bits,
                effective_width,
            )

            for output_index, row in zip(output_indices, product.long().tolist()):
                outputs[output_index] = [int(bit) for bit in row]

        return outputs

    def max_batch_size(self) -> int:
        return 256

    def _step(self, s_bits, x_bits, p_bits, d):
        if self.model is None:
            raise RuntimeError("load() must be called before prediction")
        features = torch.stack([s_bits, x_bits, p_bits], dim=-1)
        if self.device is not None and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(features, d)
            return (torch.sigmoid(logits.float()) > 0.5).float()
        return (torch.sigmoid(self.model(features, d)) > 0.5).float()

    def _scan_bits(self, bit_lists, x_bits, p_bits, effective_width):
        """Scan MSB-first bits while freezing rows during artificial padding."""
        row_count = len(bit_lists)
        scan_width = max(len(bits) for bits in bit_lists)
        padded = torch.zeros(
            (row_count, scan_width),
            dtype=torch.long,
            device=self.device,
        )
        start_positions = torch.empty(
            row_count,
            dtype=torch.long,
            device=self.device,
        )
        for row_index, bits in enumerate(bit_lists):
            start = scan_width - len(bits)
            start_positions[row_index] = start
            if bits:
                padded[row_index, start:] = torch.tensor(
                    bits,
                    dtype=torch.long,
                    device=self.device,
                )

        state = torch.zeros(
            (row_count, effective_width),
            device=self.device,
        )
        for position in range(scan_width):
            next_state = self._step(
                state,
                x_bits,
                p_bits,
                padded[:, position],
            )
            active = position >= start_positions
            state = torch.where(active[:, None], next_state, state)
        return state
