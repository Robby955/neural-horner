# MiniNeuralHorner horizon extension

This experiment asks whether the 126,603-parameter `L=2048` model develops a
joint repair after the original 60,000-update limit. It continues the banked
update-60,000 training state through update 120,000. The architecture, training
distribution, seed, optimizer moments, data stream, evaluation cases, and
selection gate remain fixed.

The parent run did not fail uniformly. Its update-60,000 fixed-width screen
scored 64/64 on tiers 6 through 9 and 63/64 on tier 10. Dynamic-width scores
were 43/64, 55/64, 49/64, 54/64, and 63/64 on tiers 6 through 10. The extension
tests whether more updates at the terminal learning rate can recover both
width regimes at one checkpoint.

## What is restored

The parent checkpoint contains the model, all 21 AdamW state entries, the
scheduler counters, the Python training-data RNG, the Torch CPU RNG, and the
CUDA RNG state. The resume path must validate the frozen receipt and checkpoint
hashes before creating an output checkpoint. It then restores every saved state
and verifies the parent screen before update 60,001.

The learning rate stays at `4.5e-5`. The extension reconstructs the original
cosine schedule from its 60,000-update definition and clamps it at the floor.
Changing the cosine horizon to 120,000 would raise the learning rate and would
be a different experiment.

## Run

Use the same recorded H100 software and runtime configuration as the parent
run. Start from a reviewed, clean commit and point these variables at the
copied parent artifacts:

```bash
V3_PARENT_RECEIPT=research/v02/runs/scale_l2048_b127_b310c5e/B127/receipt.json
V3_PARENT_CHECKPOINT=research/v02/runs/scale_l2048_b127_b310c5e/B127/weights_step60000.pt
V3_RUN_ROOT=research/v03/runs/horizon_l2048_b127_60k_120k

mkdir -p "$V3_RUN_ROOT"
set -o pipefail
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u research/v02/train_scale.py \
  --config research/v03/configs/horizon_l2048_b127_60k_120k.json \
  --arm B127 \
  --device cuda \
  --resume-receipt "$V3_PARENT_RECEIPT" \
  --resume "$V3_PARENT_CHECKPOINT" \
  --out "$V3_RUN_ROOT/B127" 2>&1 | tee "$V3_RUN_ROOT/console.log"
```

The first checkpoint that passes the declared screen, confirmation, and
small-prime gates is selected and stops training. If none passes, update
120,000 is retained as a failed-gate endpoint. A failed gate is an experimental
result, not a runtime failure.

## Interpretation

This is an exact-state resume followed by a new schedule extension. It is not
a byte-for-byte replay of a hypothetical uninterrupted 120,000-update run. The
original schedule ended at update 60,000, no uninterrupted reference exists,
and the original container image digest was not recorded. Any result therefore
applies to this hash-bound, single-seed horizon extension.

See [PROTOCOL.md](PROTOCOL.md) for the frozen gate and reporting rules.
