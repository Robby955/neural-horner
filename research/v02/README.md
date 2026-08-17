# NeuralHorner v0.2

This directory contains the MiniNeuralHorner compression and width-transfer
study. With the original two-layer
bidirectional GRU, hard binary state, and reduce/reduce/multiply Horner schedule
held fixed, how far can hidden width be reduced before exact rollout behavior
fails under a matched training budget?

Start with [`PROTOCOL.md`](PROTOCOL.md). The frozen pilot configurations live in
[`configs/`](configs/). Generated checkpoints and receipts go under `runs/`,
which is ignored because even unsuccessful runs can be large. Completed results
are summarized here only after their receipts and claim boundaries are checked.

## Provenance boundary

The production training scripts and byte-identical v8 checkpoint survive in the
original private training archive. The successful t10b command was recovered
with its v7 warm start, default seed, and optimizer schedule. The v8 checkpoint
does not contain optimizer, scheduler, Python RNG, Torch RNG, or CUDA RNG state,
and deterministic CUDA execution was not enforced. The recovered recipe does
not guarantee a byte-for-byte reproduction.

The v0.2 trainer derives from that production driver while adding explicit
architecture controls and fail-closed receipts. Its runs are new experiments,
not bit-exact replays of historical v8 training.

The public [`source_provenance.json`](source_provenance.json) removes
machine-local repository and transcript locators. Frozen receipts continue to
bind the original record by SHA-256.

## Commands

Validate the trainer and architecture counts:

```bash
python3 -m pytest tests/test_v02_training.py -q
```

Run the engineering smoke test:

```bash
python3 research/v02/train_scale.py \
  --config research/v02/configs/smoke_l32.json \
  --arm B471 \
  --out research/v02/runs/smoke_l32/B471
```

Run the frozen L=128 control only after the smoke test passes:

```bash
python3 research/v02/train_scale.py \
  --config research/v02/configs/pilot_l128.json \
  --arm B471 \
  --out research/v02/runs/pilot_l128/B471
```

The pilot ran sequentially: B471 passed before B249 was started, and B249 passed
before B127 was started.

## L256 bridge

The L256 bridge continued the exact B127 update-24,000 checkpoint from `L=128`
to `L=256`. It selected update 21,000 after an exact 512-case confirmation on
tiers 4--7 in fixed and dynamic width modes, plus both exhaustive p<64 checks.
The selected checkpoint and full receipt remain in the banked local run archive.

The parent files are not tracked in Git. To rerun this stage with the banked
artifacts, set these two variables to the copied L128 receipt and checkpoint
files, then run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 research/v02/train_scale.py \
  --config research/v02/configs/bridge_l256_b127.json \
  --arm B127 \
  --device cuda \
  --parent-receipt "$PARENT_RECEIPT" \
  --warm-start "$PARENT_CHECKPOINT" \
  --out research/v02/runs/bridge_l256_b127/B127
```

The command verifies the frozen parent receipt and checkpoint before creating
the output directory. It loads model weights only. Resume is deliberately
unsupported. The run stops at the first checkpoint that passes the declared
512-case fixed/dynamic confirmation and both exhaustive p<64 checks, or fails
at the 80,000-update cap. It also refuses a dirty Git worktree, so commit the
reviewed trainer and config before launching the canonical run.

## L512 scale bridge

`configs/scale_l512_b127.json` continues only from the selected L256 update
21,000 artifact. Its config binds the parent receipt, checkpoint, source files,
state shape and dtype signature, architecture, and exact L256 gate.

Set the variables to the L256 files under
`runs/bridge_l256_b127_6b6e1a5/B127`, then run only from a reviewed, clean
commit:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 research/v02/train_scale.py \
  --config research/v02/configs/scale_l512_b127.json \
  --arm B127 \
  --device cuda \
  --parent-receipt "$L256_RECEIPT" \
  --warm-start "$L256_CHECKPOINT" \
  --out research/v02/runs/scale_l512_b127/B127
```

Every 3,000 updates, the L512 stage requires an exact 64/64 screen for tiers
6--8 in both width modes. A passing screen triggers a 256-case confirmation.
Tier 6 must score 256/256 in each mode; tiers 7 and 8 must each score at least
255/256 in each mode. Both exhaustive p<64 contexts must remain exact. The
first checkpoint to meet those requirements is selected and training stops.

This is an approximate scale-transfer gate. Its receipt reports threshold pass
and exactness separately. It is not the later exact DAgger/repair gate and must
not be described as an exact L512 result when either tier 7 or tier 8 has a
miss.

The selected L512 checkpoint was update 48,000. Although its declared gate
allowed one miss on tiers 7 and 8, this checkpoint scored 256/256 on every
tier and mode in the confirmation and remained exact in both p<64 checks. This
is a single-seed bridge result.

## L1024 scale bridge

`configs/scale_l1024_b127.json` continues only from the selected L512 update
48,000 artifact. The parent contract preserves the gate that the L512 run was
actually selected under, including its 256/255/255 thresholds. The child does
not retroactively replace that declared contract with the selected
checkpoint's stronger observed scores.

Set the variables to the L512 files under
`runs/scale_l512_b127_7afdbaa/B127`, then run only from a reviewed, clean
commit:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 research/v02/train_scale.py \
  --config research/v02/configs/scale_l1024_b127.json \
  --arm B127 \
  --device cuda \
  --parent-receipt "$L512_RECEIPT" \
  --warm-start "$L512_CHECKPOINT" \
  --out research/v02/runs/scale_l1024_b127/B127
```

Before training, the L1024 stage evaluates the unchanged warm-start weights at
update 0. It requires an exact 64/64 screen on tiers 6--9 in fixed and dynamic
modes. A passing screen triggers a 256-case confirmation. Tiers 6--8 must be
exact; tier 9 may miss at most one case in each mode. Both p<64 contexts must
remain exact. If update 0 passes, a child checkpoint with unchanged model
weights and full L1024 provenance is selected without an optimizer step.

If update 0 fails, training begins with batch 96, which keeps
`width * batch_size` equal to the L512 stage. The same gate runs every 3,000
updates. The first checkpoint to pass is selected and training stops. Checking
tiers 6 and 7 prevents a wider checkpoint from being selected after regressing
on behavior already established at L512.

This is also an approximate scale-transfer gate. A 255/256 tier-9 result is not
an exact L1024 result, even if the run status is `completed_pass`. Exactness is
recorded separately.

The L1024 stage selected update 0 without training. Its confirmation was exact:
256/256 on tiers 6--9 in fixed and dynamic modes, plus both p<64 checks. The
model tensors are unchanged from the selected L512 checkpoint. This is a
single-seed result.

## Public Playground evidence

A completed SAIR Playground result was manually transcribed for the public
`TrickyRex/mini-neuralhorner-v02` artifact at exact revision
`d9d611833d340c72d90a97d995a94031b798cf7c`. Tiers 1--9 each scored 100/100
and tier 10 scored 85/100. The scored result is therefore 985/1000, or 98.5%;
the Playground displayed the rounded value 99% and frontier T9. Its visible
total was 1025/1100 because that total also includes the unscored T0 diagnostic,
which scored 40/100. The UI reported 156.602 seconds and a 520 KB artifact.

No public run URL or screenshot was archived, so this is a UI transcription
rather than an independently replayable Playground receipt. The
machine-readable record is
[`evidence/sair_playground_d9d611833d340c72d90a97d995a94031b798cf7c.json`](evidence/sair_playground_d9d611833d340c72d90a97d995a94031b798cf7c.json).
It remains bound to the exact tested revision even if the current Hugging Face
head differs after documentation-only corrections.

## L2048 full-width scale/repair bridge

`configs/scale_l2048_b127.json` continues only from the selected L1024 update-0
artifact. Its parent contract preserves the gate P4 was selected under, while
the frozen receipt and checkpoint hashes bind the stronger observed exact
result.

Set the variables to the L1024 files under
`runs/scale_l1024_b127_1e57afc/B127`, then run only from a reviewed, clean
commit:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 research/v02/train_scale.py \
  --config research/v02/configs/scale_l2048_b127.json \
  --arm B127 \
  --device cuda \
  --parent-receipt "$L1024_RECEIPT" \
  --warm-start "$L1024_CHECKPOINT" \
  --out research/v02/runs/scale_l2048_b127/B127
```

The stage first evaluates the unchanged model at update 0. The screen requires
64/64 on every tier from 6 through 10 in fixed and dynamic modes. A passing
screen triggers a 256-case confirmation. Retained tiers 6--9 must be exact;
tier 10 may miss at most one case in each mode. Both p<64 contexts must remain
exact. If update 0 fails, training begins with batch 48 and the gate runs every
3,000 updates through a 60,000-update cap.

The broader retained-tier gate is intentional. It makes confirmation slower,
but it prevents selection after a regression on an earlier tier. Evaluation
uses batch 64 because no backward graph is retained and that batch already ran
successfully at L2048 on the target H100.

The pre-run 64-case diagnostic scored 64/64 on tier 9 and 63/64 on tier 10 in
both modes. Fixed and dynamic inference failed on the same tier-10 case, with
five contiguous output bits wrong. The frozen update-0 screen was therefore
expected to reproduce a known failure. A passing 255/256 tier-10 confirmation
would remain approximate; only a receipt with 256/256 in every cell would be
exact on this finite gate.

The canonical P5 run was launched from the locally archived source commit
`b310c5e9e7e0e096f613e5cf6bbfa7b2b247281b`. Its update-0 screen reproduced
64/64 on tiers 6--9 and 63/64 on tier 10 in both fixed and dynamic modes.
At update 3,000, tier 10 reached 64/64 in both modes, but dynamic tier 7 fell
to 61/64 and dynamic tier 8 to 63/64. All five fixed-width cells remained
64/64. The retention gate rejected that checkpoint before confirmation.

The run completed its 60,000-update cap with status `completed_failed_gate`.
No checkpoint passed the exact screen, so the 256-case confirmation and p<64
checks were not run. At update 60,000, fixed tiers 6--10 scored
64/64, 64/64, 64/64, 64/64, and 63/64; the dynamic scores were 43/64, 55/64,
49/64, 54/64, and 63/64. No checkpoint satisfied the selection gate. The
existing public MiniNeuralHorner artifact is unchanged.

The compact [terminal evidence](evidence/l2048_scale_repair_b310c5e_terminal.json)
contains all 21 screen vectors and binds the local receipt, console log, and
failure marker by SHA-256. The full checkpoint directory is not versioned. This
single-seed, 60,000-update result does not establish a parameter lower bound or
evaluate training beyond the declared cap.
