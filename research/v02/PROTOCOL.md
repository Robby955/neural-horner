# NeuralHorner v0.2 pilot protocol

Status: **L128 PILOT COMPLETE; L256 BRIDGE COMPLETE; L512 SCALE BRIDGE COMPLETE; L1024 SCALE BRIDGE COMPLETE; L2048 SCALE/REPAIR COMPLETE, FAILED GATE**

Frozen on: 2026-08-16

P5 terminal outcome recorded on: 2026-08-17

Public research base: `307954213b23bc9540eb522adc73e792cd4bbdff`

Frozen P1 implementation: `658ef2ba5356196ea2d26455c5d24d7aae8b521a`

L256 run source: `6b6e1a59e2437c7384f68087e87dd4d34ea3c9e7`

## Question

Does hidden-width compression preserve exact rollout behavior when the original
two-layer bidirectional GRU, 96-dimensional input representation, hard binary
state, dynamic-width inference rule, and three-pass reduce/reduce/multiply
Horner schedule are held fixed?

This pilot can identify a smaller tested model that suffices under one declared
training run. It cannot identify a parameter lower bound. A failure can be an
optimization or budget failure rather than a representation failure.

## Architecture ladder

The primary ladder changes only GRU hidden width:

| arm | layers | directions | hidden | trainable parameters |
|---|---:|---:|---:|---:|
| B471 | 2 | 2 | 128 | 470,849 |
| B249 | 2 | 2 | 90 | 249,157 |
| B127 | 2 | 2 | 61 | 126,603 |
| B063 | 2 | 2 | 40 | 63,057 |
| B032 | 2 | 2 | 26 | 32,453 |

The parameter count is

\[
N(h)=24h^2+602h+577.
\]

Depth is a later, paired study. At approximately 125K parameters the intended
comparison is B127 against a one-layer bidirectional GRU with `hidden=103`
(125,001 parameters). It must not be mixed into the width pilot.

An aligned, same-pass unidirectional GRU is not an economical first ablation:
it has a representational obstruction. With `p=5`, `d=0`, and a common `x`, an
MSB-first model sees identical first symbols for `s=2` and `s=3`, while the first
output bits of `2s mod p` differ. The analogous LSB-first witness uses `s=1` and
`s=3`. A causal replacement needs delayed decoding, a second pass, or global
context and is therefore a different program.

## Phase P0: engineering smoke

`configs/smoke_l32.json` validates source/config hashing, deterministic batch
generation, checkpoint contents, receipt writing, evaluation, and clean exit.
Its scores are not scientific evidence and cannot advance an arm.

## Phase P1: local paired pilot

`configs/pilot_l128.json` fixes `L=128`, batch 256, 24,000 optimizer updates,
AdamW with peak learning rate `1.5e-3`, weight decay `0.01`, gradient clipping at
1.0, 2.5% warmup, cosine decay, and master seed 23. Every arm receives the same
ordered transition stream. Initialization and data sampling use recorded,
separate RNG streams derived from the master seed. PyTorch deterministic
algorithms are required; an unsupported deterministic kernel fails the run.

The order is fail-fast:

1. Train B471 from scratch. This is a new protocol control, not historical-run
   reproduction.
2. Continue only if B471 has finite loss/weights, passes every transition for
   all primes below 64, and scores 1.0 on every declared rollout tier.
3. Train B249 under the identical budget and data stream.
4. Continue to B127 only if B249 passes the same gates.
5. B063 and B032 remain unrun until the preceding arm passes.

Observed P1 outcome: B471, B249, and B127 completed the endpoint gate. B063
completed training and both exhaustive p<64 checks, but failed the endpoint
rollout gate. Under the frozen endpoint rule, B127 is therefore the smallest
passing parent and B032 remains unrun. These are L128 screening results, not
full-width model results.

Intermediate rollout screens use 64 cases per tier. The endpoint gate uses 512
cases per tier. Endpoint selection is fixed; an earlier lucky checkpoint cannot
replace it. Attempted and unrun arms must remain explicit in the ledger.

Training uses the declared fixed width `L=128`, matching the recovered driver.
Every rollout screen is evaluated twice: once at fixed `L=128`, and once with
the deployed batch-local dynamic width
`min(L, max(32, ceil(max_bitlen(p)/32)*32))`. The finite p<64 transition gate is
likewise run at both 128-bit and 32-bit sequence widths. Both contexts must pass;
this prevents zero-padding behavior from being mistaken for deployed behavior
in a bidirectional recurrent cell.

## Interpretation and promotion boundary

One local seed is a screening result only. A paper-level compression result
requires at least three independently initialized training seeds, progressive
width continuation through `L=2048`, the existing development batteries, the
official-source scorer, paired runtime measurements, and a fresh committed
structured evaluation that was not used for training or selection.

The existing 768-row structured battery is development data. Passing the
40,954 transitions for primes below 64 is finite computational validation, not
a Lean proof that the checkpoint realizes the transition universally. Parameter
reduction is not a latency, FLOP, memory-bandwidth, or energy claim until those
quantities are measured on the same runtime surface.

The submitted v8 checkpoint remains the only artifact qualified under the
original full-width scorer protocol. The selected B127/L512 weights were later
packaged as the public MiniNeuralHorner research artifact and evaluated in the
SAIR Playground after separate review. That package does not supersede v8 or
establish that the full P1--P5 promotion boundary was met.

## Phase P2: B127 cross-width bridge

`configs/bridge_l256_b127.json` freezes the first full-width continuation step.
It is bound to the completed B127 parent run, not merely to a compatible model
shape:

- parent receipt SHA-256:
  `fd6acc818adf73a4d3773b45bb9f7926de7551454ca4d4755827882fef9b5163`
- parent checkpoint SHA-256:
  `9a20eabd025060307697d718eea0dd415ee489155ff069dc726d1bd6d2acbae7`
- parent state shape/dtype signature SHA-256:
  `ee8b058770fbdef625a8e09e210f9f13ca5fc17ee5aa4d7bcf6bcebaf51e17fe`

The validator checks the parent experiment, arm, architecture, parameter count,
width, step, source hashes, receipt gate, checkpoint listing, checkpoint source,
and every state-dictionary key, shape, and dtype. The parent gate is the exact
512-case fixed/dynamic evaluation on tiers 4--6 plus both exhaustive p<64
transition contexts. A receipt and checkpoint that do not name one another by
the frozen checkpoint hash are rejected.

The L128 parent uses the legacy endpoint receipt, so its gate evidence is read
from the endpoint row. A future v2 parent is accepted only when its receipt
names a first-confirmed selected checkpoint, records that training stopped at
that checkpoint, and agrees on selected step, hash, path, reason, and gate. Its
gate evidence is read from the selected row's `confirmation` block, not the
64-case screen. Parent tiers, modes, confirmation count, and small-prime limit
come from the next stage's declared parent contract and must match the parent
receipt and checkpoint config exactly.

This is a warm start, not a resume. Only the model state is loaded. Optimizer,
scheduler, and RNG state are reset and the receipt records that fact. The child
runs on CUDA while the parent ran on MPS; the complete parent and child
environments remain recorded separately. `--resume` fails closed because no
cross-environment resume contract has been implemented.

The canonical CUDA run requires explicit `--device cuda` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the process environment before Python
starts. cuDNN benchmarking and both CUDA/cuDNN TF32 paths are disabled, cuDNN
deterministic mode is enabled, and float32 matmul precision is fixed at
`highest`. A mismatch aborts before artifact loading. The receipt records these
settings, the CUDA and cuDNN versions, NVIDIA driver version, and device
properties.

The canonical run also requires a clean Git worktree. This ensures the recorded
HEAD contains the trainer and config whose byte hashes appear in the receipt.

The bridge holds B127's architecture at 126,603 parameters and sets `L=256`,
tiers 4--7, batch 256, seed 23, peak learning rate `1.5e-3`, weight decay 0.01,
2.5% warmup, and an 80,000-update cap. Every 3,000 updates it runs a 64-case
screen in fixed and dynamic width modes. A screen that is exact triggers a
512-case confirmation in both modes and the two exhaustive p<64 checks. The
first checkpoint to pass all confirmation gates is selected, hash-bound, and
training stops. Later checkpoints cannot replace it. If no checkpoint confirms
by update 80,000, the bridge fails its declared gate.

Observed P2 outcome: the first confirmed checkpoint was update 21,000. It was
exact on all 512 cases for tiers 4--7 in both fixed and dynamic width modes and
exact on both exhaustive p<64 contexts. The selected checkpoint has SHA-256
`ab90103e630a26e49f1021fd06e9c63ec7ae0998b411651904b2b94497f5f1c1`.
The completed receipt has SHA-256
`c49f66783c62a05847c9e5461ffd6b7362e338a2800f28d0a50f2f3784b58158`.
This is a local single-seed bridge result, not a promoted model result.

## Phase P3: B127 L512 scale bridge

`configs/scale_l512_b127.json` is bound to the P2 selected receipt and
checkpoint. The parent contract names experiment `nh02-bridge-l256-b127`, arm
B127, width 256, update 21,000, 126,603 parameters, the architecture and state
shape/dtype signature, all source hashes, and the exact P2 gate. A different
receipt or checkpoint is rejected before the output directory is created.

P3 keeps the B127 architecture fixed and sets `L=512`, tiers 6--8, batch 192,
seed 23, peak learning rate `1.5e-3`, weight decay 0.01, 2.5% warmup, and a
50,000-update cap. Evaluation occurs every 3,000 updates in both fixed and
dynamic width modes.

The screen is exact: every tier and mode must score 64/64. Only then is the
256-case confirmation run. The confirmation thresholds are:

| tier | fixed | dynamic |
|---|---:|---:|
| 6 | 256/256 | 256/256 |
| 7 | at least 255/256 | at least 255/256 |
| 8 | at least 255/256 | at least 255/256 |

Both exhaustive p<64 transition contexts must be exact and all parameters must
be finite. The first checkpoint to meet this declared scale gate is selected,
hash-bound, and stops training. The receipt records `rollout_threshold_passed`
and `rollout_exact` separately, so a 255/256 tier cannot be reported as exact.

P3 tests whether the compressed cell can transfer to a larger sequence width.
It is deliberately an approximate bridge, not an exact L512 repair result. A
selected P3 checkpoint may seed a later, separately frozen DAgger/repair phase,
but it does not satisfy that later phase's exact gate by contract.

Observed P3 outcome: update 48,000 was the first checkpoint to pass the screen
and confirmation gate. It scored 256/256 on tiers 6--8 in both fixed and
dynamic modes and passed both exhaustive p<64 contexts. The selected checkpoint
has SHA-256
`970165b1af9e1518b83529ee8280521d9e3baece67d79efeee020e61c31f8ec7`.
The completed receipt has SHA-256
`abc5700038342725c443e200ef9bb9efd912ab4350f390272feaaaa2d392705a`.
This is a local single-seed bridge result. It has not been promoted or
published.

## Phase P4: B127 L1024 scale bridge

`configs/scale_l1024_b127.json` is bound to the P3 selected receipt and
checkpoint at update 48,000. It also binds the P3 source commit, trainer,
configuration, provenance file, architecture, parameter count, and state
shape/dtype signature. The parent gate remains the declared P3 contract:
256/256 on tier 6 and at least 255/256 on tiers 7 and 8 in both modes, plus
both exhaustive p<64 contexts. The observed P3 checkpoint happened to be exact
on all six rollout cells, but that does not change the gate under which it was
selected.

P4 holds the architecture at 126,603 parameters and sets `L=1024`, tiers 6--9,
batch 96, seed 23, peak learning rate `1.5e-3`, weight decay 0.01, 2.5% warmup,
and a 60,000-update cap. Halving the L512 batch keeps `width * batch_size` fixed
at 98,304 bit positions per optimizer update. Evaluation uses batch 32 for the
same reason.

The unchanged warm-start weights are evaluated at child update 0, before a
training batch or optimizer step. Both fixed and dynamic modes must score 64/64
on tiers 6--9. Only an exact screen triggers the 256-case confirmation. Tiers
6--8 must score 256/256 in both modes; tier 9 must score at least 255/256 in
both modes. Both exhaustive p<64 contexts must be exact and all parameters must
be finite. A passing update 0 produces a new L1024 checkpoint and receipt whose
model state is unchanged from the parent. If it fails, the same gate runs every
3,000 training updates. The first checkpoint to meet the gate is selected,
hash-bound, and stops training.

Retaining tiers 6 and 7 in the child gate is necessary. P3 contained checkpoints
where tier 8 passed its 64-case screen while tier 7 dynamic did not. Tier-8
success and the p<64 transition check therefore do not establish retained
tier-7 rollout behavior.

P4 was selected at update 0 without an optimizer step. It scored 256/256 on
tiers 6--9 in both fixed and dynamic modes and passed both exhaustive p<64
contexts. This is stronger than its declared threshold, which allowed one miss
on tier 9. The selected checkpoint has SHA-256
`d296b711bb6a7faaa1dd81e05478cfa75f11071c42a8c36fbf60e758ee7eb407`.
The completed receipt has SHA-256
`80948748a41183a809faf282600cc0f8343691b6cdb4bebaac4f1f468df95651`.
The model tensors are unchanged from P3. This remains a local single-seed
bridge result, not a promoted model result.

## Phase P5: B127 L2048 full-width scale/repair bridge

`configs/scale_l2048_b127.json` is bound to the P4 update-0 receipt and
checkpoint. The parent contract retains P4's declared 256/256/256/255
thresholds rather than rewriting them after seeing the stronger exact result.
The receipt and checkpoint hashes separately bind the actual update-0 parent,
which was exact on all eight confirmation cells and both p<64 contexts. The
standalone `SELECTED.json` is also bound by SHA-256
`53addcb1a400952cf5e52f6b8c4e0fd129ede9c1471b82a026c62dbd7783d1c7`.

P5 holds the architecture at 126,603 parameters and sets `L=2048`, tiers
6--10, batch 48, seed 23, peak learning rate `1.5e-3`, weight decay 0.01, 2.5%
warmup, and a 60,000-update cap. The training product remains
`width * batch_size = 98,304`. Rollout evaluation uses batch 64 because it has
no backward graph and the same model already completed a 64-case L2048
diagnostic at that batch size on the target H100.

The unchanged warm start is screened at update 0. Every tier from 6 through 10
must score 64/64 in fixed and dynamic modes. A passing screen triggers a
256-case confirmation. Tiers 6--9 must score 256/256 in both modes; tier 10
must score at least 255/256 in both modes. Both exhaustive p<64 contexts must
be exact and all parameters must be finite. If update 0 fails, the same gate
runs every 3,000 updates. The first passing checkpoint is selected and stops
training.

All retained tiers stay in both the screen and confirmation. This costs more
than checking only tiers 9 and 10, but it prevents a wider checkpoint from
silently trading away behavior established by P4. The 64-case screens are
exact. The confirmation gate is exact on retained tiers and approximate only
on the new tier-10 frontier. A 255/256 tier-10 result is a passing scale/repair
bridge, not an exact full-width result; the receipt records threshold pass and
exactness separately.

Before this stage was frozen, the unchanged P3/P4 tensors were evaluated on a
paired 64-case L2048 diagnostic. Tier 9 scored 64/64 in both modes. Tier 10
scored 63/64 in both modes, with the same case failing and output bits 600--604
wrong. Receipt SHA-256:
`37732f2b8ac5a2cd4c878829b4f745a3749d19387885196ec3f03410510982fc`.
This predeclared failure meant update 0 was expected to fail the exact screen;
it did not authorize changing the gate after the run began. The canonical P5
run was launched from source commit
`b310c5e9e7e0e096f613e5cf6bbfa7b2b247281b`. Its update-0 screen reproduced
64/64 on tiers 6--9 and 63/64 on tier 10 in both fixed and dynamic modes.
At update 3,000, tier 10 reached 64/64 in both modes, while dynamic tier 7
regressed to 61/64 and dynamic tier 8 to 63/64. The retention gate rejected
the checkpoint before confirmation, even though all fixed-width cells were
64/64.

P5 completed the 60,000-update cap with status `completed_failed_gate`. None of
the 21 screens passed, so the confirmation and exhaustive p<64 checks were
not run. The final screen scored 64/64, 64/64, 64/64, 64/64, and 63/64 on
fixed tiers 6--10, and 43/64, 55/64, 49/64, 54/64, and 63/64 in dynamic mode.
The receipt's `small_prime_exact: false` records the failed terminal gate; it
must not be read as an observed small-prime failure because that evaluation did
not run.

No checkpoint was selected or promoted. The complete receipt has SHA-256
`77ce649080558c7c4154599461da33d3d482516bbd1c2ecc838161ea29592705`;
the console log has SHA-256
`370fb9641e9b933b61a846048ff71883864de2c2b6ef6e48e44cc8cab0b91528`.
All screen vectors and artifact bindings are preserved in the tracked
[`evidence/l2048_scale_repair_b310c5e_terminal.json`](evidence/l2048_scale_repair_b310c5e_terminal.json).
This one-seed result shows a finite repair/retention tradeoff across sequence
contexts. It does not establish that 126,603 parameters are insufficient or
that delayed generalization cannot occur beyond the declared budget.
