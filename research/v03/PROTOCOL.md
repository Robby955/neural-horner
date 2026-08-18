# MiniNeuralHorner v0.3 horizon-extension protocol

## Research question

Can the 126,603-parameter MiniNeuralHorner cell reach the full fixed-and-dynamic
retention gate when its banked `L=2048` training state is continued from update
60,000 to update 120,000 at the terminal learning rate?

This is a delayed-generalization test. It does not change the architecture,
training distribution, seed, evaluation cases, or gate in response to the
observed trajectory.

## Frozen parent

The only permitted parent is `nh02-scale-l2048-b127`, arm `B127`, update 60,000.
The config binds:

- receipt SHA-256
  `77ce649080558c7c4154599461da33d3d482516bbd1c2ecc838161ea29592705`;
- checkpoint SHA-256
  `61c7077a82cbc53bf545e96ad131712ce1a53c24943579b34a2599a9572e5051`;
- parent source commit
  `b310c5e9e7e0e096f613e5cf6bbfa7b2b247281b`;
- parent trainer SHA-256
  `83463e0882b26bd056fd3a542f4824d28778b4f76088ca2d230c2d70d2cf957c`;
- parent config SHA-256
  `3108ecfa005e8bf96bdda61f2a0379ffd5fe045ac4cf3a9f88e469d3fe0aebfd`;
- archived parent provenance SHA-256
  `2d84692b49a3ebdc7550ff61606cb999dd27c2fbd7bcaa06f0e1da812be1b4f0`.

The parent receipt must have status `completed_failed_gate`, no selected
checkpoint, a terminal history row at update 60,000, and exactly one matching
checkpoint entry for that update. The checkpoint and receipt must agree on the
experiment, arm, architecture, parameter count, source identity, config, and
seed set.

## Resume contract

Before training, the runner must validate and restore:

1. all model tensors by strict name, shape, dtype, and signature;
2. the complete AdamW state, including 21 parameter-state entries at optimizer
   step 60,000;
3. the LambdaLR state at epoch 60,000 and scheduler step count 60,001;
4. the Python training-data RNG state;
5. the Torch CPU RNG state;
6. the single saved CUDA RNG state.

The recorded parent environment has canonical SHA-256
`bf27df9525c455210c6ce1703e5f3adb9b268f58f51bc6b3476174f3e2a288be`.
An exact-state run must fail closed if the recorded Torch, CUDA, cuDNN, driver,
device, or deterministic runtime settings differ. The current run binds its own
clean Git source separately from the archived parent source.

After restoration, the runner repeats the update-60,000 screen and requires an
exact match with the parent receipt, including scores, case manifests, sequence
widths, and recorded failures. It then restores the saved RNG states again.
Training starts at update 60,001. No child checkpoint is written for update
60,000.

## Provider interruption recovery

A provider exit may interrupt this fixed experiment after a completed child
evaluation. Recovery uses the last step for which `receipt.json` contains both
a history row and a hash-verified checkpoint entry. An unreferenced checkpoint
or temporary file is not accepted as progress. After the recorded boundary is
recomputed without any writes, such files are retained under
`interrupted_uncommitted/`.

The recovery runner requires the SHA-256 of the authoritative interrupted
receipt as an operator pin:

```bash
V3_RUN=research/v03/runs/horizon_l2048_b127_60k_120k/B127
V3_RECEIPT_SHA=$(shasum -a 256 "$V3_RUN/receipt.json" | awk '{print $1}')

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -u \
  research/v03/recover_interrupted.py \
  --config research/v03/configs/horizon_l2048_b127_60k_120k.json \
  --arm B127 \
  --device cuda \
  --expected-receipt-sha256 "$V3_RECEIPT_SHA" \
  --out "$V3_RUN"
```

Before modifying the receipt, recovery validates the exact experiment config,
original source bytes, environment, ordered evaluation history, every recorded
checkpoint hash, optimizer and scheduler counters, floor learning rate, and
Python, Torch, and CUDA random states. It then recomputes the latest screen and
requires exact structured equality with the latest receipt row. The original
receipt is copied to a hash-named backup before recovery provenance is added.
New checkpoints and receipts use temporary files followed by atomic rename.

The receipt update that records an evaluation and the later terminal receipt
are separate atomic operations. Recovery therefore also accepts either valid
terminal edge: a running receipt whose last recorded checkpoint has already
passed confirmation, or a running receipt that has reached update 120,000
without a selection. It validates the stored gate from its rollout and
small-prime evidence, then finalizes without another training update. If the
terminal receipt was written but `SELECTED.json` or the status marker was not,
recovery verifies the terminal receipt and repairs only the missing artifact.
Repeated repair is idempotent. An artifact that disagrees with the receipt is
rejected.

The original `started_at` and scientific source identity remain unchanged.
Each resumed-training or pending-terminal finalization records its receipt
input hash, checkpoint hash, recovery timestamp, executor commit and source
hashes, environment hash, boundary hash, and terminal status. Repairing a
missing auxiliary artifact does not rewrite an already terminal receipt. A
provider exit is classified as an operational interruption, not as a failed
gate.

## Schedule

The parent cosine schedule is reconstructed using its original 60,000-update
horizon and 1,500-update warmup. Its progress term is capped at one after the
parent horizon. The resulting learning rate is fixed at

```text
0.0015 * 0.03 = 0.000045
```

through update 120,000. The optimizer step precedes the scheduler step, as in
the parent run. Recomputing the cosine schedule with a 120,000-update horizon is
not permitted because it would raise the learning rate after resume.

## Evaluation and selection

The screen runs every 3,000 updates on tiers 6 through 10 in both fixed and
dynamic width modes. Every cell must score 64/64. A passing screen triggers the
frozen confirmation gate:

- tiers 6 through 9: 256/256 in both width modes;
- tier 10: at least 255/256 in both width modes;
- exhaustive transition checks for primes below 64: exact in both width modes;
- all model parameters: finite.

The first confirmed checkpoint is selected and training stops. Confirmation
thresholds distinguish a passing frontier checkpoint from an exact finite
rollout result; a 255/256 tier-10 cell is not reported as exact.

If no checkpoint confirms, training stops at update 120,000 with status
`completed_failed_gate`. The receipt must retain every screen, checkpoint hash,
the resume lineage, restoration checks, boundary verification, runtime
identity, and final gate.

## Claims and nonclaims

A passing checkpoint would show that this single training trajectory crossed
the declared finite gate after the original budget. A failed endpoint would
show that the same trajectory did not cross it by update 120,000 under the
clamped-floor extension.

Neither result proves a parameter lower bound, universal modular correctness,
or behavior under other seeds and training distributions. `Exact-state resume`
means that all saved training state was restored and validated. It does not
mean that the extension is a byte-for-byte replay of a never-stopped run: the
parent schedule had no definition beyond update 60,000, no uninterrupted
reference exists, and the original container digest was not recorded.
