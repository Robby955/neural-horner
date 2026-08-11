# Draft organizer runtime question

Do not send without Rob's approval.

The current Modular Arithmetic Challenge rules require the complete 1100-case
evaluation to finish within 300 seconds and recommend GPU batching. At official
commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`, however,
`docker/Dockerfile` installs the PyTorch CPU wheel and `docker/README.md` says the
image is CPU-only, exposes no GPU, and still needs a separate CUDA tag for GPU
evaluation. The published wrapper defaults to four CPUs, 8 GiB RAM, and a 2 GiB
`/tmp` tmpfs. Its 300-second timeout is cooperative between batches, while the
rules call model load and determinism separately bounded even though the pinned
pipeline currently only logs those durations.

Could the organizers confirm the final ranked-evaluation runtime before the
August 12 deadline?

1. Will ranked submissions run in the published four-CPU image, or in a separate
   GPU-enabled sandbox?
2. If GPU-enabled, what GPU class, CUDA/PyTorch versions, memory limit, batch
   constraints, permitted inference dtypes/autocast policy, and immutable
   container digest should contestants reproduce?
3. Is the 300-second budget measured only around `run_inference`, as current
   `rules/evaluation.md` states, and is there an external hard timeout inside a
   long `predict_digits_batch` call or only the repository's cooperative checks
   between batches?
4. What are the separately enforced load and determinism budgets, and are they
   enforced outside the currently published Python pipeline?
5. When will the final image and hardware contract be immutable for submission
   qualification?

Our model has distinct CPU and CUDA execution paths, so this changes both
feasibility and the exact numerical path that must be validated.
