# Preserved repair endpoints

These files preserve the exact local specialist endpoints used to build the two
interpolated research candidates. The original copies live under a disposable
Claude job directory and must not be treated as durable provenance.

| File | Role | SHA-256 |
| --- | --- | --- |
| `l2sp_lambda_1e-3_step1500.pt` | Endpoint for `direct_two_pass_l2sp_a0875` and its original-schedule control | `97db605caf949460b06bfb916d6a3ab0039547e6b79c89fbc48a1025a580e421` |
| `function_space_endpoint.pt` | Endpoint for `direct_two_pass_funcspace_a09375` | `2cfe7bb6b7e42a2f2e8863dd6e9122ca51c8c3435180db388e8b338a02c4497f` |

These are training-lineage artifacts, not submission files. Candidate packages
must continue to contain only their derived checkpoint and required inference
surface.
