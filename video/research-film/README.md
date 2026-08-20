# NeuralHorner research film

Remotion source for a short, silent NeuralHorner research film. The film covers the modular-arithmetic problem, the fixed three-pass Horner program, MiniNeuralHorner compression, the completed Playground result, the terminal L2048 confirmation, and the Fermat/F11 failure boundary.

The rendered claims are fixed to completed evidence:

- MiniNeuralHorner has 126,603 trainable parameters.
- The completed Playground result was 100/100 on tiers 1–9 and 85/100 on tier 10.
- The 120K L2048 screen was 640/640; the larger confirmation was 2,548/2,560 and failed the declared gate.
- Both fixed and dynamic small-prime transition checks were 40,954/40,954.
- The public Mini checkpoint scored 119/128 on the original Fermat family; all nine observed failures were F11 rows.
- The historical v8 structured battery was 759/768. Its F11 trace is shown separately from the Mini output-only diagnostic.

The Playground values are owner-transcribed from the completed UI. No screenshot, public run ID, or independent Playground replay is archived.

## Compositions

- `NeuralHorner-Research-Film`: 2,830 frames, 1920×1080, 30 fps, 94.33 seconds.
- `NeuralHorner-Research-Loop`: 450 frames, 1920×1080, 30 fps, 15 seconds.
- `NeuralHorner-Research-Poster`: 1920×1080 still for repository and release links.
- Individual scene compositions for layout review in Remotion Studio.

## Install and inspect

```bash
npm install
npm run preflight
npm run lint
npm run compositions
npm run dev
```

The evidence preflight checks the exact aggregate counts, source identities, artifact revisions, and receipt hashes in `src/data/animation-data.json` before Studio, bundling, or rendering.

## Render

```bash
npm run render:poster
npm run render:film
```

The delivery render is H.264 at 1920×1080 with BT.709 color, `yuv420p`, CRF 18, a 60-frame GOP, and no audio stream. Outputs are written to the ignored `out/` directory.

## Evidence sources

The normalized evidence data references:

- the frozen L128 pilot receipts for B471, B249, B127, and B063;
- the completed Playground transcription for `TrickyRex/mini-neuralhorner-v02@d9d611833d340c72d90a97d995a94031b798cf7c`;
- the terminal 120K horizon receipt and checkpoint;
- the Mini Fermat-only diagnostic receipt;
- the historical v8 structured-battery and F11 trace receipts.

Full hashes and source paths live in `src/data/animation-data.json`. The film does not equate Mini and v8 failure trajectories, treat the 64-case screen as confirmation, or present the failed L2048 gate as a promoted checkpoint.

Public copies of the terminal L2048 receipt and Mini Fermat diagnostic are in
[`evidence/`](evidence/). The delivery render was produced from standalone
source commit `778ddb38bb6ccccb27d9f0d222720cf35faf2be8`; its exact command and output
hashes are recorded in
[`receipts/neuralhorner-research-film-v1.json`](receipts/neuralhorner-research-film-v1.json).
`receipts/RELEASE_SHA256SUMS` records the filenames and hashes used for the
GitHub Release assets.

The pilot and standalone-render commit hashes are immutable archive identities,
not links to commits in this repository. The terminal L2048 and Mini Fermat
receipts are published here verbatim. The older pilot receipts are represented
by their recorded SHA-256 identities because the originals contain
machine-local path fields.
