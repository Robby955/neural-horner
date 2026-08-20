import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dataPath = resolve(root, "src/data/animation-data.json");
const data = JSON.parse(readFileSync(dataPath, "utf8"));
const modeArg = process.argv.find((arg) => arg.startsWith("--mode="));
const mode = modeArg?.split("=")[1] ?? "current";
const sha256 = /^[a-f0-9]{64}$/;
const gitSha = /^[a-f0-9]{40}$/;

const fail = (message) => {
  throw new Error(`Animation data preflight failed: ${message}`);
};

const requireSha256 = (value, label) => {
  if (!sha256.test(value ?? "")) {
    fail(`${label} must be a full SHA-256`);
  }
};

const requireGitSha = (value, label) => {
  if (!gitSha.test(value ?? "")) {
    fail(`${label} must be a full Git SHA`);
  }
};

const validatePairedTiers = (
  label,
  evidence,
  expectedPerMode,
  expectedCorrect,
) => {
  const tiers = Object.keys(evidence)
    .map(Number)
    .sort((a, b) => a - b);
  if (tiers.join(",") !== "6,7,8,9,10") {
    fail(`${label} must contain tiers 6 through 10`);
  }
  const correct = Object.values(evidence).reduce(
    (sum, tier) => sum + tier.fixed + tier.dynamic,
    0,
  );
  if (
    Object.values(evidence).some((tier) => tier.total !== expectedPerMode) ||
    correct !== expectedCorrect
  ) {
    fail(`${label} counts do not match the declared aggregate`);
  }
};

if (mode !== "current") {
  fail(`unsupported mode ${mode}`);
}
if (data.schema !== "neuralhorner-research-film-data-v2") {
  fail("unexpected schema");
}
for (const [label, value] of Object.entries(data.sourceCommits)) {
  requireGitSha(value, `sourceCommits.${label}`);
}
if (data.recurrence.formula !== "s' = (2s + d x) mod p") {
  fail("recurrence formula changed");
}
if (data.recurrence.phases.join(",") !== "reduce_a,reduce_b,multiply") {
  fail("three-pass schedule changed");
}

if (data.pilot.width !== 128 || data.pilot.selectedArm !== "B127") {
  fail("pilot identity does not match the frozen selection");
}
const selected = data.pilot.arms.find((arm) => arm.id === "B127");
if (
  !selected ||
  selected.parameters !== 126603 ||
  selected.outcome !== "passed"
) {
  fail("B127 must be the 126,603-parameter passing pilot arm");
}
for (const arm of data.pilot.arms) {
  requireSha256(arm.receiptSha256, `${arm.id} receipt`);
  requireSha256(arm.checkpointSha256, `${arm.id} checkpoint`);
}
const b063 = data.pilot.arms.find((arm) => arm.id === "B063");
if (!b063 || b063.outcome !== "failed_endpoint") {
  fail("B063 must remain recorded as an endpoint failure");
}
if (
  data.pilot.b063.smallPrimeCasesPerWidth !== 40954 ||
  !data.pilot.b063.smallPrimeFixedExact ||
  !data.pilot.b063.smallPrimeDynamicExact
) {
  fail("B063 small-prime evidence is incomplete");
}

const hosted = data.hostedEvaluation;
requireGitSha(hosted.evaluatedRevision, "hosted evaluated revision");
requireSha256(hosted.evidenceSha256, "hosted evidence");
if (
  hosted.status !== "completed" ||
  hosted.tiers.length !== 10 ||
  hosted.tiers.some((tier, index) => tier.tier !== index + 1) ||
  hosted.tiers
    .slice(0, 9)
    .some((tier) => tier.correct !== 100 || tier.total !== 100) ||
  hosted.tiers[9].correct !== 85 ||
  hosted.tiers[9].total !== 100
) {
  fail("hosted per-tier result changed");
}
const hostedCorrect = hosted.tiers.reduce((sum, tier) => sum + tier.correct, 0);
const hostedTotal = hosted.tiers.reduce((sum, tier) => sum + tier.total, 0);
if (
  hostedCorrect !== 985 ||
  hostedTotal !== 1000 ||
  hosted.scored.correct !== hostedCorrect ||
  hosted.scored.total !== hostedTotal ||
  hosted.diagnosticTier0.correct + hosted.scored.correct !==
    hosted.visibleTotal.correct ||
  hosted.diagnosticTier0.total + hosted.scored.total !==
    hosted.visibleTotal.total
) {
  fail("hosted aggregate result changed");
}

const full = data.fullWidth;
requireSha256(full.receiptSha256, "terminal receipt");
requireSha256(full.checkpointSha256, "terminal checkpoint");
validatePairedTiers("full-width screen", full.screen.tiers, 64, 640);
validatePairedTiers(
  "full-width confirmation",
  full.confirmation.tiers,
  256,
  2548,
);
if (
  full.status !== "completed_failed_gate" ||
  full.targetWidth !== 2048 ||
  full.parameters !== 126603 ||
  full.startStep !== 60000 ||
  full.endStep !== 120000 ||
  full.nearPass.step !== 117000 ||
  full.nearPass.correct !== 639 ||
  full.nearPass.total !== 640 ||
  full.screen.correct !== 640 ||
  full.screen.total !== 640 ||
  full.confirmation.correct !== 2548 ||
  full.confirmation.total !== 2560 ||
  full.confirmation.passed !== false ||
  full.smallPrime.fixed.correct !== 40954 ||
  full.smallPrime.fixed.total !== 40954 ||
  full.smallPrime.dynamic.correct !== 40954 ||
  full.smallPrime.dynamic.total !== 40954
) {
  fail("terminal L2048 result changed");
}

const mini = data.miniFermat;
requireGitSha(mini.artifactRevision, "Mini Fermat artifact revision");
requireSha256(mini.artifactSetSha256, "Mini Fermat artifact set");
requireSha256(mini.receiptSha256, "Mini Fermat receipt");
if (
  mini.status !== "completed_failed" ||
  mini.correct !== 119 ||
  mini.total !== 128 ||
  mini.f11FailureCount !== 9 ||
  mini.failureIndices.join(",") !== "62,87,89,94,99,108,109,117,120" ||
  mini.artifactRevision !== hosted.evaluatedRevision
) {
  fail("Mini Fermat diagnostic changed");
}

if (
  data.historicalV8.parameters !== 470849 ||
  data.historicalV8.structuredCorrect !== 759 ||
  data.historicalV8.structuredTotal !== 768 ||
  data.historicalV8.fermatCorrect !== 119 ||
  data.historicalV8.fermatTotal !== 128
) {
  fail("historical v8 evidence changed");
}
requireSha256(
  data.historicalV8.heldoutReceiptSha256,
  "historical v8 heldout receipt",
);
requireSha256(
  data.historicalV8.traceReceiptSha256,
  "historical v8 trace receipt",
);

console.log("Animation data preflight passed for terminal current evidence");
