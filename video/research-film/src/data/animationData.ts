import rawData from "./animation-data.json";

export type EvidenceMode = "current";

export type PilotArm = {
  id: string;
  hidden: number;
  parameters: number;
  outcome: "passed" | "failed_endpoint";
  receiptSha256: string;
  checkpointSha256: string;
};

export type TierCount = {
  tier: number;
  correct: number;
  total: number;
};

export type PairedTierCount = {
  fixed: number;
  dynamic: number;
  total: number;
};

type AnimationData = {
  schema: string;
  sourceCommits: {
    pilot: string;
    horizon: string;
    recoveryExecutor: string;
  };
  recurrence: {
    formula: string;
    statePolicy: string;
    phases: string[];
  };
  pilot: {
    status: string;
    role: string;
    width: number;
    optimizerSteps: number;
    masterSeed: number;
    evaluationTiers: number[];
    endpointCasesPerTierMode: number;
    selectedArm: string;
    arms: PilotArm[];
    b063: {
      smallPrimeCasesPerWidth: number;
      smallPrimeFixedExact: boolean;
      smallPrimeDynamicExact: boolean;
      endpoint: Record<string, PairedTierCount>;
    };
  };
  hostedEvaluation: {
    status: "completed";
    classification: string;
    repository: string;
    evaluatedRevision: string;
    frontierTier: number;
    tiers: TierCount[];
    scored: { correct: number; total: number };
    uiRoundedPercent: number;
    diagnosticTier0: { correct: number; total: number };
    visibleTotal: { correct: number; total: number };
    runtimeSeconds: number;
    artifactSizeDisplay: string;
    evidenceSha256: string;
  };
  fullWidth: {
    status: "completed_failed_gate";
    targetWidth: number;
    parameters: number;
    startStep: number;
    endStep: number;
    floorLearningRate: number;
    nearPass: { step: number; correct: number; total: number };
    screen: {
      casesPerTierMode: number;
      correct: number;
      total: number;
      tiers: Record<string, PairedTierCount>;
    };
    confirmation: {
      casesPerTierMode: number;
      correct: number;
      total: number;
      passed: boolean;
      tiers: Record<string, PairedTierCount>;
    };
    smallPrime: {
      fixed: { correct: number; total: number; width: number };
      dynamic: { correct: number; total: number; width: number };
    };
    receiptSha256: string;
    checkpointSha256: string;
  };
  miniFermat: {
    status: "completed_failed";
    classification: string;
    family: string;
    orientation: string;
    seed: number;
    correct: number;
    total: number;
    f11FailureCount: number;
    failureIndices: number[];
    artifactRevision: string;
    artifactSetSha256: string;
    receiptSha256: string;
  };
  historicalV8: {
    parameters: number;
    structuredCorrect: number;
    structuredTotal: number;
    fermatCorrect: number;
    fermatTotal: number;
    f11: {
      phase: string;
      phaseStep: number;
      certifiedExactPrefixTransitions: number;
      state: string;
      digit: number;
      wrongOutputBits: number;
      minimumSignedTargetLogitMargin: number;
    };
    heldoutReceiptSha256: string;
    traceReceiptSha256: string;
  };
  sources: string[];
};

export const animationData = rawData as AnimationData;

const sha256 = /^[a-f0-9]{64}$/;
const gitSha = /^[a-f0-9]{40}$/;

const assertTierTotals = (
  tiers: Record<string, PairedTierCount>,
  expectedPerMode: number,
  expectedCorrect: number,
) => {
  const tierNumbers = Object.keys(tiers)
    .map(Number)
    .sort((a, b) => a - b);
  if (tierNumbers.join(",") !== "6,7,8,9,10") {
    throw new Error("Full-width evidence must contain tiers 6 through 10");
  }
  const correct = Object.values(tiers).reduce(
    (sum, tier) => sum + tier.fixed + tier.dynamic,
    0,
  );
  if (
    Object.values(tiers).some((tier) => tier.total !== expectedPerMode) ||
    correct !== expectedCorrect
  ) {
    throw new Error(
      "Full-width tier counts do not match their declared totals",
    );
  }
};

const assertBaseEvidence = () => {
  if (animationData.schema !== "neuralhorner-research-film-data-v2") {
    throw new Error("Unexpected NeuralHorner animation-data schema");
  }
  if (
    Object.values(animationData.sourceCommits).some((sha) => !gitSha.test(sha))
  ) {
    throw new Error("Animation data is missing a full source commit");
  }

  const selected = animationData.pilot.arms.find(
    (arm) => arm.id === animationData.pilot.selectedArm,
  );
  if (
    !selected ||
    selected.id !== "B127" ||
    selected.parameters !== 126603 ||
    selected.outcome !== "passed"
  ) {
    throw new Error("Animation data does not contain the verified B127 pilot");
  }

  const hosted = animationData.hostedEvaluation;
  const hostedCorrect = hosted.tiers.reduce(
    (sum, tier) => sum + tier.correct,
    0,
  );
  const hostedTotal = hosted.tiers.reduce((sum, tier) => sum + tier.total, 0);
  if (
    hosted.status !== "completed" ||
    !gitSha.test(hosted.evaluatedRevision) ||
    !sha256.test(hosted.evidenceSha256) ||
    hosted.tiers.length !== 10 ||
    hosted.tiers.some((tier, index) => tier.tier !== index + 1) ||
    hosted.tiers
      .slice(0, 9)
      .some((tier) => tier.correct !== 100 || tier.total !== 100) ||
    hosted.tiers[9].correct !== 85 ||
    hosted.tiers[9].total !== 100 ||
    hostedCorrect !== hosted.scored.correct ||
    hostedTotal !== hosted.scored.total ||
    hosted.scored.correct !== 985 ||
    hosted.scored.total !== 1000
  ) {
    throw new Error("Hosted MiniNeuralHorner evidence is inconsistent");
  }

  const full = animationData.fullWidth;
  assertTierTotals(full.screen.tiers, 64, full.screen.correct);
  assertTierTotals(full.confirmation.tiers, 256, full.confirmation.correct);
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
    full.confirmation.passed ||
    full.smallPrime.fixed.correct !== full.smallPrime.fixed.total ||
    full.smallPrime.dynamic.correct !== full.smallPrime.dynamic.total ||
    !sha256.test(full.receiptSha256) ||
    !sha256.test(full.checkpointSha256)
  ) {
    throw new Error("Terminal L2048 evidence is inconsistent");
  }

  const mini = animationData.miniFermat;
  if (
    mini.status !== "completed_failed" ||
    mini.correct !== 119 ||
    mini.total !== 128 ||
    mini.f11FailureCount !== 9 ||
    mini.failureIndices.length !== 9 ||
    mini.artifactRevision !== hosted.evaluatedRevision ||
    !sha256.test(mini.artifactSetSha256) ||
    !sha256.test(mini.receiptSha256)
  ) {
    throw new Error("MiniNeuralHorner Fermat evidence is inconsistent");
  }
};

export const assertEvidenceMode = (mode: EvidenceMode) => {
  if (mode !== "current") {
    throw new Error(`Unsupported evidence mode: ${mode as string}`);
  }
  assertBaseEvidence();
};

export const selectedArm = animationData.pilot.arms.find(
  (arm) => arm.id === animationData.pilot.selectedArm,
) as PilotArm;

export const formatInteger = (value: number) =>
  new Intl.NumberFormat("en-US").format(value);
