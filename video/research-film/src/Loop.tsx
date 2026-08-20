import {
  AbsoluteFill,
  Easing,
  Sequence,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { StatusPill } from "./components/ResearchVisuals";
import {
  animationData,
  assertEvidenceMode,
  type EvidenceMode,
  formatInteger,
  selectedArm,
} from "./data/animationData";

const LoopPanel: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        alignItems: "center",
        backgroundColor: "#111315",
        backgroundImage:
          "linear-gradient(rgba(234,230,220,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(234,230,220,0.035) 1px, transparent 1px)",
        backgroundSize: "72px 72px",
        color: "#ece8dd",
        display: "flex",
        justifyContent: "center",
        opacity: interpolate(frame, [0, 12, 103, 119], [0, 1, 1, 0], {
          easing: Easing.bezier(0.16, 1, 0.3, 1),
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
        padding: "100px 120px",
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

const FormulaLoop: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <LoopPanel>
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            color: "#72d1c5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 22,
            fontWeight: 700,
            letterSpacing: 4,
            textTransform: "uppercase",
          }}
        >
          one learned transition
        </div>
        <div
          style={{
            color: "#ece8dd",
            fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
            fontSize: 108,
            letterSpacing: -3,
            marginTop: 28,
            scale: interpolate(frame, [8, 44], [0.95, 1], {
              easing: Easing.spring({ damping: 200 }),
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              output: "perceptual-scale",
            }),
          }}
        >
          s′ = (2s + d x) mod p
        </div>
        <div
          style={{
            color: "#a6aaa5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 28,
            letterSpacing: 13,
            marginTop: 42,
          }}
        >
          0 1 0 1 1 0 1 0 0 1 1 0 1 1 1 0
        </div>
      </div>
    </LoopPanel>
  );
};

const ScheduleLoop: React.FC = () => (
  <LoopPanel>
    <div style={{ width: "100%" }}>
      <div
        style={{
          color: "#ece8dd",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 66,
          marginBottom: 48,
          textAlign: "center",
        }}
      >
        the same weights, three times
      </div>
      <div
        style={{
          display: "grid",
          gap: 24,
          gridTemplateColumns: "repeat(3, 1fr)",
        }}
      >
        {[
          ["1", "reduce a", "x = 1"],
          ["2", "reduce b", "x = 1"],
          ["3", "multiply", "x = ā"],
        ].map(([index, title, control]) => (
          <div
            key={index}
            style={{
              border: "1px solid #4f5758",
              padding: "31px 35px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                color: "#72d1c5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 18,
                letterSpacing: 3,
              }}
            >
              PASS {index}
            </div>
            <div
              style={{
                color: "#ece8dd",
                fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
                fontSize: 45,
                marginTop: 16,
              }}
            >
              {title}
            </div>
            <div
              style={{
                color: "#a6aaa5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 20,
                marginTop: 18,
              }}
            >
              {control}
            </div>
          </div>
        ))}
      </div>
    </div>
  </LoopPanel>
);

const ResultLoop: React.FC = () => {
  return (
    <LoopPanel>
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            color: "#ece8dd",
            fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
            fontSize: 82,
            letterSpacing: -2,
          }}
        >
          NeuralHorner
        </div>
        <div
          style={{
            color: "#72d1c5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 34,
            marginTop: 26,
          }}
        >
          MiniNeuralHorner · {formatInteger(selectedArm.parameters)} parameters
        </div>
        <div
          style={{
            display: "flex",
            gap: 18,
            justifyContent: "center",
            marginTop: 35,
          }}
        >
          <StatusPill label="hosted T1–T9 exact" tone="exact" />
          <StatusPill label="L2048 gate missed" tone="failed" />
        </div>
        <div
          style={{
            color: "#a6aaa5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 19,
            lineHeight: 1.6,
            marginTop: 33,
          }}
        >
          120K screen {animationData.fullWidth.screen.correct}/
          {animationData.fullWidth.screen.total} · confirmation{" "}
          {animationData.fullWidth.confirmation.correct}/
          {animationData.fullWidth.confirmation.total}
          <br />
          Mini Fermat diagnostic {animationData.miniFermat.correct}/
          {animationData.miniFermat.total}
        </div>
      </div>
    </LoopPanel>
  );
};

export type LoopProps = {
  evidenceMode: EvidenceMode;
};

export const Loop: React.FC<LoopProps> = ({ evidenceMode }) => {
  assertEvidenceMode(evidenceMode);
  return (
    <AbsoluteFill style={{ backgroundColor: "#111315" }}>
      <Sequence durationInFrames={120} name="Loop recurrence">
        <FormulaLoop />
      </Sequence>
      <Sequence from={120} durationInFrames={120} name="Loop schedule">
        <ScheduleLoop />
      </Sequence>
      <Sequence from={240} durationInFrames={90} name="Loop compression">
        <LoopPanel>
          <div style={{ textAlign: "center", width: "100%" }}>
            <div
              style={{
                color: "#a6aaa5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 23,
                letterSpacing: 3,
                textTransform: "uppercase",
              }}
            >
              frozen L128 pilot
            </div>
            <div
              style={{
                color: "#ece8dd",
                fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
                fontSize: 98,
                marginTop: 26,
              }}
            >
              470,849 → 126,603
            </div>
            <div
              style={{
                color: "#e48a70",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 23,
                marginTop: 25,
              }}
            >
              63,057 failed the endpoint rollout gate
            </div>
          </div>
        </LoopPanel>
      </Sequence>
      <Sequence from={330} durationInFrames={120} name="Loop result">
        <ResultLoop />
      </Sequence>
    </AbsoluteFill>
  );
};
