import { Easing, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame, StatusPill } from "../components/ResearchVisuals";
import { animationData } from "../data/animationData";

export const LimitsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const mini = animationData.miniFermat;
  const v8 = animationData.historicalV8;
  const trace = v8.f11;
  const failures = new Set(mini.failureIndices);

  return (
    <SceneFrame
      eyebrow="Structured failure"
      title="Both models scored 119/128 on the Fermat family."
      footer="Mini is an output-only Fermat diagnostic. The v8 transition trace is a separate historical artifact."
    >
      <div
        style={{
          display: "grid",
          gap: 44,
          gridTemplateColumns: "1fr 1fr",
          marginTop: 34,
        }}
      >
        <div style={{ border: "1px solid #4f5758", padding: "27px 31px" }}>
          <div
            style={{
              alignItems: "center",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <StatusPill label="Mini v0.2" tone="neutral" />
            <div
              style={{
                color: "#e48a70",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 26,
              }}
            >
              {mini.correct}/{mini.total}
            </div>
          </div>
          <div
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 44,
              lineHeight: 1.15,
              marginTop: 22,
            }}
          >
            Nine observed failures.
            <br />
            All nine are F₁₁ rows.
          </div>
          <div
            style={{
              display: "grid",
              gap: 5,
              gridTemplateColumns: "repeat(16, 1fr)",
              marginTop: 28,
            }}
          >
            {Array.from({ length: mini.total }, (_, index) => {
              const failed = failures.has(index);
              return (
                <div
                  key={index}
                  style={{
                    backgroundColor: failed ? "#bb654f" : "#3a8f87",
                    height: 12,
                    opacity: interpolate(
                      frame,
                      [28 + index * 0.65, 48 + index * 0.65],
                      [0, 1],
                      {
                        extrapolateLeft: "clamp",
                        extrapolateRight: "clamp",
                      },
                    ),
                  }}
                />
              );
            })}
          </div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 17,
              lineHeight: 1.55,
              marginTop: 19,
            }}
          >
            0-based failed indices: {mini.failureIndices.join(", ")}
            <br />
            exact public artifact revision {mini.artifactRevision.slice(0, 10)}…
          </div>
        </div>

        <div
          style={{
            border: "1px solid #4f5758",
            opacity: interpolate(frame, [98, 136], [0, 1], {
              easing: Easing.bezier(0.16, 1, 0.3, 1),
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            padding: "27px 31px",
          }}
        >
          <div
            style={{
              alignItems: "center",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <StatusPill label="historical v8" tone="neutral" />
            <div
              style={{
                color: "#a6aaa5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 21,
              }}
            >
              structured {v8.structuredCorrect}/{v8.structuredTotal}
            </div>
          </div>
          <div
            style={{
              color: "#e48a70",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 18,
              fontWeight: 700,
              letterSpacing: 2,
              marginTop: 26,
            }}
          >
            ONE DECISIVE F₁₁ TRACE
          </div>
          <div
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 42,
              lineHeight: 1.16,
              marginTop: 15,
            }}
          >
            {trace.certifiedExactPrefixTransitions.toLocaleString("en-US")}{" "}
            exact transitions.
            <br />
            Then one wrong learned step.
          </div>
          <div
            style={{
              alignItems: "center",
              display: "grid",
              gridTemplateColumns: "1fr 18px",
              marginTop: 25,
            }}
          >
            <div style={{ backgroundColor: "#3a8f87", height: 13 }} />
            <div style={{ backgroundColor: "#bb654f", height: 31 }} />
          </div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 19,
              lineHeight: 1.65,
              marginTop: 22,
            }}
          >
            phase {trace.phase} · step {trace.phaseStep}
            <br />
            state {trace.state} · d = {trace.digit}
            <br />
            {trace.wrongOutputBits} wrong output bits · margin{" "}
            {trace.minimumSignedTargetLogitMargin.toFixed(2)}
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
