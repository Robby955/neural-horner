import { Easing, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame, StatusPill } from "../components/ResearchVisuals";
import {
  animationData,
  formatInteger,
  selectedArm,
} from "../data/animationData";

export const HostedScene: React.FC = () => {
  const frame = useCurrentFrame();
  const hosted = animationData.hostedEvaluation;

  return (
    <SceneFrame
      eyebrow="Hosted evaluation"
      title="The compact model was exact through tier 9 on the public Playground."
      footer={`Owner-transcribed completed Playground UI result · evaluated revision ${hosted.evaluatedRevision.slice(0, 12)}…`}
    >
      <div
        style={{
          display: "grid",
          gap: 58,
          gridTemplateColumns: "0.72fr 1.28fr",
          marginTop: 38,
        }}
      >
        <div
          style={{
            border: "1px solid #4f5758",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            padding: "30px 34px",
          }}
        >
          <div
            style={{
              color: "#72d1c5",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 88,
              letterSpacing: -3,
              lineHeight: 1,
            }}
          >
            98.5%
          </div>
          <div
            style={{
              color: "#ece8dd",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 24,
              marginTop: 18,
            }}
          >
            {formatInteger(hosted.scored.correct)} /{" "}
            {formatInteger(hosted.scored.total)} scored cases
          </div>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 12,
              marginTop: 27,
            }}
          >
            <StatusPill
              label={`frontier T${hosted.frontierTier}`}
              tone="exact"
            />
            <StatusPill
              label={`${formatInteger(selectedArm.parameters)} params`}
              tone="neutral"
            />
          </div>
          <div
            style={{
              borderTop: "1px solid #343a3c",
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 20,
              lineHeight: 1.75,
              marginTop: 26,
              paddingTop: 22,
            }}
          >
            runtime {hosted.runtimeSeconds.toFixed(3)} s
            <br />
            artifact {hosted.artifactSizeDisplay}
            <br />
            T0 diagnostic {hosted.diagnosticTier0.correct}/
            {hosted.diagnosticTier0.total} · excluded above
          </div>
        </div>

        <div
          style={{ display: "grid", gap: 13, gridTemplateColumns: "1fr 1fr" }}
        >
          {hosted.tiers.map((tier, index) => {
            const exact = tier.correct === tier.total;
            return (
              <div
                key={tier.tier}
                style={{
                  alignItems: "center",
                  border: `1px solid ${exact ? "#3d6662" : "#704238"}`,
                  display: "grid",
                  gap: 16,
                  gridTemplateColumns: "64px 1fr 88px",
                  minHeight: 70,
                  opacity: interpolate(
                    frame,
                    [28 + index * 10, 54 + index * 10],
                    [0, 1],
                    {
                      easing: Easing.bezier(0.16, 1, 0.3, 1),
                      extrapolateLeft: "clamp",
                      extrapolateRight: "clamp",
                    },
                  ),
                  padding: "12px 16px",
                  translate: interpolate(
                    frame,
                    [28 + index * 10, 54 + index * 10],
                    ["18px 0px", "0px 0px"],
                    {
                      easing: Easing.bezier(0.16, 1, 0.3, 1),
                      extrapolateLeft: "clamp",
                      extrapolateRight: "clamp",
                    },
                  ),
                }}
              >
                <div
                  style={{
                    color: exact ? "#72d1c5" : "#e48a70",
                    fontFamily:
                      'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                    fontSize: 22,
                    fontWeight: 700,
                  }}
                >
                  T{tier.tier}
                </div>
                <div style={{ backgroundColor: "#2c2221", height: 12 }}>
                  <div
                    style={{
                      backgroundColor: exact ? "#3a8f87" : "#bb654f",
                      height: "100%",
                      width: `${tier.correct}%`,
                    }}
                  />
                </div>
                <div
                  style={{
                    color: exact ? "#ece8dd" : "#e48a70",
                    fontFamily:
                      'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                    fontSize: 19,
                    textAlign: "right",
                  }}
                >
                  {tier.correct}/{tier.total}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </SceneFrame>
  );
};
