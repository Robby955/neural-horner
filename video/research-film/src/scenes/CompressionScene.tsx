import { Easing, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame, StatusPill } from "../components/ResearchVisuals";
import {
  animationData,
  formatInteger,
  selectedArm,
} from "../data/animationData";

export const CompressionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const maximum = animationData.pilot.arms[0].parameters;
  const compressionRatio = maximum / selectedArm.parameters;
  const reductionPercent = (1 - selectedArm.parameters / maximum) * 100;
  return (
    <SceneFrame
      eyebrow="Capacity pilot · L = 128"
      title="126,603 parameters passed at L = 128. 63,057 missed the rollout gate."
      footer="One seed · 24,000 updates · identical ordered transition stream · endpoint selection fixed in advance."
    >
      <div
        style={{
          display: "grid",
          gap: 56,
          gridTemplateColumns: "1.15fr 0.85fr",
          marginTop: 42,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {animationData.pilot.arms.map((arm, index) => (
            <div
              key={arm.id}
              style={{
                alignItems: "center",
                display: "grid",
                gap: 17,
                gridTemplateColumns: "78px 1fr 150px 130px",
                minHeight: 66,
              }}
            >
              <div
                style={{
                  color: arm.id === selectedArm.id ? "#ece8dd" : "#a6aaa5",
                  fontFamily:
                    'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                  fontSize: 23,
                  fontWeight: 700,
                }}
              >
                {arm.id}
              </div>
              <div
                style={{
                  backgroundColor: "#232729",
                  height: 26,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    backgroundColor:
                      arm.outcome === "passed" ? "#3a8f87" : "#bb654f",
                    height: "100%",
                    width: interpolate(
                      frame,
                      [30 + index * 18, 80 + index * 18],
                      ["0%", `${(arm.parameters / maximum) * 100}%`],
                      {
                        easing: Easing.bezier(0.16, 1, 0.3, 1),
                        extrapolateLeft: "clamp",
                        extrapolateRight: "clamp",
                      },
                    ),
                  }}
                />
              </div>
              <div
                style={{
                  color: "#ece8dd",
                  fontFamily:
                    'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                  fontSize: 21,
                  textAlign: "right",
                }}
              >
                {formatInteger(arm.parameters)}
              </div>
              <StatusPill
                label={
                  arm.id === selectedArm.id
                    ? "selected"
                    : arm.outcome === "passed"
                      ? "passed"
                      : "failed"
                }
                tone={
                  arm.id === selectedArm.id
                    ? "exact"
                    : arm.outcome === "passed"
                      ? "neutral"
                      : "failed"
                }
              />
            </div>
          ))}
        </div>
        <div
          style={{
            borderLeft: "1px solid #4f5758",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            paddingLeft: 44,
          }}
        >
          <div
            style={{
              color: "#72d1c5",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 58,
              lineHeight: 1.1,
            }}
          >
            {reductionPercent.toFixed(1)}% fewer parameters
          </div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 19,
              marginTop: 12,
            }}
          >
            470,849 → 126,603 · {compressionRatio.toFixed(2)}× reduction
          </div>
          <div
            style={{
              borderTop: "1px solid #4f5758",
              color: "#e48a70",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 32,
              lineHeight: 1.15,
              marginTop: 25,
              paddingTop: 22,
            }}
          >
            B063 passed both finite small-prime gates, then failed rollouts.
          </div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 20,
              lineHeight: 1.7,
              marginTop: 16,
            }}
          >
            fixed width: 40,954 / 40,954
            <br />
            dynamic width: 40,954 / 40,954
            <br />
            endpoint rollouts: failed
          </div>
          <div
            style={{
              borderTop: "1px solid #4f5758",
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 25,
              lineHeight: 1.25,
              marginTop: 20,
              paddingTop: 18,
            }}
          >
            Finite local coverage did not guarantee exact long composition.
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
