import { Easing, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame, StatusPill } from "../components/ResearchVisuals";
import { animationData, formatInteger } from "../data/animationData";

const tiers = [6, 7, 8, 9, 10] as const;
const modes = [
  { key: "fixed" as const, label: "fixed W=2048" },
  { key: "dynamic" as const, label: "dynamic W" },
];

export const ProgressScene: React.FC = () => {
  const frame = useCurrentFrame();
  const full = animationData.fullWidth;

  return (
    <SceneFrame
      eyebrow="L = 2048 horizon extension"
      title="The 640/640 screen became 2,548/2,560 under confirmation."
      footer="Full-state resume 60K → 120K · learning rate fixed at 4.5e−5 · no checkpoint selected."
    >
      <div
        style={{
          alignItems: "stretch",
          display: "grid",
          gap: 44,
          gridTemplateColumns: "1.35fr 0.65fr",
          marginTop: 34,
        }}
      >
        <div>
          <div
            style={{
              alignItems: "center",
              display: "grid",
              gap: 18,
              gridTemplateColumns: "1fr 80px 1fr",
              marginBottom: 26,
            }}
          >
            <div
              style={{
                border: "1px solid #4f5758",
                color: "#a6aaa5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 22,
                padding: "19px 24px",
                textAlign: "center",
              }}
            >
              {full.nearPass.step / 1000}K screen{" "}
              <span style={{ color: "#d9b66f" }}>
                {full.nearPass.correct}/{full.nearPass.total}
              </span>
            </div>
            <div
              style={{ color: "#72d1c5", fontSize: 42, textAlign: "center" }}
            >
              →
            </div>
            <div
              style={{
                backgroundColor: "#17302e",
                border: "1px solid #72d1c5",
                color: "#ece8dd",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 22,
                padding: "19px 24px",
                textAlign: "center",
              }}
            >
              120K screen <span style={{ color: "#72d1c5" }}>640/640</span>
            </div>
          </div>

          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 18,
              letterSpacing: 2,
              marginBottom: 12,
              textTransform: "uppercase",
            }}
          >
            larger confirmation
          </div>
          <div
            style={{
              display: "grid",
              gap: 10,
              gridTemplateColumns: "170px repeat(5, 1fr)",
            }}
          >
            <div />
            {tiers.map((tier) => (
              <div
                key={`header-${tier}`}
                style={{
                  color: "#a6aaa5",
                  fontFamily:
                    'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                  fontSize: 20,
                  padding: "8px 0",
                  textAlign: "center",
                }}
              >
                T{tier}
              </div>
            ))}
            {modes.flatMap((mode, rowIndex) => [
              <div
                key={`label-${mode.key}`}
                style={{
                  alignItems: "center",
                  color: "#ece8dd",
                  display: "flex",
                  fontFamily:
                    'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                  fontSize: 18,
                  minHeight: 92,
                }}
              >
                {mode.label}
              </div>,
              ...tiers.map((tier, columnIndex) => {
                const result = full.confirmation.tiers[String(tier)];
                const correct = result[mode.key];
                const exact = correct === result.total;
                return (
                  <div
                    key={`${mode.key}-${tier}`}
                    style={{
                      alignItems: "center",
                      backgroundColor: exact ? "#17302e" : "#2d201d",
                      border: `1px solid ${exact ? "#3d6662" : "#8e5042"}`,
                      color: exact ? "#72d1c5" : "#e48a70",
                      display: "flex",
                      flexDirection: "column",
                      fontFamily:
                        'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                      fontSize: 24,
                      justifyContent: "center",
                      minHeight: 92,
                      opacity: interpolate(
                        frame,
                        [
                          72 + rowIndex * 28 + columnIndex * 9,
                          98 + rowIndex * 28 + columnIndex * 9,
                        ],
                        [0, 1],
                        {
                          easing: Easing.bezier(0.16, 1, 0.3, 1),
                          extrapolateLeft: "clamp",
                          extrapolateRight: "clamp",
                        },
                      ),
                    }}
                  >
                    <div>{correct}</div>
                    <div
                      style={{ color: "#8f938f", fontSize: 15, marginTop: 7 }}
                    >
                      / 256
                    </div>
                  </div>
                );
              }),
            ])}
          </div>
        </div>

        <div
          style={{
            borderLeft: "1px solid #4f5758",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            paddingLeft: 38,
          }}
        >
          <StatusPill label="strict gate missed" tone="failed" />
          <div
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 66,
              letterSpacing: -2,
              lineHeight: 1,
              marginTop: 28,
            }}
          >
            {formatInteger(full.confirmation.correct)}
          </div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 22,
              marginTop: 10,
            }}
          >
            / {formatInteger(full.confirmation.total)}
          </div>
          <div
            style={{
              borderTop: "1px solid #343a3c",
              color: "#72d1c5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 20,
              lineHeight: 1.6,
              marginTop: 28,
              paddingTop: 22,
            }}
          >
            small-prime transitions
            <br />
            {formatInteger(full.smallPrime.fixed.correct)} /{" "}
            {formatInteger(full.smallPrime.fixed.total)} fixed
            <br />
            {formatInteger(full.smallPrime.dynamic.correct)} /{" "}
            {formatInteger(full.smallPrime.dynamic.total)} dynamic
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
