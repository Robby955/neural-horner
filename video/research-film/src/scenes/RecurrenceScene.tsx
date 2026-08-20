import {
  Easing,
  Interactive,
  interpolate,
  useCurrentFrame,
} from "remotion";
import {BitTape, SceneFrame, StatusPill} from "../components/ResearchVisuals";

export const RecurrenceScene: React.FC = () => {
  const frame = useCurrentFrame();
  const revealOutput = frame >= 180;

  return (
    <SceneFrame
      eyebrow="One integer step"
      title="A Horner transition doubles, conditionally adds, then reduces."
      footer="Exact example covered by the B127 exhaustive p < 64 transition gate."
    >
      <div
        style={{
          alignItems: "center",
          display: "grid",
          gap: 52,
          gridTemplateColumns: "0.9fr 1.2fr",
          marginTop: 54,
        }}
      >
        <div style={{display: "flex", flexDirection: "column", gap: 17}}>
          <BitTape bits="01011" label="s = 11" />
          <BitTape bits="00111" label="x = 7" />
          <BitTape bits="11101" label="p = 29" />
          <div
            style={{
              alignItems: "center",
              display: "flex",
              gap: 18,
              marginLeft: 96,
              marginTop: 12,
            }}
          >
            <div
              style={{
                color: "#a6aaa5",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 24,
              }}
            >
              d
            </div>
            <div
              style={{
                alignItems: "center",
                backgroundColor: "#bb654f",
                border: "1px solid #e48a70",
                color: "#fff4e9",
                display: "flex",
                fontFamily:
                  'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
                fontSize: 28,
                height: 48,
                justifyContent: "center",
                width: 48,
              }}
            >
              1
            </div>
          </div>
        </div>
        <div>
          <Interactive.Div
            name="Recurrence equation"
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 73,
              letterSpacing: -1,
              lineHeight: 1.1,
              opacity: interpolate(frame, [62, 98], [0, 1], {
                easing: Easing.bezier(0.16, 1, 0.3, 1),
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            s′ = (2s + d x) mod p
          </Interactive.Div>
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 32,
              marginTop: 32,
              opacity: interpolate(frame, [105, 140], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            (2 · 11 + 1 · 7) mod 29 = 0
          </div>
          <div
            style={{
              alignItems: "center",
              display: "flex",
              gap: 24,
              marginTop: 42,
              opacity: interpolate(frame, [170, 208], [0, 1], {
                easing: Easing.bezier(0.16, 1, 0.3, 1),
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <BitTape bits="00000" label="s′" activeIndex={revealOutput ? -1 : 0} />
            <StatusPill label="exact wrap" tone="exact" />
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
