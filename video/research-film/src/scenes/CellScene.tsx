import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { BitTape, SceneFrame, StatusPill } from "../components/ResearchVisuals";

const RecurrentLayer: React.FC<{ label: string; delay: number }> = ({
  label,
  delay,
}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        border: "1px solid #4f5758",
        height: 86,
        opacity: interpolate(frame, [delay, delay + 24], [0, 1], {
          easing: Easing.bezier(0.16, 1, 0.3, 1),
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
        position: "relative",
      }}
    >
      <div
        style={{
          color: "#a6aaa5",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 18,
          left: 22,
          position: "absolute",
          top: 14,
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "#72d1c5",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 30,
          left: 170,
          letterSpacing: 13,
          position: "absolute",
          right: 36,
          top: 21,
          whiteSpace: "nowrap",
        }}
      >
        → → → → → → →
      </div>
      <div
        style={{
          bottom: 12,
          color: "#d9b66f",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 26,
          left: 190,
          letterSpacing: 13,
          position: "absolute",
          right: 30,
          whiteSpace: "nowrap",
        }}
      >
        ← ← ← ← ← ← ←
      </div>
      <div
        style={{
          backgroundColor: "#72d1c5",
          borderRadius: 8,
          height: 10,
          left: interpolate(frame, [delay + 10, delay + 95], [170, 670], {
            easing: Easing.linear,
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          position: "absolute",
          top: 29,
          width: 10,
        }}
      />
      <div
        style={{
          backgroundColor: "#d9b66f",
          borderRadius: 8,
          height: 10,
          left: interpolate(frame, [delay + 10, delay + 95], [670, 170], {
            easing: Easing.linear,
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          position: "absolute",
          top: 57,
          width: 10,
        }}
      />
    </div>
  );
};

export const CellScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame
      eyebrow="The learned cell"
      title="Two bidirectional recurrent layers read the complete binary state."
      footer="Architecture: 3 channels → 96-dimensional projection + digit embedding → 2-layer BiGRU → bit logits."
    >
      <div
        style={{
          display: "grid",
          gap: 46,
          gridTemplateColumns: "0.78fr 1.22fr 0.52fr",
          marginTop: 44,
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 14,
            justifyContent: "center",
          }}
        >
          <BitTape bits="01011010" label="s" compact />
          <BitTape bits="00110101" label="x" compact />
          <BitTape bits="11101011" label="p" compact />
          <div
            style={{
              alignItems: "center",
              display: "flex",
              gap: 16,
              marginLeft: 72,
              marginTop: 14,
            }}
          >
            <StatusPill label="d = 0 or 1" tone="neutral" />
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
          <div
            style={{
              alignItems: "center",
              border: "1px solid #4f5758",
              color: "#a6aaa5",
              display: "flex",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 22,
              height: 64,
              justifyContent: "center",
            }}
          >
            96-dimensional position representation
          </div>
          <RecurrentLayer
            label="BiGRU layer 1 · hidden 61 × 2 directions"
            delay={48}
          />
          <RecurrentLayer
            label="BiGRU layer 2 · hidden 61 × 2 directions"
            delay={82}
          />
        </div>
        <Interactive.Div
          name="Hard state output"
          style={{
            alignItems: "center",
            borderLeft: "1px solid #4f5758",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            opacity: interpolate(frame, [140, 182], [0, 1], {
              easing: Easing.bezier(0.16, 1, 0.3, 1),
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            paddingLeft: 34,
          }}
        >
          <div
            style={{
              color: "#a6aaa5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 21,
              textAlign: "center",
            }}
          >
            logit &gt; 0
          </div>
          <div
            style={{
              color: "#ece8dd",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 46,
              letterSpacing: 7,
              lineHeight: 1.4,
              marginTop: 18,
              textAlign: "center",
            }}
          >
            0 1 1 0
            <br />1 0 0 1
          </div>
          <div style={{ marginTop: 24 }}>
            <StatusPill label="hard state" tone="exact" />
          </div>
        </Interactive.Div>
      </div>
      <div
        style={{
          color: "#a6aaa5",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 28,
          marginTop: 34,
          opacity: interpolate(frame, [200, 234], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          textAlign: "center",
        }}
      >
        The BiGRU moves across bit positions inside one state. The Horner clock
        moves between states.
      </div>
    </SceneFrame>
  );
};
