import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { formatInteger, selectedArm } from "../data/animationData";

const ResultCard: React.FC<{
  label: string;
  value: string;
  tone: "teal" | "coral";
}> = ({ label, value, tone }) => (
  <div
    style={{
      border: `1px solid ${tone === "teal" ? "#3d6662" : "#704238"}`,
      minWidth: 330,
      padding: "20px 26px",
      textAlign: "left",
    }}
  >
    <div
      style={{
        color: tone === "teal" ? "#72d1c5" : "#e48a70",
        fontFamily:
          'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 27,
      }}
    >
      {value}
    </div>
    <div
      style={{
        color: "#a6aaa5",
        fontFamily:
          'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 17,
        marginTop: 9,
      }}
    >
      {label}
    </div>
  </div>
);

export const EndScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        alignItems: "center",
        backgroundColor: "#111315",
        backgroundImage:
          "radial-gradient(circle at center, rgba(114,209,197,0.11), transparent 47%), linear-gradient(rgba(234,230,220,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(234,230,220,0.025) 1px, transparent 1px)",
        backgroundSize: "auto, 72px 72px, 72px 72px",
        color: "#ece8dd",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        justifyContent: "center",
        opacity: interpolate(frame, [0, 18, 214, 229], [0, 1, 1, 0], {
          easing: Easing.bezier(0.16, 1, 0.3, 1),
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
        padding: "76px 110px",
        width: "100%",
      }}
    >
      <Interactive.Div
        name="End title"
        style={{
          color: "#ece8dd",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 112,
          fontWeight: 500,
          letterSpacing: -4,
          lineHeight: 1,
          scale: interpolate(frame, [8, 42], [0.96, 1], {
            easing: Easing.spring({ damping: 200 }),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            output: "perceptual-scale",
          }),
        }}
      >
        NeuralHorner
      </Interactive.Div>
      <div
        style={{
          color: "#72d1c5",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 31,
          letterSpacing: 1,
          marginTop: 23,
        }}
      >
        MiniNeuralHorner · {formatInteger(selectedArm.parameters)} parameters
      </div>
      <div style={{ display: "flex", gap: 18, marginTop: 38 }}>
        <ResultCard
          label="hosted tiers 1–9"
          value="100 / 100 each"
          tone="teal"
        />
        <ResultCard label="120K L2048 screen" value="640 / 640" tone="teal" />
        <ResultCard
          label="larger confirmation · strict gate missed"
          value="2,548 / 2,560"
          tone="coral"
        />
      </div>
      <div
        style={{
          color: "#a6aaa5",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 30,
          marginTop: 38,
        }}
      >
        Experiments in learned modular arithmetic
      </div>
      <div
        style={{
          color: "#8f938f",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 26,
          letterSpacing: 1,
          marginTop: 19,
        }}
      >
        github.com/Robby955/neural-horner
      </div>
    </div>
  );
};
