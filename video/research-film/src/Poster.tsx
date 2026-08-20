import { AbsoluteFill } from "remotion";
import {
  animationData,
  formatInteger,
  selectedArm,
} from "./data/animationData";

const PosterMetric: React.FC<{
  label: string;
  value: string;
  accent?: boolean;
}> = ({ label, value, accent = false }) => (
  <div
    style={{
      borderLeft: `2px solid ${accent ? "#e48a70" : "#72d1c5"}`,
      paddingLeft: 20,
    }}
  >
    <div
      style={{
        color: accent ? "#e48a70" : "#ece8dd",
        fontFamily:
          'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 27,
      }}
    >
      {value}
    </div>
    <div
      style={{
        color: "#8f938f",
        fontFamily:
          'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 18,
        marginTop: 8,
      }}
    >
      {label}
    </div>
  </div>
);

export const Poster: React.FC = () => {
  const full = animationData.fullWidth;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#111315",
        backgroundImage:
          "radial-gradient(circle at 76% 26%, rgba(114,209,197,0.12), transparent 32%), linear-gradient(rgba(234,230,220,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(234,230,220,0.035) 1px, transparent 1px)",
        backgroundSize: "auto, 72px 72px, 72px 72px",
        color: "#ece8dd",
        padding: "92px 112px",
      }}
    >
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
        Research film
      </div>
      <div
        style={{
          color: "#ece8dd",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 132,
          letterSpacing: -5,
          lineHeight: 0.96,
          marginTop: 26,
        }}
      >
        NeuralHorner
      </div>
      <div
        style={{
          color: "#a6aaa5",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 42,
          marginTop: 21,
        }}
      >
        Learned modular arithmetic under exact evaluation
      </div>

      <div
        style={{
          alignItems: "center",
          display: "grid",
          gap: 70,
          gridTemplateColumns: "1.22fr 0.78fr",
          marginTop: 72,
        }}
      >
        <div>
          <div
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 80,
              letterSpacing: -2,
            }}
          >
            s′ = (2s + d x) mod p
          </div>
          <div
            style={{
              color: "#72d1c5",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: 25,
              letterSpacing: 12,
              marginTop: 38,
            }}
          >
            0 1 1 0 1 0 0 1 1 1 0 0 1 0 1 1
          </div>
        </div>
        <div
          style={{
            alignItems: "center",
            border: "1px solid #4f5758",
            borderRadius: 999,
            display: "flex",
            height: 148,
            justifyContent: "center",
            justifySelf: "end",
            width: 148,
          }}
        >
          <div
            style={{
              borderBottom: "27px solid transparent",
              borderLeft: "42px solid #72d1c5",
              borderTop: "27px solid transparent",
              marginLeft: 11,
            }}
          />
        </div>
      </div>

      <div
        style={{
          borderTop: "1px solid #4f5758",
          display: "grid",
          gap: 45,
          gridTemplateColumns: "repeat(3, 1fr)",
          marginTop: 72,
          paddingTop: 37,
        }}
      >
        <PosterMetric
          label="parameter reduction"
          value={`${formatInteger(animationData.historicalV8.parameters)} → ${formatInteger(selectedArm.parameters)}`}
        />
        <PosterMetric label="hosted tiers 1–9" value="100 / 100 each" />
        <PosterMetric
          accent
          label="L2048 confirmation · strict gate missed"
          value={`${formatInteger(full.confirmation.correct)} / ${formatInteger(full.confirmation.total)}`}
        />
      </div>
    </AbsoluteFill>
  );
};
