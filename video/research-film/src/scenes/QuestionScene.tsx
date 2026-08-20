import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame } from "../components/ResearchVisuals";

const DecimalBand: React.FC<{
  label: string;
  value: string;
  delay: number;
}> = ({ label, value, delay }) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        alignItems: "center",
        borderBottom: "1px solid #343a3c",
        display: "grid",
        gridTemplateColumns: "70px 1fr",
        opacity: interpolate(frame, [delay, delay + 22], [0, 1], {
          easing: Easing.bezier(0.16, 1, 0.3, 1),
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
        padding: "18px 0",
        translate: interpolate(
          frame,
          [delay, delay + 22],
          ["24px 0px", "0px 0px"],
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
          color: "#72d1c5",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 42,
          fontStyle: "italic",
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "#8f938f",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 26,
          letterSpacing: 5,
          overflow: "hidden",
          whiteSpace: "nowrap",
        }}
      >
        {value}
      </div>
    </div>
  );
};

export const QuestionScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame
      eyebrow="The scale question"
      title="Can one fixed-weight cell multiply modulo p as width grows?"
      footer="Problem framing: Terence Tao, “Modular Arithmetic Challenge,” 8 June 2026."
    >
      <div
        style={{
          display: "grid",
          gap: 72,
          gridTemplateColumns: "1.2fr 0.8fr",
          marginTop: 70,
        }}
      >
        <div>
          <DecimalBand label="a" value="917582304177…641092787333" delay={30} />
          <DecimalBand label="b" value="284109653842…830177520419" delay={42} />
          <DecimalBand
            label="p"
            value="223203050209…066524860021499"
            delay={54}
          />
        </div>
        <Interactive.Div
          name="Modular product mark"
          style={{
            alignItems: "center",
            border: "1px solid #4f5758",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            opacity: interpolate(frame, [62, 92], [0, 1], {
              easing: Easing.bezier(0.16, 1, 0.3, 1),
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
            padding: 34,
            scale: interpolate(frame, [62, 92], [0.96, 1], {
              easing: Easing.spring({ damping: 200 }),
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              output: "perceptual-scale",
            }),
          }}
        >
          <div
            style={{
              color: "#ece8dd",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 76,
              lineHeight: 1,
            }}
          >
            a · b
          </div>
          <div
            style={{
              color: "#72d1c5",
              fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
              fontSize: 42,
              marginTop: 22,
            }}
          >
            mod p
          </div>
        </Interactive.Div>
      </div>
    </SceneFrame>
  );
};
