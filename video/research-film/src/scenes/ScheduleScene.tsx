import { Easing, Interactive, interpolate, useCurrentFrame } from "remotion";
import { SceneFrame } from "../components/ResearchVisuals";

const PassCard: React.FC<{
  index: number;
  title: string;
  control: string;
  digits: string;
  result: string;
  delay: number;
}> = ({ index, title, control, digits, result, delay }) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        border: "1px solid #4f5758",
        minHeight: 180,
        opacity: interpolate(frame, [delay, delay + 28], [0, 1], {
          easing: Easing.bezier(0.16, 1, 0.3, 1),
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
        padding: "25px 28px",
        translate: interpolate(
          frame,
          [delay, delay + 28],
          ["0px 22px", "0px 0px"],
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
          alignItems: "center",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <div
          style={{
            color: "#72d1c5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 19,
            fontWeight: 700,
            letterSpacing: 2,
          }}
        >
          PASS {index}
        </div>
        <div
          style={{
            color: "#8f938f",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 16,
          }}
        >
          same weights
        </div>
      </div>
      <div
        style={{
          color: "#ece8dd",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 40,
          marginTop: 17,
        }}
      >
        {title}
      </div>
      <div
        style={{
          color: "#a6aaa5",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 18,
          lineHeight: 1.7,
          marginTop: 12,
        }}
      >
        x = {control}
        <br />
        data bits = {digits}
      </div>
      <div
        style={{
          borderTop: "1px solid #343a3c",
          color: "#d9b66f",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: 20,
          marginTop: 18,
          paddingTop: 14,
        }}
      >
        {result}
      </div>
    </div>
  );
};

export const ScheduleScene: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame
      eyebrow="The outer program"
      title="One learned transition is reused in a fixed three-pass schedule."
      footer="Lean sources and evaluation receipts are linked in the repository."
    >
      <div
        style={{
          alignItems: "center",
          display: "grid",
          gap: 34,
          gridTemplateColumns: "1fr 1fr 1fr",
          marginTop: 47,
          position: "relative",
        }}
      >
        <PassCard
          index={1}
          title="reduce a"
          control="1"
          digits="bits(a), MSB first"
          result="ā = a mod p"
          delay={38}
        />
        <PassCard
          index={2}
          title="reduce b"
          control="1"
          digits="bits(b), MSB first"
          result="b̄ = b mod p"
          delay={94}
        />
        <PassCard
          index={3}
          title="multiply"
          control="ā"
          digits="bits(b̄), MSB first"
          result="s = a · b mod p"
          delay={150}
        />
      </div>
      <Interactive.Div
        name="Proof and evidence boundary"
        style={{
          display: "grid",
          gap: 22,
          gridTemplateColumns: "1fr 1fr",
          marginTop: 30,
          opacity: interpolate(frame, [205, 245], [0, 1], {
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <div
          style={{
            border: "1px solid #3d6662",
            color: "#72d1c5",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 22,
            padding: "20px 25px",
          }}
        >
          <strong>LEAN</strong>
          <span style={{ color: "#ece8dd", marginLeft: 18 }}>
            integer recurrence + three-pass schedule
          </span>
        </div>
        <div
          style={{
            border: "1px solid #765f36",
            color: "#d9b66f",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 22,
            padding: "20px 25px",
          }}
        >
          <strong>EMPIRICAL</strong>
          <span style={{ color: "#ece8dd", marginLeft: 18 }}>
            learned cell + rollout behavior
          </span>
        </div>
      </Interactive.Div>
    </SceneFrame>
  );
};
