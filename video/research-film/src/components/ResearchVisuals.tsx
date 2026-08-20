import type { ReactNode } from "react";
import {
  AbsoluteFill,
  Easing,
  Interactive,
  interpolate,
  useCurrentFrame,
} from "remotion";

export const SceneFrame: React.FC<{
  eyebrow: string;
  title: string;
  children: ReactNode;
  footer?: string;
}> = ({ eyebrow, title, children, footer }) => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#111315",
        backgroundImage:
          "radial-gradient(circle at 76% 18%, rgba(114,209,197,0.055), transparent 30%), linear-gradient(rgba(234,230,220,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(234,230,220,0.035) 1px, transparent 1px)",
        backgroundSize: "auto, 72px 72px, 72px 72px",
        color: "#ece8dd",
        fontFamily:
          'Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
        padding: "92px 112px 84px",
      }}
    >
      <Interactive.Div
        name="Scene eyebrow"
        style={{
          color: "#72d1c5",
          fontSize: 24,
          fontWeight: 700,
          letterSpacing: 4,
          lineHeight: 1,
          textTransform: "uppercase",
          opacity: interpolate(frame, [4, 24], [0, 1], {
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [4, 24], ["0px 14px", "0px 0px"], {
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {eyebrow}
      </Interactive.Div>
      <Interactive.Div
        name="Scene title"
        style={{
          color: "#ece8dd",
          fontFamily: '"Iowan Old Style", Baskerville, Georgia, serif',
          fontSize: 74,
          fontWeight: 500,
          letterSpacing: -2.5,
          lineHeight: 1.04,
          marginTop: 20,
          maxWidth: 1450,
          opacity: interpolate(frame, [10, 34], [0, 1], {
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
          translate: interpolate(frame, [10, 34], ["0px 18px", "0px 0px"], {
            easing: Easing.bezier(0.16, 1, 0.3, 1),
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {title}
      </Interactive.Div>
      <div style={{ flex: 1, minHeight: 0 }}>{children}</div>
      {footer ? (
        <Interactive.Div
          name="Scene source note"
          style={{
            bottom: 40,
            color: "#8f938f",
            fontFamily:
              'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
            fontSize: 20,
            left: 112,
            letterSpacing: 0.2,
            position: "absolute",
          }}
        >
          {footer}
        </Interactive.Div>
      ) : null}
    </AbsoluteFill>
  );
};

export const BitTape: React.FC<{
  bits: string;
  label: string;
  activeIndex?: number;
  compact?: boolean;
}> = ({ bits, label, activeIndex = -1, compact = false }) => {
  return (
    <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
      <div
        style={{
          color: "#a6aaa5",
          fontFamily:
            'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
          fontSize: compact ? 20 : 24,
          minWidth: compact ? 54 : 78,
          textAlign: "right",
        }}
      >
        {label}
      </div>
      <div style={{ display: "flex", gap: compact ? 5 : 8 }}>
        {bits.split("").map((bit, index) => (
          <div
            key={`${label}-${index}`}
            style={{
              alignItems: "center",
              backgroundColor:
                index === activeIndex
                  ? "#bb654f"
                  : bit === "1"
                    ? "#255d59"
                    : "#191d1f",
              border: `1px solid ${
                index === activeIndex
                  ? "#e48a70"
                  : bit === "1"
                    ? "#72d1c5"
                    : "#3d4344"
              }`,
              color: index === activeIndex ? "#fff4e9" : "#ece8dd",
              display: "flex",
              fontFamily:
                'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
              fontSize: compact ? 18 : 25,
              height: compact ? 34 : 46,
              justifyContent: "center",
              width: compact ? 26 : 38,
            }}
          >
            {bit}
          </div>
        ))}
      </div>
    </div>
  );
};

export const StatusPill: React.FC<{
  label: string;
  tone: "exact" | "failed" | "progress" | "neutral";
}> = ({ label, tone }) => {
  const color =
    tone === "exact"
      ? "#72d1c5"
      : tone === "failed"
        ? "#e48a70"
        : tone === "progress"
          ? "#d9b66f"
          : "#a6aaa5";
  return (
    <div
      style={{
        border: `1px solid ${color}`,
        borderRadius: 999,
        color,
        fontFamily:
          'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
        fontSize: 18,
        fontWeight: 700,
        letterSpacing: 1.2,
        padding: "9px 15px",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </div>
  );
};

export const Rule: React.FC<{ color?: string }> = ({ color = "#3d4344" }) => (
  <div style={{ backgroundColor: color, height: 1, width: "100%" }} />
);

export const HashLabel: React.FC<{ children: ReactNode }> = ({ children }) => (
  <div
    style={{
      color: "#8f938f",
      fontFamily:
        'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
      fontSize: 17,
      letterSpacing: 0.3,
    }}
  >
    {children}
  </div>
);
