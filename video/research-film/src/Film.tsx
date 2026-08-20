import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { assertEvidenceMode, type EvidenceMode } from "./data/animationData";
import { CellScene } from "./scenes/CellScene";
import { CompressionScene } from "./scenes/CompressionScene";
import { EndScene } from "./scenes/EndScene";
import { HostedScene } from "./scenes/HostedScene";
import { LimitsScene } from "./scenes/LimitsScene";
import { ProgressScene } from "./scenes/ProgressScene";
import { QuestionScene } from "./scenes/QuestionScene";
import { RecurrenceScene } from "./scenes/RecurrenceScene";
import { ScheduleScene } from "./scenes/ScheduleScene";

export type FilmProps = {
  evidenceMode: EvidenceMode;
};

export const Film: React.FC<FilmProps> = ({ evidenceMode }) => {
  assertEvidenceMode(evidenceMode);

  return (
    <TransitionSeries>
      <TransitionSeries.Sequence durationInFrames={180} name="Question">
        <QuestionScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence durationInFrames={300} name="Recurrence">
        <RecurrenceScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence durationInFrames={330} name="Cell">
        <CellScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence
        durationInFrames={360}
        name="Three pass schedule"
      >
        <ScheduleScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence
        durationInFrames={360}
        name="Compression pilot"
      >
        <CompressionScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence
        durationInFrames={300}
        name="Hosted evaluation"
      >
        <HostedScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence
        durationInFrames={420}
        name="Full width result"
      >
        <ProgressScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence durationInFrames={420} name="Fermat result">
        <LimitsScene />
      </TransitionSeries.Sequence>
      <TransitionSeries.Transition
        presentation={fade()}
        timing={linearTiming({ durationInFrames: 10 })}
      />
      <TransitionSeries.Sequence durationInFrames={240} name="End card">
        <EndScene />
      </TransitionSeries.Sequence>
    </TransitionSeries>
  );
};
