import type { CalculateMetadataFunction } from "remotion";
import { Composition, Folder, Still } from "remotion";
import { Film, type FilmProps } from "./Film";
import { Loop, type LoopProps } from "./Loop";
import { Poster } from "./Poster";
import { assertEvidenceMode } from "./data/animationData";
import { CellScene } from "./scenes/CellScene";
import { CompressionScene } from "./scenes/CompressionScene";
import { EndScene } from "./scenes/EndScene";
import { HostedScene } from "./scenes/HostedScene";
import { LimitsScene } from "./scenes/LimitsScene";
import { ProgressScene } from "./scenes/ProgressScene";
import { QuestionScene } from "./scenes/QuestionScene";
import { RecurrenceScene } from "./scenes/RecurrenceScene";
import { ScheduleScene } from "./scenes/ScheduleScene";

const calculateFilmMetadata: CalculateMetadataFunction<FilmProps> = ({
  props,
}) => {
  assertEvidenceMode(props.evidenceMode);
  return {};
};

const calculateLoopMetadata: CalculateMetadataFunction<LoopProps> = ({
  props,
}) => {
  assertEvidenceMode(props.evidenceMode);
  return {};
};

export const MiniNeuralHornerCompositions: React.FC = () => {
  return (
    <>
      <Composition
        id="NeuralHorner-Research-Film"
        component={Film}
        durationInFrames={2830}
        fps={30}
        width={1920}
        height={1080}
        calculateMetadata={calculateFilmMetadata}
        defaultProps={{ evidenceMode: "current" }}
      />
      <Composition
        id="NeuralHorner-Research-Loop"
        component={Loop}
        durationInFrames={450}
        fps={30}
        width={1920}
        height={1080}
        calculateMetadata={calculateLoopMetadata}
        defaultProps={{ evidenceMode: "current" }}
      />
      <Still
        id="NeuralHorner-Research-Poster"
        component={Poster}
        width={1920}
        height={1080}
      />
      <Folder name="NeuralHorner-Scenes">
        <Composition
          id="Scene-Question"
          component={QuestionScene}
          durationInFrames={180}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Recurrence"
          component={RecurrenceScene}
          durationInFrames={300}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Cell"
          component={CellScene}
          durationInFrames={330}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Schedule"
          component={ScheduleScene}
          durationInFrames={360}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Compression"
          component={CompressionScene}
          durationInFrames={360}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Hosted"
          component={HostedScene}
          durationInFrames={300}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Progress"
          component={ProgressScene}
          durationInFrames={420}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-Limits"
          component={LimitsScene}
          durationInFrames={420}
          fps={30}
          width={1920}
          height={1080}
        />
        <Composition
          id="Scene-End"
          component={EndScene}
          durationInFrames={240}
          fps={30}
          width={1920}
          height={1080}
        />
      </Folder>
    </>
  );
};
