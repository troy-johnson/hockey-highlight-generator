import React from 'react';
import {Composition, registerRoot} from 'remotion';
import {RecapGraphics} from './RecapGraphics';
import {OpenStinger, PeriodWipe} from './Stingers';

const Root = () => <>{[['RecapGraphics', RecapGraphics], ['OpenStinger', OpenStinger],
  ['PeriodWipe', PeriodWipe]].map(([id, component]) => <Composition key={id} id={id} component={component}
  width={1920} height={1080} fps={30} durationInFrames={30}
  calculateMetadata={({props}) => ({width: props.width, height: props.height,
    fps: props.fps, durationInFrames: props.durationFrames})}/>)}</>;

registerRoot(Root);
