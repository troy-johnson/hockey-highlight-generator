import React from 'react';
import {Composition, registerRoot} from 'remotion';
import {RecapGraphics} from './RecapGraphics';

const Root = () => <Composition id="RecapGraphics" component={RecapGraphics}
  width={1920} height={1080} fps={30} durationInFrames={30}
  calculateMetadata={({props}) => ({width: props.width, height: props.height,
    fps: props.fps, durationInFrames: props.durationFrames})}/>;

registerRoot(Root);
