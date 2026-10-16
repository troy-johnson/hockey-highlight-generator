// PROTOTYPE — Graphics Package variants. Open `npm run studio` and pick a composition in the left panel.
import React from 'react';
import {registerRoot, Composition} from 'remotion';
import {Storyboard, GAME, T, FPS} from './shared';
import {VariantA} from './VariantA';
import {VariantB} from './VariantB';
import {VariantC} from './VariantC';
import {VariantD, GAME_D} from './VariantD';
import {Open3DReview, Period3DReview, PeriodSkate3DReview} from './Stinger3D';

const bind = (V) => (props) => <Storyboard V={V} {...props} />;
const VARIANTS = [['A-Broadcast', bind(VariantA)], ['B-Glass', bind(VariantB)], ['C-IceEdge', bind(VariantC)], ['D-IcePakSlab', bind(VariantD), GAME_D]];
const Root = () => (
  <>
    {VARIANTS.flatMap(([id, V, game = GAME]) => [
      ['Focus', 'home'], ['Neutral', 'neutral'],
    ].map(([p, perspective]) => (
      <Composition key={`${id}-${p}`} id={`${id}-${p}`} component={V} width={1920} height={1080} fps={FPS}
        durationInFrames={T.total} defaultProps={{game, perspective}} />
    )))}
    <Composition id="Stinger3D-Open" component={Open3DReview} width={1920} height={1080} fps={FPS} durationInFrames={180} />
    <Composition id="Stinger3D-Period" component={Period3DReview} width={1920} height={1080} fps={FPS} durationInFrames={120} />
    <Composition id="Stinger3D-PeriodSkate" component={PeriodSkate3DReview} width={1920} height={1080} fps={FPS} durationInFrames={120} />
  </>
);
registerRoot(Root);
