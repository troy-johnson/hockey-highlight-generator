import React from 'react';
import {AbsoluteFill, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {OpeningRail, Stinger} from './Stingers';
import {FrostPane, PeriodRail} from './FrostPane';
import {FinalCard} from './FinalCard';
import {RegionalCard, StackedBug} from './IcePakGraphics';

const Event = ({event, props}) => {
  const absoluteFrame = event.startFrame + useCurrentFrame();
  const vertical = props.height > props.width;
  const score = props.events.find(e => e.kind === 'scorebug' && e.startFrame <= absoluteFrame &&
    absoluteFrame < e.startFrame + e.durationFrames)?.data.score;
  const common = {data: {...event.data, score: event.data.score ?? score}, teams: props.teams,
    tokens: props.tokens, duration: event.durationFrames, vertical};
  if (props.designPreview && event.data.status === 'design_sample') {
    common.teams = Object.fromEntries(['home', 'away'].map(side => [side, {
      name: `Sample ${side === 'home' ? 'Home' : 'Away'}`, acronym: side === 'home' ? 'HME' : 'AWY',
      primary: '#607D91', secondary: '#EEF3F6', line: '#9FB2C1', logo: null,
    }]));
  }
  if (event.kind === 'scorebug') return <StackedBug event={event} props={{...props, teams: common.teams}} absoluteFrame={absoluteFrame} vertical={vertical}/>;
  if (event.kind === 'event_bar' || event.kind === 'replay') return null;
  if (['goal', 'penalty', 'call', 'period'].includes(event.kind)) return <>
    {event.kind === 'period' ? <PeriodRail {...common}/> : <RegionalCard {...common} kind={event.kind}/>}
    {event.data.status === 'design_sample' && <div style={{position: 'absolute', left: 96,
      top: vertical ? 400 : 240, color: '#9FB2C1', fontSize: 24, fontWeight: 650, letterSpacing: 2}}>
      {event.kind.toUpperCase()} · DESIGN SAMPLE</div>}
  </>;
  if (event.kind === 'replay_pane') return <FrostPane vertical={vertical} replay
    team={props.perspective === 'focus' && props.focusSide ? props.teams[props.focusSide] : null}/>;
  if (event.kind === 'final') return <FinalCard {...common}/>;
  if (event.kind === 'open_stinger') return <Stinger {...common}/>;
  if (event.kind === 'matchup_rail') return <OpeningRail {...common}/>;
  if (event.kind === 'period_wipe') return <FrostPane vertical={vertical}
    team={event.data.status === 'design_sample' || props.perspective !== 'focus' ? null :
      common.teams[event.data.team || props.focusSide]}/>;
  throw new Error(`Unknown graphics event ${event.kind}`);
};

export const RecapGraphics = props => <AbsoluteFill style={{fontFamily: 'Inter, Arial, sans-serif',
  fontVariantNumeric: 'tabular-nums', color: '#EEF3F6'}}>
  <style>{`@font-face {font-family: Inter; src: url('${staticFile('Inter.ttf')}'); font-weight: 100 900;}
    @font-face {font-family: Inter; src: url('${staticFile('Inter-Italic.ttf')}'); font-weight: 100 900; font-style: italic;}`}</style>
  <div style={{position: 'absolute', width: props.height > props.width ? 1080 : 1920,
    height: props.height > props.width ? 1920 : 1080, transformOrigin: '0 0',
    transform: props.height > props.width ? `scale(${props.width / 1080}, ${props.height / 1920})` :
      `scale(${props.width / 1920}, ${props.height / 1080})`}}>
    {props.events.map((event, i) => <Sequence key={i} from={event.startFrame} durationInFrames={event.durationFrames}
      layout="none"><Event event={event} props={props}/></Sequence>)}
  </div>
</AbsoluteFill>;
