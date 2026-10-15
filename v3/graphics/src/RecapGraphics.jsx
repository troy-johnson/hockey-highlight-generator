import React from 'react';
import {AbsoluteFill, Easing, Img, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {OpeningRail, Stinger} from './Stingers';
import {SkateCutAccent} from './SkateCutAccent';
import {EventRail} from './EventRail';
import {FrostPane, PeriodRail} from './FrostPane';
import {FinalCard} from './FinalCard';

// Rectangular plates keep the text and team cells aligned.
const white = '#FFFFFF';
const Slab = ({children, style, cutDelay = 0}) => <div style={{display: 'flex', width: 'fit-content',
  position: 'relative',
  outline: `3px solid ${white}`,
  boxShadow: '0 10px 30px rgba(0,0,0,.45)', ...style}}>{children}<SkateCutAccent delay={cutDelay}/></div>;
const Cell = ({bg, w, h = 70, children, style}) => <div style={{background: bg,
  width: w, height: h, display: 'flex', alignItems: 'center', justifyContent: 'center',
  flexShrink: 0, overflow: 'hidden', ...style}}><div style={{
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12}}>{children}</div></div>;
const Edge = ({team, h = 70, w = 12}) => <div style={{width: w, height: h, background: team.secondary}}/>;
const TeamChip = ({team, h = 70, logoH = h - 12, pad = 18, size = 30, mirror = false}) => {
  const logo = team.logo ? <Img key="logo" src={staticFile(team.logo)} style={{width: logoH,
    height: logoH, objectFit: 'contain'}}/> : null;
  const name = <span key="name" style={{fontSize: size, fontWeight: 900, fontStyle: 'italic',
    letterSpacing: 1, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{team.acronym}</span>;
  return <Cell bg="#19212A" h={h} style={{padding: `0 ${pad}px`,
    borderBottom: `3px solid ${team.line ?? team.primary}`, boxSizing: 'border-box'}}>{mirror ? [name, logo] : [logo, name]}</Cell>;
};
// A score change is carved: a blade scratch crosses the score cell, then fades.
const Carve = ({frame, color}) => {
  const draw = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, (frame - 1) / 7)));
  const fade = Math.max(0, Math.min(1, 1 - (frame - 14) / 12));
  return <svg viewBox="0 0 64 56" style={{position: 'absolute', inset: 0, overflow: 'visible',
    pointerEvents: 'none', opacity: fade, filter: `drop-shadow(0 0 3px ${color})`}}>
    {[0, 5].map(o => <path key={o} d={`M-4 ${48 - o} Q30 ${30 - o} 68 ${10 - o}`} fill="none"
      stroke={o ? color : '#FFFFFF'} strokeWidth={o ? 1.5 : 2.2} strokeLinecap="round"
      pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw}/>)}
  </svg>;
};
const periodLabel = n => String(n).toUpperCase() === 'SO' ? 'SO' :
  ({1: '1st', 2: '2nd', 3: '3rd'}[n] || 'OT');

const ScoreSlab = ({data, teams, tokens, label, h = 70, stateWidth = 104, pop = 0}) =>
  <Slab><Edge team={teams.home} h={h}/><TeamChip team={teams.home} h={h} size={h === 90 ? 38 : 30}/>
    <Cell bg={tokens.score} h={h} w={h === 90 ? 120 : 86}><b style={{fontSize: h === 90 ? 64 : 50,
      color: '#0A0A0A', transform: `scale(${1 + pop * .35})`}}>{data.score[0]}</b></Cell>
    <Cell bg={tokens.state} h={h} w={stateWidth}><b style={{fontSize: h === 90 ? 54 : 30}}>{label}</b></Cell>
    <Cell bg={tokens.score} h={h} w={h === 90 ? 120 : 86}><b style={{fontSize: h === 90 ? 64 : 50,
      color: '#0A0A0A', transform: `scale(${1 + pop * .35})`}}>{data.score[1]}</b></Cell>
    <TeamChip team={teams.away} h={h} size={h === 90 ? 38 : 30} mirror/><Edge team={teams.away} h={h}/></Slab>;

const Bug = ({data, teams, tokens, previousScore, previousPeriod, replay, vertical}) => {
  const accent = tokens?.accent ?? '#7FD3F0';
  const f = useCurrentFrame();
  const roll = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, f / 10)));
  const enter = previousScore || data.initiallyVisible ? 1 : Easing.out(Easing.cubic)(Math.max(0, Math.min(1, f / 12)));
  const flash = (i, scoreCell = false) => {
    const scored = previousScore && data.score[i] > previousScore[i];
    const strength = scored ? Math.max(0, Math.min(1, (30 - f) / 12)) : 0;
    const primary = teams[i === 0 ? 'home' : 'away'].primary;
    const rgb = [1, 3, 5].map(n => {
      const channel = parseInt(primary.slice(n, n + 2), 16);
      return scoreCell ? channel * .9 + 255 * .1 : channel;
    });
    const base = scoreCell ? [35, 43, 52] : [20, 27, 35];
    const displayed = rgb.map((v, j) => Math.round(v * strength + base[j] * (1 - strength)));
    const linear = displayed.map(v => {
      const s = v / 255;
      return s <= .04045 ? s / 12.92 : ((s + .055) / 1.055) ** 2.4;
    });
    const luminance = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
    return {background: scoreCell ? `rgb(${displayed.join(',')})` : `rgba(${rgb.join(',')},${strength})`,
      color: luminance > .179 ? '#0A0A0A' : '#EEF3F6'};
  };
  const number = i => {
    const changed = previousScore && previousScore[i] !== data.score[i];
    return <div style={{width: 64, height: 56, overflow: 'hidden', position: 'relative',
      ...flash(i, true), fontSize: 36, fontWeight: 800, fontVariantNumeric: 'tabular-nums'}}>
      <div style={{transform: changed ? `translateY(${-roll * 56}px)` : undefined}}>
        {(changed ? [previousScore[i], data.score[i]] : [data.score[i]]).map((value, j) =>
          <div key={j} style={{height: 56, display: 'flex', justifyContent: 'center',
            alignItems: 'center'}}>{value}</div>)}
      </div>
      {changed && data.score[i] > previousScore[i] && <Carve frame={f}
        color={teams[i === 0 ? 'home' : 'away'].line ?? '#EEF3F6'}/>}
      {changed && data.score[i] < previousScore[i] && <div style={{position: 'absolute',
        bottom: 0, height: 3, width: '100%', background: '#EEF3F6',
        opacity: Math.max(0, 1 - f / 18)}}/>}
    </div>;
  };
  const periodCell = () => {
    // A new period rolls in and flashes generic ice-white, like a score change; it is not team-specific.
    const changed = previousPeriod != null && previousPeriod !== data.period;
    const strength = changed ? Math.max(0, Math.min(1, (30 - f) / 12)) : 0;
    const labels = changed ? [periodLabel(previousPeriod), periodLabel(data.period)] : [periodLabel(data.period)];
    return <div style={{width: 98, height: 56, overflow: 'hidden', position: 'relative',
      background: `rgba(238,243,246,${strength})`, fontSize: 22, fontWeight: 650,
      color: strength > .35 ? '#0A0A0A' : '#9FB2C1'}}>
      <div style={{transform: changed ? `translateY(${-roll * 56}px)` : undefined}}>
        {labels.map((label, j) => <div key={j} style={{height: 56, display: 'flex',
          justifyContent: 'center', alignItems: 'center'}}>{label}</div>)}
      </div>
    </div>;
  };
  const chip = side => <div style={{height: 56, width: 154, position: 'relative',
    display: 'flex', alignItems: 'center', gap: 10, paddingLeft: 14, boxSizing: 'border-box',
    ...flash(side === 'home' ? 0 : 1)}}>
    {teams[side].logo && <Img src={staticFile(teams[side].logo)} style={{width: 36,
      height: 36, objectFit: 'contain'}}/>}
    <span style={{fontSize: Math.min(24, (teams[side].logo ? 86 : 126) /
      Math.max(1, teams[side].acronym.length * .8)), fontWeight: 750, letterSpacing: .5,
      whiteSpace: 'nowrap'}}>{teams[side].acronym}</span>
    <div style={{position: 'absolute', bottom: 0, left: 0, right: 0, borderBottom: `3px solid ${teams[side].line ?? teams[side].primary}`,
      boxSizing: 'border-box', pointerEvents: 'none'}}/>
  </div>;
  return <div style={{position: 'absolute', zIndex: 10, left: vertical ? (1080 - 535 * 1.3) / 2 : 96,
    top: vertical ? 220 : 54, color: '#EEF3F6', transformOrigin: 'top left',
    opacity: enter, transform: `translateY(${(1 - enter) * -6}px) scale(${vertical ? 1.3 : 1})`,
    filter: 'drop-shadow(0 5px 12px rgba(0,0,0,.4))'}}>
    <div style={{display: 'flex', height: 56, background: 'linear-gradient(#19212A, #0E1319)',
      borderTop: '1px solid rgba(238,243,246,.32)', boxSizing: 'border-box'}}>
      {chip('home')}{number(0)}
      <div style={{width: 1, background: 'rgba(238,243,246,.15)'}}/>
      {chip('away')}{number(1)}
      {periodCell()}
    </div>
    {replay && <div style={{position: 'absolute', top: 56, right: 0, width: 98,
      height: 27, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#0E1319', borderBottom: `2px solid ${accent}`, fontSize: 15,
      letterSpacing: 1.5, fontWeight: 700}}>REPLAY</div>}
  </div>;
};

const Motion = ({duration, children, style, delay = 0, drop = false}) => {
  delay = Math.min(delay, Math.floor(duration / 3));
  const f = useCurrentFrame() - delay;
  duration -= delay;
  const enter = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, f / Math.min(18, duration / 3))));
  const exit = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, (duration - f) / Math.min(14, duration / 3))));
  const reveal = enter * exit;
  return <div style={{clipPath: drop ? `inset(-90px -90px calc(${(1 - reveal) * 115}% - ${reveal * 90}px) -90px)` :
    `inset(-90px calc(${(1 - reveal) * 115}% - ${reveal * 90}px) -90px -90px)`, ...style}}>{children}</div>;
};

const Particles = ({teams}) => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{overflow: 'hidden', pointerEvents: 'none'}}>
    {Array.from({length: 24}, (_, i) => <div key={i} style={{position: 'absolute',
      left: `${((i * 47 + f * (i % 2 ? .08 : -.06)) % 110 + 110) % 110 - 5}%`,
      top: `${(i * 29) % 100}%`, width: 20 + i % 5 * 12, height: 4,
      background: i % 2 ? teams.home.secondary : teams.away.primary,
      opacity: .22, filter: `blur(${2 + i % 3}px)`, transform: 'skewX(-16deg)'}}/>)}
  </AbsoluteFill>;
};

const Period = ({data, teams, tokens, duration}) => <Motion duration={duration}
  style={{position: 'absolute', left: 0, bottom: 96, width: 1920, display: 'flex', justifyContent: 'center'}}>
  <ScoreSlab {...{data, teams, tokens}} h={76} stateWidth={240} label={periodLabel(data.period) === 'SO' ?
    'SHOOTOUT' : periodLabel(data.period) === 'OT' ? 'OVERTIME' : `${periodLabel(data.period)} PERIOD`}/>
</Motion>;

const Event = ({event, props}) => {
  const absoluteFrame = event.startFrame + useCurrentFrame();
  const common = {data: event.data, teams: props.teams, tokens: props.tokens, duration: event.durationFrames};
  if (props.designPreview && event.data.status === 'design_sample') {
    common.teams = Object.fromEntries(['home', 'away'].map(side => [side, {
      name: `Sample ${side === 'home' ? 'Home' : 'Away'}`, acronym: side === 'home' ? 'HME' : 'AWY',
      primary: '#607D91', secondary: '#EEF3F6', line: '#9FB2C1', logo: null,
    }]));
  }
  if (event.kind === 'scorebug') {
    const previous = props.events.find(e => e.kind === 'scorebug' &&
      e.startFrame + e.durationFrames === event.startFrame && e.data.status === event.data.status);
    const replay = props.events.some(e => e.kind === 'replay' && e.startFrame <= absoluteFrame &&
      absoluteFrame < e.startFrame + e.durationFrames);
    return <><Bug {...common} previousScore={previous?.data.score}
      previousPeriod={previous?.data.period} replay={replay}
      vertical={props.height > props.width}/>
      {props.designPreview && event.data.status === 'design_sample' &&
        <div style={{position: 'absolute', zIndex: 20, left: 96,
          top: props.height > props.width ? 360 : 180, color: '#9FB2C1', fontSize: 24,
          fontWeight: 650, letterSpacing: 2}}>PERIOD CHANGE · DESIGN SAMPLE</div>}</>;
  }
  if (['goal', 'penalty', 'call', 'period'].includes(event.kind)) return <>
    {event.kind === 'period' ? <PeriodRail {...common} vertical={props.height > props.width}/> :
      <EventRail {...common} kind={event.kind} vertical={props.height > props.width}/>}
    {event.kind !== 'period' && event.data.status === 'design_sample' && <div style={{position: 'absolute',
      left: 96, top: props.height > props.width ? 360 : 180, color: '#9FB2C1',
      fontSize: 24, fontWeight: 650, letterSpacing: 2}}>
      {event.kind === 'period' ? 'PERIOD CHANGE' : event.kind === 'call' ? 'FICTIONAL CALL' : 'PENALTY'} · DESIGN SAMPLE
    </div>}
  </>;
  if (event.kind === 'replay_pane') return <FrostPane vertical={props.height > props.width} replay
    team={props.focusSide ? props.teams[props.focusSide] : null}/>;
  if (event.kind === 'replay') {
    if (props.events.some(e => e.kind === 'scorebug' && e.startFrame <= absoluteFrame &&
      absoluteFrame < e.startFrame + e.durationFrames)) return null;
    if (props.events.some(e => ['goal', 'penalty'].includes(e.kind) &&
      e.startFrame <= absoluteFrame && absoluteFrame < e.startFrame + e.durationFrames)) return null;
    const f = absoluteFrame - event.startFrame;
    return <div style={{position: 'absolute', left: 76, top: 44,
      opacity: Math.min(1, f / 10, (event.durationFrames - f) / 10)}}>
    <Slab style={{outline: 'none'}}><div style={{width: 10, height: 48, background: props.tokens.label}}/>
      <Cell bg={props.tokens.state} w={170} h={48}><b style={{fontSize: 24, letterSpacing: 4}}>REPLAY</b></Cell></Slab>
  </div>;
  }
  if (event.kind === 'final') return <>
    <FinalCard {...common} vertical={props.height > props.width}/>
    {props.designPreview && event.data.goalieSample === true && <div style={{position: 'absolute', zIndex: 31,
      right: 40, top: 24, color: '#9FB2C1', fontSize: 20, fontWeight: 650, letterSpacing: 2}}>
      GOALIE NUMBERS · DESIGN SAMPLE</div>}
  </>;
  if (event.kind === 'open_stinger') return <Stinger {...common}/>;
  if (event.kind === 'matchup_rail') return <OpeningRail {...common}/>;
  if (event.kind === 'period_wipe') return <FrostPane vertical={props.height > props.width}
    team={event.data.status === 'design_sample' ? null :
      common.teams[event.data.team || props.focusSide]}/>;
  throw new Error(`Unknown graphics event ${event.kind}`);
};

export const RecapGraphics = props => <AbsoluteFill style={{fontFamily: 'Inter, Arial, sans-serif',
  fontVariantNumeric: 'tabular-nums', color: white}}>
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
