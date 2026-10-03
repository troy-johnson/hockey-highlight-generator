import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';

// Variant D: skewed slabs, semantic colors, and mirrored team tiles.
const white = '#FFFFFF';
const Slab = ({children, style}) => <div style={{display: 'flex', width: 'fit-content',
  transform: 'skewX(-16deg)', outline: `3px solid ${white}`,
  boxShadow: '0 10px 30px rgba(0,0,0,.45)', ...style}}>{children}</div>;
const Cell = ({bg, w, h = 70, children, style}) => <div style={{background: bg,
  width: w, height: h, display: 'flex', alignItems: 'center', justifyContent: 'center',
  flexShrink: 0, overflow: 'hidden', ...style}}><div style={{transform: 'skewX(16deg)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12}}>{children}</div></div>;
const Edge = ({team, h = 70}) => <div style={{width: 12, height: h, background: team.secondary}}/>;
const TeamChip = ({team, h = 70, size = 30, mirror = false}) => {
  const logo = team.logo ? <Img key="logo" src={staticFile(team.logo)} style={{width: h - 12,
    height: h - 12, objectFit: 'contain'}}/> : null;
  const name = <span key="name" style={{fontSize: size, fontWeight: 900, fontStyle: 'italic',
    letterSpacing: 1, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{team.acronym}</span>;
  return <Cell bg={team.primary} h={h} style={{padding: '0 22px'}}>{mirror ? [name, logo] : [logo, name]}</Cell>;
};
const periodLabel = n => n > 3 ? 'OT' : `P${n}`;

const ScoreSlab = ({data, teams, tokens, label, h = 70, stateWidth = 104, pop = 0}) =>
  <Slab><Edge team={teams.home} h={h}/><TeamChip team={teams.home} h={h} size={h === 90 ? 38 : 30}/>
    <Cell bg={tokens.score} h={h} w={h === 90 ? 120 : 86}><b style={{fontSize: h === 90 ? 64 : 50,
      color: '#0A0A0A', transform: `scale(${1 + pop * .35})`}}>{data.score[0]}</b></Cell>
    <Cell bg={tokens.state} h={h} w={stateWidth}><b style={{fontSize: h === 90 ? 54 : 30}}>{label}</b></Cell>
    <Cell bg={tokens.score} h={h} w={h === 90 ? 120 : 86}><b style={{fontSize: h === 90 ? 64 : 50,
      color: '#0A0A0A', transform: `scale(${1 + pop * .35})`}}>{data.score[1]}</b></Cell>
    <TeamChip team={teams.away} h={h} size={h === 90 ? 38 : 30} mirror/><Edge team={teams.away} h={h}/></Slab>;

const Bug = ({data, teams, tokens, changed}) => {
  const f = useCurrentFrame();
  const pop = changed ? Math.max(0, 1 - Math.abs(f - 6) / 10) : 0;
  return <div style={{position: 'absolute', left: 76, top: 44}}>
    <ScoreSlab {...{data, teams, tokens}} label={periodLabel(data.period)} pop={pop}/>
  </div>;
};

const Motion = ({duration, children, style, delay = 0, drop = false}) => {
  delay = Math.min(delay, Math.floor(duration / 3));
  const f = useCurrentFrame() - delay;
  duration -= delay;
  const enter = Math.max(0, Math.min(1, (f + 1) / Math.min(16, duration / 3)));
  const exit = Math.min(1, (duration - f) / Math.min(10, duration / 3));
  const reveal = enter * exit;
  return <div style={{clipPath: drop ? `inset(-90px -90px calc(${(1 - reveal) * 115}% - ${reveal * 90}px) -90px)` :
    `inset(-90px calc(${(1 - reveal) * 115}% - ${reveal * 90}px) -90px -90px)`, ...style}}>{children}</div>;
};

const Goal = ({data, teams, tokens, duration, compact}) => {
  const team = teams[data.team];
  const f = useCurrentFrame();
  const word = interpolate(f, [8, 26], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp',
    easing: Easing.out(Easing.back(1.3))});
  if (compact) return <Motion duration={duration} style={{position: 'absolute', left: 76, top: 160}}>
    <Slab><Edge team={team}/><TeamChip team={team}/>
      <Cell bg={tokens.hero} w={170}><b style={{fontSize: 42, fontStyle: 'italic', color: '#0A0A0A'}}>GOAL</b></Cell>
      <Cell bg={tokens.info} style={{padding: '0 30px'}}><b style={{fontSize: 30}}>{data.num} {data.scorer}</b></Cell>
      <Edge team={team}/></Slab>
  </Motion>;
  const h = 128;
  return <Motion duration={duration} style={{position: 'absolute', left: 86, top: 160}}>
    <Slab><Edge team={team} h={h}/><TeamChip team={team} h={h} size={48}/>
      <Cell bg={tokens.hero} h={h} w={500}><b style={{fontSize: 112,
        fontWeight: 900, fontStyle: 'italic', letterSpacing: 4, color: '#0A0A0A', opacity: word,
        transform: `translateX(${(1 - word) * 120}px) scale(${.85 + .15 * word})`}}>GOAL</b></Cell>
      <Edge team={team} h={h}/></Slab>
    <Motion duration={duration} delay={18} drop><Slab style={{marginLeft: 150, marginTop: 10}}>
      <Cell bg={team.primary} h={62} w={80}><b style={{fontSize: 32}}>{data.num}</b></Cell>
      <Cell bg={tokens.info} h={62} style={{padding: '0 30px', maxWidth: 1350}}>
        <b style={{fontSize: 32}}>{data.scorer}</b>
        <b style={{fontSize: 22, color: tokens.label}}>{data.type}</b>
        <b style={{fontSize: 22, color: tokens.label}}>{data.assists.length ? 'ASSISTS:' : 'UNASSISTED'}</b>
        <span style={{fontSize: 24}}>{data.assists.join(', ')}</span>
      </Cell></Slab></Motion>
  </Motion>;
};

const Penalty = ({data, teams, tokens, duration}) => {
  const team = teams[data.team];
  return <Motion duration={duration} style={{position: 'absolute', left: 76, top: 160}}>
    <Slab><Cell bg={team.primary} w={96}><b style={{fontSize: 40}}>{data.num}</b></Cell>
      <Cell bg={tokens.info} style={{padding: '0 34px'}}><b style={{fontSize: 34}}>{data.name}</b></Cell>
      <Cell bg={tokens.state} w={150}><b style={{fontSize: 36}}>{/^\d+$/.test(String(data.minutes)) ? `${data.minutes}:00` : data.minutes}</b></Cell>
      <Cell bg={tokens.info} style={{padding: '0 34px'}}><b style={{fontSize: 28}}>{data.infraction}</b></Cell>
      <TeamChip team={team}/><Edge team={team}/></Slab>
  </Motion>;
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
  style={{position: 'absolute', left: 0, bottom: 90, width: 1920, display: 'flex', justifyContent: 'center'}}>
  <ScoreSlab {...{data, teams, tokens}} stateWidth={240} label={data.period > 3 ? 'OVERTIME' : `PERIOD ${data.period}`}/>
</Motion>;

const FinalTable = ({team, table, tokens}) => {
  const rowHeight = Math.min(46, 650 / Math.max(1, table.skaters.length + table.goalies.length + 3));
  const size = Math.min(28, rowHeight * .65);
  return <div style={{width: 840, background: tokens.info, padding: '18px 24px', boxSizing: 'border-box'}}>
    <div style={{background: team.primary, borderLeft: `12px solid ${team.secondary}`, padding: '10px 20px',
      fontSize: 32, fontWeight: 900, marginBottom: 12}}>{team.name}</div>
    <table style={{width: '100%', borderCollapse: 'collapse', fontSize: size, fontVariantNumeric: 'tabular-nums'}}>
      <thead style={{color: tokens.label}}><tr>{['#', 'PLAYER', 'G', 'A', 'PTS', 'PIM'].map(t => <th key={t}
        style={{textAlign: t === 'PLAYER' ? 'left' : 'center', height: rowHeight}}>{t}</th>)}</tr></thead>
      <tbody>{table.skaters.map((p, i) => <tr key={p.num} style={{background: i % 2 ? '#1C1C1C' : 'transparent', height: rowHeight}}>
        {[p.num, p.name, p.g, p.a, p.pts, p.pim ?? '—'].map((v, j) => <td key={j} style={{textAlign: j === 1 ? 'left' : 'center',
          maxWidth: j === 1 ? 360 : undefined, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis'}}>{v}</td>)}</tr>)}</tbody>
    </table>
    {table.goalies.length ? table.goalies.map(p => <div key={p.num} style={{fontSize: size, paddingTop: 12}}>
      {p.num} {p.name} <b style={{color: tokens.label}}>SAVES</b> {p.saves}</div>) :
      <div style={{fontSize: 22, color: '#D9D9D9', paddingTop: 12}}>Goalie saves unavailable</div>}
  </div>;
};

const Final = ({data, teams, tokens, duration}) => <AbsoluteFill style={{background: 'rgba(13,27,42,.9)'}}>
  <Particles teams={teams}/><Motion duration={duration} style={{position: 'absolute', left: 0, top: 100, width: 1920,
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 34}}>
    <ScoreSlab {...{data, teams, tokens}} h={90} stateWidth={250} label="FINAL"/>
    <div style={{display: 'flex', gap: 40}}>{['home', 'away'].map(side =>
      <FinalTable key={side} team={teams[side]} table={data.table[side]} tokens={tokens}/>)}</div>
  </Motion>
</AbsoluteFill>;

const Event = ({event, props}) => {
  const common = {data: event.data, teams: props.teams, tokens: props.tokens, duration: event.durationFrames};
  if (event.kind === 'scorebug') {
    const previous = props.events.find(e => e.kind === 'scorebug' &&
      e.startFrame + e.durationFrames === event.startFrame);
    return <Bug {...common} changed={previous && previous.data.score.some((s, i) => s !== event.data.score[i])}/>;
  }
  if (event.kind === 'goal') return <Goal {...common} compact={props.perspective === 'focus' &&
    props.focusSide !== null && event.data.team !== props.focusSide}/>;
  if (event.kind === 'penalty') return <Penalty {...common}/>;
  if (event.kind === 'period') return <><Particles teams={props.teams}/><Period {...common}/></>;
  if (event.kind === 'final') return <Final {...common}/>;
  throw new Error(`Unknown graphics event ${event.kind}`);
};

export const RecapGraphics = props => <AbsoluteFill style={{fontFamily: 'Inter, Arial, sans-serif', color: white}}>
  <style>{`@font-face {font-family: Inter; src: url('${staticFile('Inter.ttf')}'); font-weight: 100 900;}
    @font-face {font-family: Inter; src: url('${staticFile('Inter-Italic.ttf')}'); font-weight: 100 900; font-style: italic;}`}</style>
  <div style={{position: 'absolute', width: 1920, height: 1080, transformOrigin: '0 0',
    transform: `scale(${props.width / 1920}, ${props.height / 1080})`}}>
    {props.events.map((event, i) => <Sequence key={i} from={event.startFrame} durationInFrames={event.durationFrames}
      layout="none"><Event event={event} props={props}/></Sequence>)}
  </div>
</AbsoluteFill>;
