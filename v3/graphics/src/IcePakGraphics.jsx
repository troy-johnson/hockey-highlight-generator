import React from 'react';
import {Easing, Img, staticFile, useCurrentFrame} from 'remotion';
import {scoreFlash} from './score_flash.mjs';

// Shared plates and motion from the approved v21 regional package.
export const plate = '#0E1319';
export const cell = '#19212A';
export const grey = '#232B34';
export const ink = '#EEF3F6';
export const frost = '#9FB2C1';
export const clamp = v => Math.max(0, Math.min(1, v));
export const io = Easing.inOut(Easing.cubic);
export const enter = (f, start, duration = 8) => io(clamp((f - start) / duration));
export const slide = k => ({opacity: k, transform: `translateX(${(1 - k) * -40}px)`});
export const glow = (team, end, direction = 90) => `linear-gradient(${direction}deg, ${team.primary}D8 0%, ${cell} ${Math.min(100, end * 1.3)}%)`;
export const periodLabel = n => String(n).toUpperCase() === 'SO' ? 'SO' : ({1: '1st', 2: '2nd', 3: '3rd'}[n] || 'OT');
export const dateLabel = date => /^\d{4}-\d{2}-\d{2}$/.test(date || '') ?
  new Date(`${date}T00:00:00Z`).toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC'}).toUpperCase() : date;

let textContext;
// Measure the loaded Inter font. Preserve the complete text, including surnames.
export const fit = (text, base, width, spacing = 0) => {
  const value = String(text ?? '');
  if (typeof document !== 'undefined' && document.fonts.check('850 32px Inter')) {
    textContext ??= document.createElement('canvas').getContext('2d');
    textContext.font = `850 ${base}px Inter`;
    const available = Math.max(1, width - Math.max(0, [...value].length - 1) * spacing);
    return Math.min(base, base * available / Math.max(1, textContext.measureText(value).width));
  }
  return Math.min(base, width / Math.max(1, [...value].length * 1.1));
};
export const Crest = ({team, size}) => team?.logo ? <Img src={staticFile(team.logo)}
  style={{width: size, height: size, objectFit: 'contain', flexShrink: 0}}/> :
  <span style={{width: size, flexShrink: 0, textAlign: 'center', fontSize: fit(team?.acronym, 18, size)}}>{team?.acronym}</span>;
export const Accent = ({color, p, h = 4}) => <div style={{position: 'absolute', left: `${(1 - p) * 50}%`, bottom: 0,
  height: h, width: `${p * 100}%`, background: color}}/>;

const TEAM = 170, SCORE = 54, PERIOD = 78, WIDTH = TEAM + SCORE + PERIOD;
const ScoreCell = ({team, value, previous, local}) => {
  const changed = previous !== undefined && local >= 0 && local < 30 && previous !== value;
  const roll = changed ? io(clamp(local / 10)) : 1;
  const flash = changed ? (local < 18 ? 1 : io(clamp((30 - local) / 12))) : 0;
  const color = value < previous ? ink : team.primary;
  return <div style={{width: SCORE, height: 44, overflow: 'hidden', position: 'relative',
    fontSize: 28, fontWeight: 800, ...scoreFlash(color, flash)}}>
    <div style={{transform: changed ? `translateY(${-roll * 44}px)` : undefined}}>
      {(changed ? [previous, value] : [value]).map((v, i) => <div key={i} style={{height: 44, display: 'flex',
        alignItems: 'center', justifyContent: 'center'}}>{v}</div>)}</div></div>;
};
const PeriodCell = ({period, previous, local}) => {
  const active = previous != null && previous !== period && local >= 0 && local < 40;
  const roll = active ? io(clamp(local / 10)) : 1;
  const flash = active ? io(clamp((30 - local) / 12)) : 0;
  return <div style={{width: PERIOD, height: 90, overflow: 'hidden', background: `rgba(238,243,246,${flash})`,
    color: flash > .35 ? plate : frost, fontSize: 22, fontWeight: 650}}>
    <div style={{transform: active ? `translateY(${-roll * 90}px)` : undefined}}>
      {(active ? [previous, period] : [period]).map((p, i) => <div key={i} style={{height: 90, display: 'flex',
        alignItems: 'center', justifyContent: 'center'}}>{periodLabel(p)}</div>)}</div></div>;
};

export const EventBar = ({team, text, frame, duration = 70}) => {
  const drop = enter(frame, 0, 10), grow = enter(frame, 6, Math.max(1, duration - 16));
  const exit = enter(frame, duration - 10, 10), reveal = drop * (1 - exit);
  return <div style={{height: 40 * reveal, overflow: 'hidden', marginTop: 2}}>
    <div style={{position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', height: 40,
      width: WIDTH, transform: `translateY(${(reveal - 1) * 40}px)`, color: team.text ?? ink,
      background: `linear-gradient(180deg, rgba(255,255,255,.10), transparent 45%), ${team.primary}`,
      boxShadow: 'inset 0 0 0 999px rgba(0,0,0,.14)'}}>
      <span style={{fontSize: fit(text, 25 + 3 * grow, WIDTH - 20, 4 + 6 * grow), fontWeight: 850,
        letterSpacing: 4 + 6 * grow, whiteSpace: 'nowrap'}}>{text}</span>
      <Accent color={team.secondary} p={grow}/></div></div>;
};

export const StackedBug = ({event, props, absoluteFrame, vertical}) => {
  const motion = event.animation || {};
  const score = {local: absoluteFrame - (motion.scoreFrame ?? event.startFrame), previous: motion.previousScore};
  const period = {local: absoluteFrame - (motion.periodFrame ?? event.startFrame), previous: motion.previousPeriod};
  const k = motion.initiallyVisible || event.data.initiallyVisible ? 1 : enter(absoluteFrame, event.startFrame, 12);
  const bar = props.events.find(e => e.kind === 'event_bar' && e.startFrame <= absoluteFrame && absoluteFrame < e.startFrame + e.durationFrames);
  const replay = props.events.find(e => e.kind === 'replay' && e.startFrame <= absoluteFrame && absoluteFrame < e.startFrame + e.durationFrames);
  const row = (side, i) => <div style={{display: 'flex', height: 44}}>
    <div style={{width: TEAM, display: 'flex', alignItems: 'center', gap: 12, padding: '0 14px', boxSizing: 'border-box',
      background: glow(props.teams[side], 80)}}><Crest team={props.teams[side]} size={32}/>
      <span style={{fontSize: fit(props.teams[side].acronym, 21, 98), fontWeight: 750}}>{props.teams[side].acronym}</span></div>
    <ScoreCell team={props.teams[side]} value={event.data.score[i]} previous={score.previous?.[i]} local={score.local}/></div>;
  const r = replay ? absoluteFrame - replay.startFrame : 0;
  const rk = replay ? enter(r, 0) * (1 - enter(r, replay.durationFrames - 8)) : 0;
  return <div style={{position: 'absolute', zIndex: 40, left: vertical ? (1080 - WIDTH * 1.3) / 2 : 96,
    top: vertical ? 220 : 54, transform: `scale(1.3) translateY(${(1 - k) * -8}px)`, transformOrigin: '0 0',
    opacity: k, color: ink, filter: 'drop-shadow(0 5px 12px rgba(0,0,0,.4))'}}>
    <div style={{display: 'flex', background: plate}}><div style={{display: 'flex', flexDirection: 'column', gap: 2}}>
      {row('home', 0)}{row('away', 1)}</div><PeriodCell period={event.data.period} previous={period.previous} local={period.local}/></div>
    {bar && <EventBar team={props.teams[bar.data.team]} text={bar.data.label} frame={absoluteFrame - bar.startFrame} duration={bar.durationFrames}/>}
    {!bar && replay && <div style={{height: 28 * rk, overflow: 'hidden', marginTop: 2}}>
      <div style={{height: 28, width: WIDTH, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#DCE6EC',
        color: plate, fontSize: 15, fontWeight: 800, letterSpacing: 4, transform: `translateY(${(rk - 1) * 28}px)`}}>REPLAY</div></div>}
  </div>;
};

export const RegionalCard = ({data, teams, duration, vertical, kind}) => {
  const frame = useCurrentFrame(), c = frame - 6;
  if (c < 0) return null;
  const team = teams[data.team], goal = kind === 'goal', call = kind === 'call';
  const num = data.num && data.num !== '?' ? `#${data.num.replace(/^#/, '')} ` : '';
  const player = goal ? data.scorer : data.name;
  const title = call ? data.result : `${num}${player?.startsWith('#') ? '' : player || team.name}`;
  const kicker = call ? 'OFFICIATING DECISION' : `${team.name.toUpperCase()} ${goal ? 'GOAL' : 'PENALTY'}`;
  const assists = (data.assists || []).map(a => /^\d/.test(a) ? `#${a}` : a);
  const detail = call ? data.reason : goal ? (data.assistsKnown === false ? 'ASSISTS NOT VERIFIED' :
    assists.length ? `${assists.length === 1 ? 'ASSIST' : 'ASSISTS'}: ${assists.join(', ')}` : 'UNASSISTED') :
    [data.infraction, ...(data.additional || []).map(p => p.classification?.toUpperCase())].filter(Boolean).join(' · ');
  const valueLabel = goal ? 'SCORE' : call ? 'RESULT' : (data.classification === 'major' ? 'MAJOR' : 'PENALTY');
  const value = goal ? (data.score || []).join('–') : call ? data.penaltyDisposition :
    data.minutes == null || data.minutes === '?' ? '—' : `${data.minutes} MIN`;
  const g = enter(c, 0), logo = enter(c, 3, 6), name = enter(c, 9, 6), box = enter(c, 12, 6);
  const chars = Math.round(kicker.length * enter(c, 6));
  const exit = enter(frame, duration - 10, 10), p = enter(c, 18, 80);
  const width = vertical ? 840 : 760, textWidth = width - 290;
  return <div style={{position: 'absolute', zIndex: 20, left: 96, bottom: vertical ? 430 : 74,
    display: 'flex', height: 104, width, color: ink, opacity: 1 - exit,
    transform: `translateX(${exit * -30}px)`, filter: 'drop-shadow(0 8px 18px rgba(0,0,0,.35))'}}>
    <div style={{position: 'relative', flex: 1, overflow: 'hidden'}}>
      <div style={{position: 'absolute', inset: 0, background: plate, clipPath: `inset(0 ${(1 - g) * 100}% 0 0)`}}>
        <div style={{position: 'absolute', inset: 0, background: glow(team, 55)}}/></div>
      <div style={{position: 'relative', height: '100%', display: 'flex', alignItems: 'center', gap: 20, padding: '0 24px'}}>
        <div style={{opacity: logo, transform: `scale(${.6 + .4 * logo})`}}><Crest team={team} size={64}/></div>
        <div style={{minWidth: 0}}><div style={{fontSize: fit(kicker, 16, textWidth, 3), fontWeight: 800,
          letterSpacing: 3, color: frost, minHeight: 19, whiteSpace: 'nowrap'}}>{kicker.slice(0, chars)}</div>
          <div style={{fontSize: fit(title, 34, textWidth), fontWeight: 750, lineHeight: 1.1, marginTop: 6,
            opacity: name, whiteSpace: 'nowrap', transform: `translateY(${(1 - name) * 14}px)`}}>{title}</div>
          <div style={{fontSize: fit(detail, 19, textWidth), fontWeight: 650, color: frost, marginTop: 4,
            opacity: name, whiteSpace: 'nowrap'}}>{detail}</div></div></div><Accent color={team.secondary} p={p}/></div>
    <div style={{width: 130, height: 104, overflow: 'hidden'}}><div style={{height: '100%', display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: grey,
      transform: `translateY(${(box - 1) * 104}px)`}}>
      <div style={{fontSize: 13, fontWeight: 800, letterSpacing: 2.5, color: frost}}>{valueLabel}</div>
      <div style={{fontSize: fit(value, 38, 118), fontWeight: 800, marginTop: 2, whiteSpace: 'nowrap'}}>{value}</div></div></div>
  </div>;
};

export const RegionalMatchup = ({data, teams, duration, vertical}) => {
  const m = useCurrentFrame(), p = enter(m, 20, 60), exit = enter(m, duration - 10, 10);
  const first = data.team === 'away' ? 'away' : 'home', second = first === 'home' ? 'away' : 'home';
  const teamCell = (side, start, mirror) => <div style={{position: 'relative', width: vertical ? 840 : 430, height: 96,
    display: 'flex', alignItems: 'center', gap: 20, padding: '0 26px', boxSizing: 'border-box',
    flexDirection: mirror && !vertical ? 'row-reverse' : 'row', background: glow(teams[side], 75, mirror ? 270 : 90),
    ...slide(enter(m, start))}}><Crest team={teams[side]} size={66}/>
    <span style={{fontSize: fit(teams[side].name, 34, vertical ? 660 : 292), fontWeight: 750, whiteSpace: 'nowrap'}}>{teams[side].name}</span>
    <Accent color={teams[side].secondary} p={p}/></div>;
  return <div style={{position: 'absolute', zIndex: 30, left: 0, right: 0, bottom: vertical ? 430 : 74,
    display: 'flex', flexDirection: 'column', alignItems: 'center', color: ink, opacity: 1 - exit,
    transform: `translateX(${exit * -30}px)`}}>
    <div style={{fontSize: 22, fontWeight: 800, letterSpacing: 4, marginBottom: 10,
      textShadow: '0 2px 8px rgba(0,0,0,.6)', ...slide(enter(m, 0))}}>{dateLabel(data.date)}</div>
    <div style={{display: 'flex', flexDirection: vertical ? 'column' : 'row', background: plate,
      boxShadow: '0 12px 30px rgba(0,0,0,.45)'}}>{teamCell(first, 2, false)}
      <div style={{width: vertical ? 840 : 90, height: vertical ? 48 : 96, display: 'flex', alignItems: 'center',
        justifyContent: 'center', background: grey, fontSize: 24, fontWeight: 800, letterSpacing: 3, color: frost,
        ...slide(enter(m, 6))}}>VS</div>{teamCell(second, 9, true)}</div></div>;
};

export const RegionalResult = ({data, teams, frame, lift, vertical}) => {
  const row = (side, i, start) => <div style={{display: 'flex', height: 92, ...slide(enter(frame, start))}}>
    <div style={{width: vertical ? 560 : 560, display: 'flex', alignItems: 'center', gap: 22, padding: '0 26px',
      boxSizing: 'border-box', background: glow(teams[side], 70), opacity: data.score[i] < data.score[1 - i] ? .62 : 1}}>
      <Crest team={teams[side]} size={66}/><span style={{fontSize: fit(teams[side].name, 36, 420),
        fontWeight: 750, whiteSpace: 'nowrap'}}>{teams[side].name}</span></div>
    <div style={{width: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', background: grey,
      fontSize: 56, fontWeight: 800, color: data.score[i] < data.score[1 - i] ? '#7D8A95' : ink}}>{data.score[i]}</div></div>;
  const top = (vertical ? 700 : 380) + ((vertical ? 300 : 90) - (vertical ? 700 : 380)) * lift;
  return <div style={{position: 'absolute', top, left: 0, right: 0, display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
    <div style={{fontSize: 22, fontWeight: 800, letterSpacing: 4, color: frost, marginBottom: 14,
      ...slide(enter(frame, 0))}}>{dateLabel(data.date)}</div>
    <div style={{display: 'flex', boxShadow: '0 18px 40px rgba(0,0,0,.5)'}}><div style={{display: 'flex', flexDirection: 'column', gap: 3}}>
      {row('home', 0, 2)}{row('away', 1, 5)}</div><div style={{width: 150, display: 'flex', alignItems: 'center',
        justifyContent: 'center', background: plate, fontSize: 30, fontWeight: 800, letterSpacing: 5,
        ...slide(enter(frame, 8))}}>FINAL</div></div></div>;
};
