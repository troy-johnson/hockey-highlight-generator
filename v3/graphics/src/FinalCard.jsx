import React from 'react';
import {AbsoluteFill, Easing, Img, staticFile, useCurrentFrame} from 'remotion';
import {fit as fitText, glow, RegionalResult} from './IcePakGraphics';

// Black Ice FINAL: the result plate first, then the verified statistics.
const ice = '#EEF3F6';
const frost = '#9FB2C1';
const cyan = '#7FD3F0';
const clamp = v => Math.max(0, Math.min(1, v));
const out = Easing.inOut(Easing.cubic);
const inOut = Easing.inOut(Easing.cubic);
const fmt = v => v == null ? '—' : Number.isFinite(v) && Number.isInteger(v) ? v : Number.isFinite(v) ? Number(v.toFixed(2)) : v;  // 2:15 PIM shows 2.25, not 2.3 (spec 003)
const STATS = [['g', 'G'], ['a', 'A'], ['pts', 'PTS'], ['pim', 'PIM']];
// Shrink text to fit its width instead of clipping it (about .58 em per character).
const fit = fitText;

const Crest = ({team, size}) => team.logo ? <Img src={staticFile(team.logo)}
  style={{width: size, height: size, objectFit: 'contain', flexShrink: 0}}/> : null;

const Table = ({team, table, width, maxRows, start}) => {
  const f = useCurrentFrame();
  // Zero PIM is a recorded value. Both teams use the approved common grid.
  const stats = STATS.filter(([key]) => table.skaters.some(p => p[key] != null));
  const h = Math.max(20, Math.min(50, (maxRows - 54) / Math.max(1, table.skaters.length)));
  const nameWidth = width - 64 - 16 - 16 - stats.length * 72;
  const cell = {width: 72, textAlign: 'center', flexShrink: 0, fontVariantNumeric: 'tabular-nums'};
  // Long rosters keep the whole reveal inside about one second.
  const step = Math.min(3, 30 / Math.max(1, table.skaters.length));
  const reveal = i => {
    const k = out(clamp((f - start - i * step) / 10));
    return {opacity: k, transform: `translateX(${(1 - k) * -40}px)`};
  };
  const line = (key, i, num, name, values, bold) => <div key={key} style={{display: 'flex',
    alignItems: 'center', height: h, paddingRight: 16, background: i % 2 ? '#151C24' : '#19212A', ...reveal(i + 1)}}>
    <div style={{width: 64, alignSelf: 'stretch', display: 'flex', alignItems: 'center',
      justifyContent: 'center', background: '#232B34',
      fontSize: Math.min(21, h * .5), fontWeight: 750}}>{num && num !== '?' ? `#${String(num).replace(/^#/, '')}` : num}</div>
    <div style={{flex: 1, minWidth: 0, paddingLeft: 16, fontSize: fit(name, Math.min(24, h * .52), nameWidth),
      fontWeight: 600, whiteSpace: 'nowrap'}}>{name}</div>
    {values.map((v, j) => <div key={j} style={{...cell, fontSize: Math.min(23, h * .52),
      fontWeight: bold === j ? 800 : 500, color: bold === j ? ice : '#C9D4DC'}}>{fmt(v)}</div>)}
  </div>;
  const ptsIndex = stats.findIndex(([key]) => key === 'pts');
  return <div style={{width}}>
    <div style={{display: 'flex', alignItems: 'center', height: 54, paddingRight: 16, position: 'relative',
      background: glow(team, 60), ...reveal(0)}}>
      <div style={{width: 64, height: 54, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#232B34', flexShrink: 0}}><Crest team={team} size={36}/></div>
      <span style={{flex: 1, minWidth: 0, paddingLeft: 16, fontSize: fit(team.name.toUpperCase(), 19, nameWidth, 3),
        fontWeight: 800, letterSpacing: 3, whiteSpace: 'nowrap'}}>{team.name.toUpperCase()}</span>
      {stats.map(([, label]) => <div key={label} style={{...cell, fontSize: 15, fontWeight: 800,
        letterSpacing: 2, color: frost}}>{label}</div>)}
      <div style={{position: 'absolute', bottom: 0, left: 0, width: `${out(clamp((f - start - 4) / 30)) * 100}%`,
        height: 3, background: team.secondary}}/>
    </div>
    {table.skaters.map((p, i) => line(`s-${p.num}-${i}`, i, p.num, p.name,
      stats.map(([key]) => p[key]), ptsIndex))}
  </div>;
};

// A team with no recorded skater stats keeps its place as a dark, faded shim.
const EmptyTable = ({team, width, height, start}) => {
  const f = useCurrentFrame();
  const k = out(clamp((f - start) / 12));
  return <div style={{width, opacity: .55 * k, transform: `translateY(${(1 - k) * 10}px)`}}>
    <div style={{display: 'flex', alignItems: 'center', height: 52, gap: 12, paddingLeft: 14,
      background: '#0E1319', borderBottom: `3px solid ${team.line ?? team.primary}`}}>
      <Crest team={team} size={34}/>
      <span style={{flex: 1, fontSize: fit(team.name, 22, width - 74), fontWeight: 750,
        letterSpacing: 1.5, whiteSpace: 'nowrap'}}>{team.name.toUpperCase()}</span>
    </div>
    <div style={{height, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'linear-gradient(180deg, rgba(25,33,42,.9), rgba(14,19,25,.6))',
      fontSize: 20, fontWeight: 700, letterSpacing: 4, color: frost}}>NO STATS RECORDED</div>
  </div>;
};

// Goaltending: one table for both teams. Shots against and saves appear only when known.
const GSTATS = [['sa', 'SA'], ['ga', 'GA'], ['saves', 'SV'], ['sv_pct', 'SV%']];
const pct = v => v == null ? '—' : v >= 1 ? '1.000' : v.toFixed(3).replace(/^0/, '');
const GoalieTable = ({rows, width, start, accent}) => {
  const f = useCurrentFrame();
  const stats = GSTATS.filter(([key]) => rows.some(r => r.p[key] != null));
  const h = 46;
  const cell = {width: 86, textAlign: 'center', flexShrink: 0, fontVariantNumeric: 'tabular-nums'};
  const reveal = i => {
    const k = out(clamp((f - start - i * 3) / 10));
    return {opacity: k, transform: `translateY(${(1 - k) * 10}px)`};
  };
  const nameWidth = width - 64 - 34 - 12 - 18 - 12 - stats.length * 86;
  return <div style={{width}}>
    <div style={{display: 'flex', alignItems: 'center', height: 44, paddingLeft: 14,
      background: '#0E1319', borderBottom: `2px solid ${accent}`, ...reveal(0)}}>
      <span style={{flex: 1, fontSize: 18, fontWeight: 750, letterSpacing: 3}}>GOALTENDING</span>
      {stats.map(([, label]) => <div key={label} style={{...cell, fontSize: 16, fontWeight: 700,
        letterSpacing: 2, color: frost}}>{label}</div>)}<div style={{width: 12}}/>
    </div>
    {rows.map(({team, p}, i) => <div key={i} style={{display: 'flex', alignItems: 'center', height: h,
      gap: 0, background: i % 2 ? '#151C24' : '#19212A', ...reveal(i + 1)}}>
      <div style={{width: 64, alignSelf: 'stretch', display: 'flex', alignItems: 'center',
        justifyContent: 'center', background: '#232B34',
        fontSize: h * .5, fontWeight: 750}}>{p.num}</div>
      <div style={{width: 12}}/><Crest team={team} size={34}/>
      <div style={{flex: 1, minWidth: 0, paddingLeft: 12, fontSize: fit(p.name, 24, nameWidth),
        fontWeight: 600, whiteSpace: 'nowrap'}}>{p.name}</div>
      {stats.map(([key], j) => <div key={key} style={{...cell, fontSize: 24,
        fontWeight: key === 'sv_pct' ? 800 : 500, color: key === 'sv_pct' ? ice : '#C9D4DC'}}>
        {key === 'sv_pct' ? pct(p[key]) : fmt(p[key])}</div>)}
      <div style={{width: 12}}/>
    </div>)}
  </div>;
};

export const FinalCard = ({data, teams, tokens, duration, vertical}) => {
  const f = useCurrentFrame();
  const pages = Math.max(1, ...['home', 'away'].map(s => Math.ceil((data.table?.[s]?.skaters.length || 0) / 10)));
  const page = Math.min(pages - 1, Math.floor(f / (duration / pages)));
  data = {...data, table: Object.fromEntries(['home', 'away'].map(s => [s,
    {...data.table?.[s], skaters: (data.table?.[s]?.skaters || []).slice(page * 10, (page + 1) * 10)}]))};
  const accent = tokens?.accent ?? cyan;
  const W = vertical ? 1080 : 1920;
  const recorded = s => (data.table?.[s]?.skaters || []).length > 0;
  // When one team has stats, both teams keep a place; an empty side shows a faded shim.
  const sides = ['home', 'away'].some(recorded) ? ['home', 'away'] : [];
  const shimH = vertical ? 96 : 150;
  const goalies = ['home', 'away'].flatMap(s => (data.table?.[s]?.goalies || [])
    .map(p => ({team: teams[s], p})));
  const any = sides.length || goalies.length;
  const lift = any ? inOut(clamp((f - 30) / 16)) : 0;
  const leave = 1 - inOut(clamp((f - (duration - 10)) / 10));
  const tableTop = vertical ? 570 : 360;
  const tableWidth = vertical ? 960 : 700;
  const goalieH = goalies.length ? 44 + goalies.length * 46 + (vertical ? 28 : 24) : 0;
  const space = (vertical ? 1760 : 1050) - tableTop - goalieH;
  const tallest = Math.max(1, ...sides.map(s => data.table[s]?.skaters?.length || 0));
  const skaterH = Math.min(space, 52 + tallest * 50);
  return <AbsoluteFill style={{zIndex: 30, opacity: leave, color: ice,
    background: 'radial-gradient(ellipse at 50% 40%, #151C24 0%, #0A0D12 60%, #06080B 100%)'}}>
    <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: 1,
      background: 'rgba(238,243,246,.18)'}}/>
    <RegionalResult data={data} teams={teams} frame={f} lift={lift} vertical={vertical}/>
    {any > 0 && f >= 38 && <div style={{position: 'absolute', top: tableTop, left: 0, width: W,
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: vertical ? 28 : 24}}>
      {sides.length > 0 && <div style={{display: 'flex', flexDirection: vertical ? 'column' : 'row',
        alignItems: vertical ? 'center' : 'flex-start', gap: vertical ? 28 : 40}}>
        {sides.map(s => recorded(s)
          ? <Table key={s} team={teams[s]} table={data.table[s]} width={tableWidth} start={40}
              maxRows={vertical ? (sides.every(recorded) ? (space - 28) / 2 : space - 28 - 52 - shimH)
                : skaterH}/>
          : <EmptyTable key={s} team={teams[s]} width={tableWidth} start={40}
              height={vertical ? shimH : Math.min(shimH, skaterH - 52)}/>)}
      </div>}
      {goalies.length > 0 && <GoalieTable rows={goalies} width={vertical ? 960 : 1240}
        start={70} accent={accent}/>}
    </div>}
  </AbsoluteFill>;
};
