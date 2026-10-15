import React from 'react';
import {Easing, Img, staticFile, useCurrentFrame} from 'remotion';

const ink = '#EEF3F6';
const label = '#9FB2C1';

// Black Ice blade cut: a bright blade tip carves the rule line in from left to right,
// and the kit-coloured line stays behind it.
export const BladeLine = ({color, frame, width, start = 2, length = 14, height = 3}) => {
  const t = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, (frame - start) / length)));
  const tip = Math.max(0, 1 - Math.max(0, frame - start - length) / 8);
  return <div style={{position: 'relative', height, width}}>
    <div style={{position: 'absolute', left: 0, top: 0, height, width: t * width, background: color}}/>
    <div style={{position: 'absolute', top: -2, left: t * width - 46, width: 46, height: height + 4,
      opacity: tip * (t > 0 ? 1 : 0), background: 'linear-gradient(90deg, rgba(238,243,246,0), #FFFFFF)',
      boxShadow: `0 0 10px 2px ${color}`, borderRadius: 3}}/>
  </div>;
};

export const EventRail = ({data, teams, tokens, duration, vertical, kind}) => {
  const f = useCurrentFrame();
  const entry = kind === 'penalty' ? 12 : 14;
  const enter = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, f / entry)));
  const exit = Easing.in(Easing.cubic)(Math.max(0, Math.min(1, (f - duration + 11) / 10)));
  const team = teams[data.team];
  const call = kind === 'call';
  const goal = kind === 'goal';
  const player = goal ? data.scorer : data.name;
  const named = player && !player.startsWith('#');
  const title = call ? data.result : named ? player : data.num && data.num !== '?' ?
    `${goal ? 'SCORER' : 'PLAYER'} #${data.num}` : team.name;
  const assists = (data.assists || []).filter(Boolean);
  const detail = call ? data.reason : goal ?
    (data.assistsKnown === false ? 'ASSISTS NOT VERIFIED' : assists.length ? `${assists.length === 1 ? 'ASSIST' : 'ASSISTS'}  ${assists.join(' · ')}` :
      'UNASSISTED') : data.infraction;
  const assessed = /^[\d.+:]+$/.test(String(data.minutes)) ? `${data.minutes} MIN` :
    data.minutes && !['?', '—'].includes(String(data.minutes)) ? String(data.minutes) : '';
  // The rule line under the ice: the kit accent for calls, the team's line for its own events.
  const accent = call ? tokens?.accent ?? '#7FD3F0' : team.line ?? '#9FB2C1';
  const width = vertical ? 840 : 1010;
  const size = vertical ? 34 : 30;
  const fontSize = title.length > 42 ? size - 4 : size;
  return <div style={{position: 'absolute', zIndex: 20, left: vertical ? 96 : 96,
    bottom: vertical ? 430 : 74, width, color: ink,
    opacity: enter * (1 - exit), transform: `translateX(${(1 - enter) * -32 + exit * 24}px)`,
    filter: 'drop-shadow(0 8px 18px rgba(0,0,0,.35))'}}>
    <BladeLine color={accent} frame={f} width={width}/>
    <div style={{background: 'linear-gradient(#19212A, #0E1319)',
      padding: vertical ? '22px 48px 22px 24px' : '18px 48px 18px 22px'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 18}}>
        {!call && <div style={{width: 64, height: 64, background: '#232D38', display: 'flex', alignItems: 'center',
          justifyContent: 'center', flexShrink: 0}}>{team.logo ? <Img src={staticFile(team.logo)}
            style={{width: 56, height: 56, objectFit: 'contain'}}/> :
            <b style={{fontSize: Math.min(22, 60 / Math.max(1, team.acronym.length * .8))}}>{team.acronym}</b>}</div>}
        <div style={{flex: 1, minWidth: 0}}>
          <div style={{fontSize: vertical ? 18 : 15, letterSpacing: 1.8, color: label,
            fontWeight: 650, marginBottom: 5}}>{call ? 'OFFICIATING DECISION · DESIGN SAMPLE' :
            `${team.name.toUpperCase()}${named && data.num ? `  ·  #${data.num}` : ''}`}</div>
          <div style={{fontSize, lineHeight: 1.15, fontWeight: 750, overflowWrap: 'break-word', textWrap: 'balance'}}>{title}</div>
        </div>
        {!call && <div style={{alignSelf: 'stretch', width: goal ? 128 : 154,
          borderLeft: '1px solid rgba(238,243,246,.18)', paddingLeft: 18,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0}}><b style={{fontSize: goal ? 44 : 17, letterSpacing: 1}}>{goal ? 'GOAL' : 'PENALTY'}</b>
          {!goal && assessed && <b style={{fontSize: 28, color: ink, marginTop: 6}}>{assessed}</b>}</div>}
      </div>
      <div style={{marginTop: 14, paddingTop: 12, borderTop: '1px solid rgba(238,243,246,.12)',
        paddingLeft: call ? 0 : 82, fontSize: vertical ? 24 : 21, lineHeight: 1.3,
        color: goal ? label : ink, fontWeight: 600, overflowWrap: 'break-word', textWrap: 'balance'}}>{detail}</div>
      {call && <div style={{marginTop: 8, fontSize: vertical ? 21 : 19,
        color: label, letterSpacing: .5}}>{data.penaltyDisposition}</div>}
    </div>
  </div>;
};
