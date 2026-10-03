// PROTOTYPE — Variant B "Glass Minimal": modern streaming look. Floating frosted bug bottom-left;
// GOAL is giant kinetic type across the lower third; toasts instead of trays; flash-cut period change.
import React from 'react';
import {AbsoluteFill, useCurrentFrame, Easing} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Inter';
import {inOut, ramp, Mark, scorerLine, other} from './shared';

const {fontFamily} = loadFont();
const GLASS = {background: 'rgba(14,18,26,.55)', backdropFilter: 'blur(18px) saturate(140%)', WebkitBackdropFilter: 'blur(18px)',
  border: '1px solid rgba(255,255,255,.14)', borderRadius: 18, color: '#fff'};
const L = 56, B = 52;

const Row = ({team, score, pop = 0}) => (
  <div style={{display: 'flex', alignItems: 'center', height: 50, gap: 14}}>
    <div style={{width: 8, height: 30, borderRadius: 4, background: team.color === '#12305E' ? team.accent : team.color}} />
    <div style={{flex: 1, fontSize: 24, fontWeight: 600, letterSpacing: .5}}>{team.short}</div>
    <div style={{fontSize: 34, fontWeight: 800, fontVariantNumeric: 'tabular-nums', transform: `scale(${1 + pop * .4})`}}>{score}</div>
  </div>
);

const Bug = ({g, dur, period, score, changeAt}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 18, 14);
  const [h, a] = score(f);
  const pop = changeAt != null ? Math.max(0, 1 - Math.abs(f - changeAt - 6) / 10) : 0;
  return (
    <div style={{position: 'absolute', left: L, bottom: B, width: 300, padding: '10px 20px 12px', ...GLASS,
      opacity: t, transform: `translateY(${(1 - t) * 30}px)`}}>
      <Row team={g.home} score={h} pop={pop} />
      <Row team={g.away} score={a} />
      <div style={{fontSize: 16, opacity: .7, letterSpacing: 3, marginTop: 4, display: 'flex', justifyContent: 'space-between'}}>
        <span>{period}</span>{g.focus && g[g.focus].logo ? <span>{g[g.focus].short}</span> : <span>{g.league}</span>}
      </div>
    </div>
  );
};

const GoalCard = ({g, goal, dur, compact}) => {
  const f = useCurrentFrame();
  const team = g[goal.team];
  if (compact) {
    const t = inOut(f, dur, 16, 12);
    return (
      <div style={{position: 'absolute', left: L, bottom: B, ...GLASS, padding: '18px 24px', display: 'flex', alignItems: 'center', gap: 18,
        opacity: t, transform: `translateY(${(1 - t) * 30}px)`}}>
        <div style={{width: 8, height: 44, borderRadius: 4, background: team.color}} />
        <div>
          <div style={{fontSize: 30, fontWeight: 800}}>{team.short} goal</div>
          <div style={{fontSize: 20, opacity: .75}}>{scorerLine(goal)} · {goal.time}</div>
        </div>
      </div>
    );
  }
  const band = inOut(f, dur, 14, 16);
  const letters = 'GOAL'.split('');
  const name = ramp(f, 24, 44);
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: 440 * band,
        background: `linear-gradient(to top, ${team.color}ee 0%, ${team.color}88 45%, transparent 100%)`}} />
      <div style={{position: 'absolute', left: 70, bottom: 120, display: 'flex', alignItems: 'flex-end', gap: 50, opacity: band}}>
        <div style={{display: 'flex'}}>
          {letters.map((c, i) => {
            const k = ramp(f, 4 + i * 4, 20 + i * 4, 0, 1, Easing.out(Easing.back(1.6)));
            return <div key={i} style={{color: '#fff', fontSize: 230, fontWeight: 900, lineHeight: .8, letterSpacing: -6,
              transform: `translateY(${(1 - k) * 120}px)`, opacity: k}}>{c}</div>;
          })}
        </div>
        <div style={{color: '#fff', marginBottom: 12, opacity: name, transform: `translateX(${(1 - name) * 40}px)`}}>
          <div style={{fontSize: 52, fontWeight: 800}}>{scorerLine(goal)}</div>
          <div style={{fontSize: 26, opacity: .85}}>{goal.assists.length ? `Assists  ${goal.assists.join(', ')}` : 'Unassisted'}</div>
          <div style={{fontSize: 22, opacity: .65, marginTop: 6}}>{team.name} · {goal.time}</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Toast = ({dur, dot, title, body}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 12);
  return (
    <div style={{position: 'absolute', right: 56, top: 52, ...GLASS, padding: '16px 22px', minWidth: 380,
      opacity: t, transform: `translateX(${(1 - t) * 60}px)`}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 10, fontSize: 18, letterSpacing: 3, opacity: .8}}>
        <div style={{width: 10, height: 10, borderRadius: 5, background: dot}} />{title}
      </div>
      <div style={{fontSize: 28, fontWeight: 700, marginTop: 6}}>{body}</div>
    </div>
  );
};

const PenaltyCard = ({g, pen, dur}) => (
  <Toast dur={dur} dot="#F5C400" title="PENALTY" body={`${g[pen.team].short} #${pen.num} · ${pen.infraction[0]}${pen.infraction.slice(1).toLowerCase()} · ${pen.minutes} min`} />
);
const Flap = ({g, flap, dur}) => (
  <Toast dur={dur} dot={g[flap.team].accent} title={`GOAL · ${flap.time}`} body={`${g[flap.team].short} — ${scorerLine(flap)}`} />
);

const ReplayTag = ({dur}) => {
  const f = useCurrentFrame();
  return (
    <div style={{position: 'absolute', right: 60, top: 56, color: '#fff', opacity: inOut(f, dur, 12, 12), fontSize: 20, letterSpacing: 8,
      display: 'flex', alignItems: 'center', gap: 14}}>
      <div style={{width: 40, height: 2, background: '#fff'}} />REPLAY
    </div>
  );
};

const PeriodWipe = ({g, dur}) => {
  const f = useCurrentFrame();
  const flash = f < dur / 2 ? ramp(f, 0, dur / 2, 0, 1, Easing.in(Easing.quad)) : ramp(f, dur / 2, dur, 1, 0, Easing.out(Easing.quad));
  return <AbsoluteFill style={{background: `radial-gradient(circle, rgba(255,255,255,${flash}) 0%, rgba(210,230,255,${flash * .9}) 60%, rgba(160,190,230,${flash * .8}) 100%)`}} />;
};

const PeriodCard = ({g, period, score, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 18, 18);
  const [num, word] = period.split(' ');
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle, rgba(0,0,0,${.45 * t}) 0%, rgba(0,0,0,${.15 * t}) 70%)`,
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', color: '#fff'}}>
      <div style={{fontSize: 220, fontWeight: 900, letterSpacing: -8, lineHeight: 1, opacity: t, transform: `scale(${0.9 + 0.1 * t})`}}>{num.toLowerCase()}</div>
      <div style={{fontSize: 30, letterSpacing: 14, opacity: t}}>{word}</div>
      <div style={{fontSize: 28, marginTop: 20, opacity: t * .85}}>{g.home.short} {score[0]}  —  {score[1]} {g.away.short}</div>
    </AbsoluteFill>
  );
};

const Open = ({g, dur}) => {
  const f = useCurrentFrame();
  const out = ramp(f, dur - 14, dur, 1, 0, Easing.in(Easing.cubic));
  const team = g.focus ? g[g.focus] : null;
  const track = ramp(f, 0, 70, 60, 8);
  const sub = ramp(f, 50, 80);
  return (
    <AbsoluteFill style={{background: '#07090d', opacity: out, color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
      {team && <div style={{opacity: ramp(f, 10, 40), marginBottom: 30}}><Mark team={team} size={150} /></div>}
      <div style={{fontSize: team ? 150 : 110, fontWeight: 900, letterSpacing: track, opacity: ramp(f, 0, 30)}}>
        {team ? team.name : `${g.home.name}  ·  ${g.away.name}`}
      </div>
      <div style={{fontSize: 28, letterSpacing: 6, opacity: sub * .8, marginTop: 20}}>
        {team ? `vs ${g[other(g.focus)].name} — ` : ''}{g.league} · {g.date}
      </div>
    </AbsoluteFill>
  );
};

const FinalCard = ({g, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 22, 18);
  const cols = g.focus ? [g.focus] : ['home', 'away'];
  return (
    <AbsoluteFill style={{backdropFilter: `blur(${24 * t}px)`, background: `rgba(6,8,12,${.6 * t})`, color: '#fff',
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 140, opacity: t}}>
      <div style={{textAlign: 'center'}}>
        <div style={{fontSize: 22, letterSpacing: 10, opacity: .7}}>FINAL</div>
        <div style={{fontSize: 200, fontWeight: 900, letterSpacing: -6, lineHeight: 1}}>{g.final.home}<span style={{opacity: .35}}>–</span>{g.final.away}</div>
        <div style={{fontSize: 28, opacity: .8}}>{g.home.short}  ·  {g.away.short}</div>
      </div>
      <div style={{display: 'flex', gap: 60}}>
        {cols.map((c) => (
          <div key={c}>
            <div style={{fontSize: 18, letterSpacing: 6, opacity: .6, marginBottom: 10}}>{g[c].short} LEADERS</div>
            {g.final.leaders[c].map(([n, st]) => (
              <div key={n} style={{display: 'flex', justifyContent: 'space-between', gap: 40, fontSize: 32, lineHeight: 1.7, borderBottom: '1px solid rgba(255,255,255,.12)'}}>
                <span>{n}</span><span style={{fontWeight: 800}}>{st}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const Watermark = ({g, dur}) => {
  const f = useCurrentFrame();
  return <div style={{position: 'absolute', right: 56, bottom: 52, opacity: .4 * inOut(f, dur, 20, 20), filter: 'grayscale(1) brightness(2)'}}><Mark team={g[g.focus]} size={56} /></div>;
};

export const VariantB = {name: 'Glass Minimal', font: fontFamily, Bug, GoalCard, PenaltyCard, Flap, ReplayTag, PeriodWipe, PeriodCard, Open, FinalCard, Watermark};
