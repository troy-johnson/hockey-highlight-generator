// PROTOTYPE — Variant A "Broadcast Slab": classic network package. Horizontal top-left bug;
// GOAL card grows out of the bug slot; trays drop under the bug; full-screen colour wipe.
import React from 'react';
import {AbsoluteFill, useCurrentFrame, Easing} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Oswald';
import {inOut, ramp, Mark, scorerLine, other} from './shared';

const {fontFamily} = loadFont();
const DARK = '#0B0F17';
const X = 48, Y = 40, H = 62;

const TeamCell = ({team, score, pop = 0}) => (
  <div style={{display: 'flex', height: H}}>
    <div style={{background: team.color, width: 150, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '0 10px'}}>
      {team.logo ? <Mark team={team} size={46} /> : null}
      <div style={{color: '#fff', fontSize: 24, fontWeight: 600, letterSpacing: 1}}>{team.abbr}</div>
    </div>
    <div style={{background: '#fff', color: DARK, width: 64, display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: 40, fontWeight: 700, transform: `scale(${1 + pop * 0.35})`}}>{score}</div>
  </div>
);

const Bug = ({g, dur, period, score, changeAt}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 12);
  const [h, a] = score(f);
  const pop = changeAt != null ? Math.max(0, 1 - Math.abs(f - changeAt - 6) / 10) : 0;
  return (
    <div style={{position: 'absolute', left: X, top: Y, display: 'flex', boxShadow: '0 6px 24px rgba(0,0,0,.35)',
      clipPath: `inset(0 ${(1 - t) * 100}% 0 0)`}}>
      <TeamCell team={g.home} score={h} pop={pop} />
      <TeamCell team={g.away} score={a} />
      <div style={{background: DARK, color: '#fff', width: 92, height: H, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 26, fontWeight: 500}}>{period}</div>
    </div>
  );
};

const GoalCard = ({g, goal, dur, compact}) => {
  const f = useCurrentFrame();
  const team = g[goal.team];
  const t = inOut(f, dur, 18, 14);
  const word = ramp(f, 6, 22);
  const tray = inOut(f - 16, dur - 16, 14, 10);
  if (compact) {
    return (
      <div style={{position: 'absolute', left: X, top: Y, height: H, display: 'flex', alignItems: 'center', background: team.color,
        clipPath: `inset(0 ${(1 - t) * 100}% 0 0)`, padding: '0 22px', gap: 16, color: '#fff'}}>
        <Mark team={team} size={46} wordStyle={{fontSize: 24}} />
        <div style={{fontSize: 30, fontWeight: 700, letterSpacing: 2}}>GOAL</div>
        <div style={{fontSize: 22, opacity: .85}}>{scorerLine(goal)} · {goal.time}</div>
      </div>
    );
  }
  return (
    <div style={{position: 'absolute', left: X, top: Y}}>
      <div style={{display: 'flex', height: 96, width: 640, background: team.color, clipPath: `inset(0 ${(1 - t) * 100}% 0 0)`, boxShadow: '0 8px 30px rgba(0,0,0,.4)'}}>
        <div style={{width: 150, background: 'rgba(0,0,0,.25)', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          <Mark team={team} size={80} wordStyle={{fontSize: 22, textAlign: 'center', whiteSpace: 'normal'}} />
        </div>
        <div style={{flex: 1, display: 'flex', alignItems: 'center', paddingLeft: 26, overflow: 'hidden'}}>
          <div style={{color: '#fff', fontSize: 78, fontWeight: 700, letterSpacing: 10, transform: `translateY(${(1 - word) * 90}px)`}}>GOAL</div>
        </div>
        <div style={{width: 10, background: team.accent}} />
      </div>
      <div style={{background: DARK, color: '#fff', width: 640, padding: '12px 22px', clipPath: `inset(0 0 ${(1 - tray) * 100}% 0)`}}>
        <div style={{fontSize: 36, fontWeight: 600}}>{scorerLine(goal)}</div>
        <div style={{fontSize: 22, opacity: .75, marginTop: 2}}>
          {goal.assists.length ? `ASST: ${goal.assists.join(', ')}` : 'UNASSISTED'} · {goal.time}
        </div>
      </div>
    </div>
  );
};

const Tray = ({dur, accent, children}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 14, 12);
  return (
    <div style={{position: 'absolute', left: X, top: Y + H, display: 'flex', clipPath: `inset(0 0 ${(1 - t) * 100}% 0)`}}>
      <div style={{width: 10, background: accent}} />
      <div style={{background: 'rgba(11,15,23,.92)', color: '#fff', height: 50, display: 'flex', alignItems: 'center', gap: 18, padding: '0 20px', fontSize: 24}}>{children}</div>
    </div>
  );
};

const PenaltyCard = ({g, pen, dur}) => (
  <Tray dur={dur} accent="#F5C400">
    <b style={{color: '#F5C400', letterSpacing: 2}}>PENALTY</b>
    <span>{g[pen.team].short} #{pen.num}</span><span>{pen.infraction}</span><span style={{opacity: .7}}>{pen.minutes}:00</span>
  </Tray>
);

const Flap = ({g, flap, dur}) => (
  <Tray dur={dur} accent={g[flap.team].accent}>
    <b style={{letterSpacing: 2}}>GOAL</b><span>{g[flap.team].short}</span><span>{scorerLine(flap)}</span><span style={{opacity: .7}}>{flap.time}</span>
  </Tray>
);

const ReplayTag = ({dur}) => {
  const f = useCurrentFrame();
  return (
    <div style={{position: 'absolute', left: X, top: Y, opacity: inOut(f, dur, 10, 10), background: DARK, color: '#fff',
      padding: '10px 18px', fontSize: 26, letterSpacing: 3, display: 'flex', alignItems: 'center', gap: 10}}>
      <div style={{width: 12, height: 12, borderRadius: 6, background: '#E0242F'}} />REPLAY
    </div>
  );
};

const brandTeam = (g) => (g.focus ? g[g.focus] : null);

const PeriodWipe = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = brandTeam(g);
  const x = ramp(f, 0, dur * 0.45, -110, 0, Easing.out(Easing.quad));
  const x2 = ramp(f, dur * 0.55, dur, 0, 110, Easing.in(Easing.quad));
  const pos = f < dur * 0.5 ? x : x2;
  return (
    <AbsoluteFill style={{transform: `translateX(${pos}%) skewX(-12deg)`, background: team ? team.color : DARK,
      display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 60px rgba(0,0,0,.6)'}}>
      <div style={{transform: 'skewX(12deg)'}}>{team ? <Mark team={team} size={420} /> : <div style={{color: '#fff', fontSize: 120}}>{g.league}</div>}</div>
    </AbsoluteFill>
  );
};

const PeriodCard = ({g, period, score, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 14);
  return (
    <div style={{position: 'absolute', left: '50%', bottom: 90, transform: 'translateX(-50%)', display: 'flex',
      clipPath: `inset(0 ${(1 - t) * 50}% 0 ${(1 - t) * 50}%)`, boxShadow: '0 8px 30px rgba(0,0,0,.4)'}}>
      <div style={{background: DARK, color: '#fff', fontSize: 40, padding: '14px 34px', letterSpacing: 4, fontWeight: 600}}>{period}</div>
      <div style={{background: g.home.color, color: '#fff', fontSize: 34, padding: '14px 24px'}}>{g.home.abbr} {score[0]}</div>
      <div style={{background: g.away.color, color: '#fff', fontSize: 34, padding: '14px 24px'}}>{g.away.abbr} {score[1]}</div>
    </div>
  );
};

const Open = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = brandTeam(g);
  const out = ramp(f, dur - 12, dur, 1, 0, Easing.in(Easing.cubic));
  const blur = ramp(f, 0, 40, 18, 0);
  const sc = ramp(f, 0, 60, 1.25, 1);
  const glow = ramp(f, 20, 50, 0, 1) * ramp(f, 50, 110, 1, 0.25);
  const line = ramp(f, 55, 80);
  return (
    <AbsoluteFill style={{background: 'radial-gradient(circle at 50% 45%, #1a2c4d 0%, #05070c 70%)', opacity: out,
      display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
      <div style={{filter: `blur(${blur}px) drop-shadow(0 0 ${40 * glow}px rgba(120,200,255,${glow}))`, transform: `scale(${sc})`,
        display: 'flex', alignItems: 'center', gap: 80}}>
        {team ? <Mark team={team} size={560} /> : (<>
          <Mark team={g.home} size={360} /><div style={{color: '#fff', fontSize: 80}}>VS</div><Mark team={g.away} size={360} wordStyle={{fontSize: 110}} />
        </>)}
      </div>
      <div style={{color: '#fff', fontSize: 34, letterSpacing: 6, marginTop: 30, opacity: line, transform: `translateY(${(1 - line) * 20}px)`}}>
        {team ? `VS ${g[other(g.focus)].name} · ` : ''}{g.league} · {g.date}
      </div>
    </AbsoluteFill>
  );
};

const FinalCard = ({g, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 20, 16);
  const cols = g.focus ? [g.focus] : ['home', 'away'];
  return (
    <AbsoluteFill style={{background: `rgba(5,8,14,${0.78 * t})`, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
      <div style={{opacity: t, transform: `translateY(${(1 - t) * 30}px)`, color: '#fff', width: 1100}}>
        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '2px solid rgba(255,255,255,.3)', paddingBottom: 18}}>
          <div style={{width: 300, display: 'flex', justifyContent: 'center'}}><Mark team={g.home} size={140} /></div>
          <div style={{display: 'flex', alignItems: 'center', gap: 34, fontSize: 110, fontWeight: 700}}>
            {g.final.home}<span style={{fontSize: 34, letterSpacing: 4, fontWeight: 400}}>FINAL</span>{g.final.away}
          </div>
          <div style={{width: 300, display: 'flex', justifyContent: 'center'}}><Mark team={g.away} size={140} wordStyle={{fontSize: 56}} /></div>
        </div>
        <div style={{display: 'flex', justifyContent: 'space-around', marginTop: 20}}>
          {cols.map((c) => (
            <div key={c} style={{textAlign: 'center'}}>
              {g.final.leaders[c].map(([n, st]) => <div key={n} style={{fontSize: 36, lineHeight: 1.6}}>{n}: {st}</div>)}
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Watermark = ({g, dur}) => {
  const f = useCurrentFrame();
  return (
    <div style={{position: 'absolute', right: 44, bottom: 34, opacity: 0.55 * inOut(f, dur, 20, 20)}}>
      <Mark team={g[g.focus]} size={70} />
    </div>
  );
};

export const VariantA = {name: 'Broadcast Slab', font: fontFamily, Bug, GoalCard, PenaltyCard, Flap, ReplayTag, PeriodWipe, PeriodCard, Open, FinalCard, Watermark};
