// PROTOTYPE — Variant C "Ice Edge": brand-led, angular. Symmetric top-centre bug of skewed tiles;
// GOAL is a diagonal shard stinger that leaves a skewed name plate; shard-band period wipe.
import React from 'react';
import {AbsoluteFill, useCurrentFrame, Easing} from 'remotion';
import {loadFont} from '@remotion/google-fonts/BarlowCondensed';
import {inOut, ramp, Mark, scorerLine, other} from './shared';

const {fontFamily} = loadFont('normal', {weights: ['600', '800'], subsets: ['latin']});
const SK = -14;
const DARK = '#0A1426';
const Tile = ({bg, w, children, style}) => (
  <div style={{background: bg, width: w, height: 58, transform: `skewX(${SK}deg)`, display: 'flex', alignItems: 'center', justifyContent: 'center', ...style}}>
    <div style={{transform: `skewX(${-SK}deg)`, display: 'flex', alignItems: 'center', gap: 10, color: '#fff'}}>{children}</div>
  </div>
);

const Bug = ({g, dur, period, score, changeAt}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 12);
  const [h, a] = score(f);
  const pop = changeAt != null ? Math.max(0, 1 - Math.abs(f - changeAt - 6) / 10) : 0;
  return (
    <div style={{position: 'absolute', top: 34, left: '50%', transform: `translateX(-50%) translateY(${(1 - t) * -90}px)`, display: 'flex', gap: 4,
      fontStyle: 'italic', fontWeight: 800, filter: 'drop-shadow(0 6px 14px rgba(0,0,0,.4))'}}>
      <Tile bg={g.home.color} w={190}>{g.home.logo && <Mark team={g.home} size={44} />}<span style={{fontSize: 28}}>{g.home.abbr}</span></Tile>
      <Tile bg="#fff" w={70}><span style={{color: DARK, fontSize: 42, transform: `scale(${1 + pop * .4})`}}>{h}</span></Tile>
      <Tile bg={DARK} w={96}><span style={{fontSize: 24, fontWeight: 600}}>{period}</span></Tile>
      <Tile bg="#fff" w={70}><span style={{color: DARK, fontSize: 42}}>{a}</span></Tile>
      <Tile bg={g.away.color} w={190}>{g.away.logo && <Mark team={g.away} size={44} />}<span style={{fontSize: 28}}>{g.away.abbr}</span></Tile>
    </div>
  );
};

const Shards = ({f, color, accent, dur}) => {
  const bands = [color, accent, '#ffffff', color];
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      {bands.map((c, i) => {
        const x = ramp(f, i * 2, dur * 0.5 + i * 2, -160, 160, Easing.inOut(Easing.cubic));
        return <div key={i} style={{position: 'absolute', top: -200, bottom: -200, left: `${x + i * 6}%`, width: `${22 - i * 4}%`, background: c,
          transform: `skewX(${SK * 2}deg)`, opacity: i === 2 ? .9 : 1}} />;
      })}
    </AbsoluteFill>
  );
};

const GoalCard = ({g, goal, dur, compact}) => {
  const f = useCurrentFrame();
  const team = g[goal.team];
  if (compact) {
    const t = inOut(f, dur, 14, 12);
    return (
      <div style={{position: 'absolute', top: 34, left: '50%', transform: `translateX(-50%) scaleX(${t})`, fontStyle: 'italic'}}>
        <Tile bg={team.color} w={620} style={{height: 58}}>
          <span style={{fontSize: 34, fontWeight: 800}}>{team.name} GOAL</span><span style={{fontSize: 24, opacity: .85}}>{scorerLine(goal)} · {goal.time}</span>
        </Tile>
      </div>
    );
  }
  const plate = inOut(f - 16, dur - 16, 16, 14);
  return (
    <AbsoluteFill>
      {f < 40 && <Shards f={f} color={team.color} accent={team.accent} dur={40} />}
      <div style={{position: 'absolute', left: 60, bottom: 90, fontStyle: 'italic', transform: `translateX(${(1 - plate) * -700}px)`}}>
        <div style={{display: 'flex', alignItems: 'stretch'}}>
          <div style={{background: team.color, transform: `skewX(${SK}deg)`, padding: '0 34px', display: 'flex', alignItems: 'center', gap: 20}}>
            <div style={{transform: `skewX(${-SK}deg)`, display: 'flex', alignItems: 'center', gap: 20}}>
              <Mark team={team} size={110} wordStyle={{fontSize: 40}} />
              <div style={{color: '#fff', fontSize: 120, fontWeight: 800, lineHeight: 1}}>GOAL</div>
            </div>
          </div>
          <div style={{width: 18, background: team.accent, transform: `skewX(${SK}deg)`, marginLeft: 6}} />
        </div>
        <div style={{background: DARK, color: '#fff', transform: `skewX(${SK}deg)`, marginTop: 6, marginLeft: -12, padding: '8px 34px', width: 'fit-content'}}>
          <div style={{transform: `skewX(${-SK}deg)`}}>
            <div style={{fontSize: 46, fontWeight: 800}}>{scorerLine(goal)}</div>
            <div style={{fontSize: 26, fontWeight: 600, opacity: .8}}>{goal.assists.length ? `ASSISTS: ${goal.assists.join(' · ')}` : 'UNASSISTED'} — {goal.time}</div>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const Plate = ({dur, accent, children, top}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 14, 12);
  return (
    <div style={{position: 'absolute', top: top ?? undefined, bottom: top ? undefined : 90, left: top ? '50%' : 60,
      transform: top ? `translateX(-50%) translateY(${(1 - t) * -40}px)` : `translateX(${(1 - t) * -600}px)`, opacity: top ? t : 1,
      fontStyle: 'italic', display: 'flex'}}>
      <div style={{width: 14, background: accent, transform: `skewX(${SK}deg)`, marginRight: 4}} />
      <Tile bg="rgba(10,20,38,.94)" w="auto" style={{padding: '0 28px'}}>{children}</Tile>
    </div>
  );
};

const PenaltyCard = ({g, pen, dur}) => (
  <Plate dur={dur} accent="#F5C400">
    <span style={{color: '#F5C400', fontSize: 30, fontWeight: 800}}>PENALTY</span>
    <span style={{fontSize: 28, fontWeight: 600}}>{g[pen.team].name} #{pen.num} — {pen.infraction} — {pen.minutes}:00</span>
  </Plate>
);
const Flap = ({g, flap, dur}) => (
  <Plate dur={dur} accent={g[flap.team].accent} top={100}>
    <span style={{fontSize: 28, fontWeight: 800}}>GOAL</span>
    <span style={{fontSize: 26, fontWeight: 600}}>{g[flap.team].name} · {scorerLine(flap)} · {flap.time}</span>
  </Plate>
);

const ReplayTag = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  return (
    <div style={{position: 'absolute', top: 40, left: 60, opacity: inOut(f, dur, 10, 10), fontStyle: 'italic'}}>
      <Tile bg={team ? team.color : DARK} w={200}><span style={{fontSize: 30, fontWeight: 800, letterSpacing: 2}}>REPLAY</span></Tile>
    </div>
  );
};

const PeriodWipe = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : g.home;
  const cover = f < dur / 2 ? ramp(f, 0, dur / 2, 0, 1, Easing.out(Easing.cubic)) : ramp(f, dur / 2, dur, 1, 2, Easing.in(Easing.cubic));
  const bands = [team.color, DARK, team.accent, team.color, DARK];
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      {bands.map((c, i) => {
        const left = -140 + cover * 120 + i * 10 - (i % 2) * 6;
        return <div key={i} style={{position: 'absolute', top: -300, bottom: -300, left: `${left}%`, width: '60%', background: c, transform: `skewX(${SK * 2}deg)`}} />;
      })}
      {g.focus && <AbsoluteFill style={{display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 1 - Math.abs(cover - 1) * 3}}><Mark team={team} size={380} /></AbsoluteFill>}
    </AbsoluteFill>
  );
};

const PeriodCard = ({g, period, score, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 14);
  return (
    <div style={{position: 'absolute', left: 60, top: 440, transform: `translateX(${(1 - t) * -900}px)`, fontStyle: 'italic', display: 'flex', gap: 6}}>
      <Tile bg={DARK} w={460} style={{height: 110}}><span style={{fontSize: 80, fontWeight: 800}}>{period}</span></Tile>
      <Tile bg={g.home.color} w={170} style={{height: 110}}><span style={{fontSize: 34, fontWeight: 600}}>{g.home.abbr}</span><span style={{fontSize: 64, fontWeight: 800}}>{score[0]}</span></Tile>
      <Tile bg={g.away.color} w={170} style={{height: 110}}><span style={{fontSize: 34, fontWeight: 600}}>{g.away.abbr}</span><span style={{fontSize: 64, fontWeight: 800}}>{score[1]}</span></Tile>
    </div>
  );
};

const Open = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  const out = ramp(f, dur - 12, dur, 1, 0, Easing.in(Easing.cubic));
  const logo = ramp(f, 22, 46, 0, 1, Easing.out(Easing.back(1.4)));
  const plate = ramp(f, 50, 74);
  return (
    <AbsoluteFill style={{background: DARK, opacity: out, overflow: 'hidden', fontStyle: 'italic'}}>
      {f < 60 && <Shards f={f} color={(team || g.home).color} accent={(team || g.home).accent} dur={60} />}
      <AbsoluteFill style={{display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
        <div style={{transform: `scale(${logo})`, display: 'flex', alignItems: 'center', gap: 60}}>
          {team ? <Mark team={team} size={520} /> : (<>
            <Mark team={g.home} size={340} /><span style={{color: '#fff', fontSize: 90, fontWeight: 800}}>VS</span><Mark team={g.away} size={340} wordStyle={{fontSize: 120}} />
          </>)}
        </div>
        <div style={{marginTop: 26, opacity: plate, transform: `translateX(${(1 - plate) * 200}px)`}}>
          <Tile bg={(team || g.home).color} w="auto" style={{padding: '0 40px'}}>
            <span style={{fontSize: 36, fontWeight: 800}}>{team ? `VS ${g[other(g.focus)].name}` : 'GAME RECAP'}</span>
            <span style={{fontSize: 28, fontWeight: 600, opacity: .85}}>{g.league} · {g.date}</span>
          </Tile>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const FinalCard = ({g, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 20, 16);
  const lead = g.focus || 'home';
  const cols = g.focus ? [g.focus] : ['home', 'away'];
  return (
    <AbsoluteFill style={{fontStyle: 'italic', color: '#fff'}}>
      <div style={{position: 'absolute', top: -100, bottom: -100, left: `${-60 + t * 50}%`, width: '62%', background: g[lead].color, transform: `skewX(${SK}deg)`}} />
      <div style={{position: 'absolute', top: -100, bottom: -100, right: `${-60 + t * 50}%`, width: '58%', background: 'rgba(10,20,38,.94)', transform: `skewX(${SK}deg)`}} />
      <div style={{position: 'absolute', left: 120, top: 250, opacity: t}}>
        <Mark team={g[lead]} size={260} wordStyle={{fontSize: 90}} />
        <div style={{fontSize: 44, fontWeight: 600, letterSpacing: 6, marginTop: 20}}>FINAL</div>
        <div style={{fontSize: 190, fontWeight: 800, lineHeight: 1}}>{g.final.home}–{g.final.away}</div>
      </div>
      <div style={{position: 'absolute', right: 140, top: 280, opacity: t, display: 'flex', gap: 60}}>
        {cols.map((c) => (
          <div key={c}>
            <div style={{fontSize: 30, fontWeight: 800, color: g[c].accent, marginBottom: 12}}>{g[c].name}</div>
            {g.final.leaders[c].map(([n, st]) => <div key={n} style={{fontSize: 42, fontWeight: 600, lineHeight: 1.5}}>{n} <span style={{fontWeight: 800}}>{st}</span></div>)}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const Watermark = ({g, dur}) => {
  const f = useCurrentFrame();
  return <div style={{position: 'absolute', right: 50, bottom: 40, opacity: .6 * inOut(f, dur, 20, 20)}}><Mark team={g[g.focus]} size={80} /></div>;
};

export const VariantC = {name: 'Ice Edge', font: fontFamily, Bug, GoalCard, PenaltyCard, Flap, ReplayTag, PeriodWipe, PeriodCard, Open, FinalCard, Watermark};
