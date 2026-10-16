// PROTOTYPE — Variant D "Ice Pak Slab" (round 3). Built from the user's concept board with their real assets.
// Round 3: semantic colour tokens (same meaning → same background), team tiles in PRIMARY with SECONDARY
// end stripes, square logos + acronyms in the bug and GOAL card, both teams on FINAL, and Sportsnet-style
// stingers (particle dashes, slats that assemble the slab, shrinking end caps, masked reveals).
import React from 'react';
import {AbsoluteFill, useCurrentFrame, Easing, Img, staticFile} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Inter';
import {inOut, ramp, GAME} from './shared';
import {makeOpen, makePeriodWipe} from './StingerD';

const {fontFamily} = loadFont('normal', {weights: ['600', '700', '800', '900'], subsets: ['latin']});
loadFont('italic', {weights: ['800', '900'], subsets: ['latin']});

// Palette (user board, light blue matched to the IP logo) and SEMANTIC tokens.
const P = {logoBlue: '#5AA4D0', navy: '#0D1B2A', white: '#FFFFFF', light: '#D9D9D9', grey: '#6B6B6B', black: '#0A0A0A'};
const TOK = {
  score: P.white,     // numbers that are the score
  state: P.navy,      // game state: period, penalty length, FINAL, REPLAY, date
  info: P.black,      // player/event details: names, assists, infraction, scoring rows
  hero: P.white,      // the big GOAL word panel
  label: P.logoBlue,  // small accent labels inside info panels (ASSISTS:)
};
const SK = -16;
const X = 56, Y = 44;

// Default acronym: first letter of each word + last letter of the last word, skipping a plural S
// (ICE PAK → IPK, SALTY BOYS → SBY). A Team Config can override it.
export const acronym = (name) => {
  const w = name.split(/\s+/).filter(Boolean);
  const last = w[w.length - 1].toUpperCase();
  const tail = last.length > 2 && last.endsWith('S') ? last[last.length - 2] : last[last.length - 1];
  return (w.map((x) => x[0]).join('') + tail).toUpperCase().slice(0, 3);
};

// Team Config: primary (tile background), secondary (edge stripe), square logo, wordmark (open/stinger/watermark).
export const GAME_D = {
  ...GAME,
  home: {...GAME.home, abbr: 'IPK', primary: '#1E3160', secondary: P.logoBlue, square: 'icepak_crest.png', wordmark: 'icepak_wordmark.png'},
  away: {...GAME.away, abbr: acronym(GAME.away.name), primary: '#C8102E', secondary: P.white, square: 'saltyboys_logo.png', wordmark: null},
  oppGoal: {...GAME.oppGoal, scorer: 'H. SKATER'},
  penalty: {...GAME.penalty, name: 'K. SKATER'},
  final: {
    ...GAME.final,
    // skaters: [num, name, G, A, PIM] — eligible when they have points or PIM; goalie saves come from the
    // Scoresheet GOALKEEPING section when it is filled in.
    table: {
      home: {skaters: [[22, 'A. SKATER', 2, 1, 0], [17, 'B. SKATER', 1, 2, 2], [8, 'D. SKATER', 0, 3, 0], [5, 'E. SKATER', 1, 1, 0], [11, 'C. SKATER', 1, 0, 0], [3, 'F. SKATER', 0, 0, 2]],
        goalie: [30, 'G. GOALIE', 24, 2]},
      away: {skaters: [[9, 'H. SKATER', 1, 0, 0], [12, 'J. SKATER', 1, 0, 0], [4, 'K. SKATER', 0, 1, 2]],
        goalie: [1, 'L. GOALIE', 18, 5]},
    },
  },
};

// Reveal masks with generous negative insets so skewed corners and outlines are never clipped.
const PAD = 90;
const wipeX = (t) => `inset(-${PAD}px calc(${(1 - t) * 115}% - ${t * PAD}px) -${PAD}px -${PAD}px)`;
const dropY = (t) => `inset(-${PAD}px -${PAD}px calc(${(1 - t) * 115}% - ${t * PAD}px) -${PAD}px)`;

const Slab = ({children, style, outline = true}) => (
  <div style={{display: 'flex', width: 'fit-content', transform: `skewX(${SK}deg)`, outline: outline ? `3px solid ${P.white}` : 'none',
    boxShadow: '0 10px 30px rgba(0,0,0,.45)', ...style}}>{children}</div>
);
const Cell = ({bg, w, h, children, style}) => (
  <div style={{background: bg, width: w, height: h, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', flexShrink: 0, ...style}}>
    <div style={{transform: `skewX(${-SK}deg)`, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12}}>{children}</div>
  </div>
);
const Pic = ({src, h, style}) => <Img src={staticFile(src)} style={{height: h, width: 'auto', display: 'block', ...style}} />;
const Edge = ({team, h, w = 12}) => <div style={{width: w, height: h, background: team.secondary, flexShrink: 0}} />;
// Team identity: square logo + acronym on the team PRIMARY.
const TeamChip = ({team, h, logoH, size = 30, pad = 18, abbr = true, mirror = false}) => {
  const logo = <Pic key="l" src={team.square} h={logoH} />;
  const text = abbr ? <span key="t" style={{fontSize: size, fontWeight: 900, fontStyle: 'italic', color: P.white, letterSpacing: 1}}>{team.abbr}</span> : null;
  return <Cell bg={team.primary} w="auto" h={h} style={{padding: `0 ${pad}px`}}>{mirror ? [text, logo] : [logo, text]}</Cell>;
};
const Num = ({team, h, w = 96, size = 44, n}) => (
  <Cell bg={team.primary} w={w} h={h}><span style={{fontSize: size, fontWeight: 800, color: P.white}}>{n}</span></Cell>
);
const player = (num, name) => (name ? `${num} ${name}` : `#${num}`);

const Bug = ({g, dur, period, score, changeAt}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 12);
  const [h, a] = score(f);
  const pop = changeAt != null ? Math.max(0, 1 - Math.abs(f - changeAt - 6) / 10) : 0;
  const H = 70;
  const num = {fontSize: 50, fontWeight: 800, color: P.black, fontVariantNumeric: 'tabular-nums'};
  return (
    <div style={{position: 'absolute', left: X + 20, top: Y, clipPath: wipeX(t)}}>
      <Slab>
        <Edge team={g.home} h={H} />
        <TeamChip team={g.home} h={H} logoH={58} />
        <Cell bg={TOK.score} w={86} h={H}><span style={{...num, transform: `scale(${1 + pop * .35})`}}>{h}</span></Cell>
        <Cell bg={TOK.state} w={104} h={H}><span style={{fontSize: 30, fontWeight: 800, color: P.white}}>{period}</span></Cell>
        <Cell bg={TOK.score} w={86} h={H} style={{borderLeft: `2px solid ${P.light}`}}><span style={num}>{a}</span></Cell>
        <TeamChip team={g.away} h={H} logoH={58} mirror />
        <Edge team={g.away} h={H} />
      </Slab>
    </div>
  );
};

const GoalCard = ({g, goal, dur, compact}) => {
  const f = useCurrentFrame();
  const team = g[goal.team];
  const t = inOut(f, dur, 18, 14);
  if (compact) {
    return (
      <div style={{position: 'absolute', left: X + 20, top: Y, clipPath: wipeX(t)}}>
        <Slab>
          <Edge team={team} h={70} />
          <TeamChip team={team} h={70} logoH={58} />
          <Cell bg={TOK.hero} w={170} h={70}><span style={{fontSize: 42, fontWeight: 900, fontStyle: 'italic', color: P.black}}>GOAL</span></Cell>
          <Cell bg={TOK.info} w="auto" h={70} style={{padding: '0 30px'}}><span style={{fontSize: 30, fontWeight: 800, color: P.white}}>{player(goal.num, goal.scorer)}</span></Cell>
          <Edge team={team} h={70} w={8} />
        </Slab>
      </div>
    );
  }
  const word = ramp(f, 8, 26, 0, 1, Easing.out(Easing.back(1.3)));
  const tray = inOut(f - 18, dur - 18, 14, 10);
  const H = 128;
  const n = goal.assists.length;
  return (
    <div style={{position: 'absolute', left: X + 30, top: Y}}>
      <div style={{clipPath: wipeX(t)}}>
        <Slab>
          <Edge team={team} h={H} w={16} />
          <TeamChip team={team} h={H} logoH={96} size={48} pad={28} />
          <Cell bg={TOK.hero} w={500} h={H}>
            <span style={{fontSize: 112, fontWeight: 900, fontStyle: 'italic', color: P.black, letterSpacing: 4,
              transform: `translateX(${(1 - word) * 120}px) scale(${0.85 + 0.15 * word})`, opacity: word}}>GOAL</span>
          </Cell>
          <Edge team={team} h={H} w={16} />
        </Slab>
      </div>
      <div style={{marginLeft: 150, marginTop: 8, clipPath: dropY(tray)}}>
        <Slab style={{background: TOK.info}}>
          <Num team={team} h={62} w={80} size={32} n={goal.num} />
          <Cell bg={TOK.info} w="auto" h={62} style={{padding: '0 34px'}}>
            <span style={{fontSize: 32, fontWeight: 800, color: P.white, marginRight: 14}}>{goal.scorer}</span>
            <span style={{fontSize: 30, color: P.grey}}>/</span>
            <span style={{fontSize: 24, fontWeight: 700, color: TOK.label, marginLeft: 10}}>{n === 0 ? 'UNASSISTED' : n === 1 ? 'ASSIST:' : 'ASSISTS:'}</span>
            <span style={{fontSize: 24, fontWeight: 700, color: P.white}}>{goal.assists.join(', ')}</span>
          </Cell>
        </Slab>
      </div>
    </div>
  );
};

const PenaltyCard = ({g, pen, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 12);
  const team = g[pen.team];
  const H = 70;
  return (
    <div style={{position: 'absolute', left: X + 20, top: Y, clipPath: wipeX(t)}}>
      <Slab>
        <Num team={team} h={H} n={pen.num} />
        <Cell bg={TOK.info} w="auto" h={H} style={{padding: '0 34px'}}><span style={{fontSize: 34, fontWeight: 800, color: P.white}}>{pen.name || `#${pen.num}`}</span></Cell>
        <Cell bg={TOK.state} w={130} h={H}><span style={{fontSize: 36, fontWeight: 800, color: P.white}}>{pen.minutes}:00</span></Cell>
        <Cell bg={TOK.info} w="auto" h={H} style={{padding: '0 34px'}}><span style={{fontSize: 28, fontWeight: 700, color: P.white}}>{pen.infraction}</span></Cell>
        <TeamChip team={team} h={H} logoH={58} />
        <Edge team={team} h={H} />
      </Slab>
    </div>
  );
};

const Flap = ({g, flap, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 14, 12);
  const team = g[flap.team];
  return (
    <div style={{position: 'absolute', left: X + 110, top: Y + 80, clipPath: dropY(t)}}>
      <Slab outline={false}>
        <Cell bg={TOK.hero} w={110} h={48}><span style={{fontSize: 24, fontWeight: 900, fontStyle: 'italic', color: P.black}}>GOAL</span></Cell>
        <Num team={team} h={48} w={64} size={24} n={flap.num} />
        <Cell bg={TOK.info} w="auto" h={48} style={{padding: '0 24px'}}>
          <span style={{fontSize: 24, fontWeight: 800, color: P.white}}>{flap.scorer}</span>
          <span style={{fontSize: 22, color: P.light}}>{flap.time}</span>
        </Cell>
      </Slab>
    </div>
  );
};

const ReplayTag = ({dur}) => {
  const f = useCurrentFrame();
  return (
    <div style={{position: 'absolute', left: X + 20, top: Y, opacity: inOut(f, dur, 10, 10)}}>
      <Slab outline={false}><div style={{width: 10, height: 48, background: P.logoBlue}} /><Cell bg={TOK.state} w={170} h={48}><span style={{fontSize: 24, fontWeight: 800, letterSpacing: 4, color: P.white}}>REPLAY</span></Cell></Slab>
    </div>
  );
};

// ---------- Stinger kit: seeded particle dashes + slats ----------
const rnd = (seed) => { let x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };
// Dashes fly in from both sides, decelerate, then drift (parallax); `exitAt` flings them out.
const Dashes = ({f, colors, count = 46, enterEnd = 26, exitAt = null, dur = 180, band = [0, 100]}) => (
  <AbsoluteFill style={{overflow: 'hidden', pointerEvents: 'none'}}>
    {Array.from({length: count}).map((_, i) => {
      const dir = rnd(i + 1) > 0.5 ? 1 : -1;
      const y = band[0] + rnd(i + 7) * (band[1] - band[0]);
      const len = 12 + rnd(i + 3) * (rnd(i + 5) > 0.8 ? 160 : 60);
      const th = 3 + Math.round(rnd(i + 11) * 7);
      const home = 4 + rnd(i + 13) * 92;              // resting x (%)
      const delay = Math.round(rnd(i + 17) * 12);
      const drift = (0.02 + rnd(i + 19) * 0.08) * dir;  // %/frame during hold
      const inK = ramp(f, delay, enterEnd + delay, 0, 1, Easing.out(Easing.cubic));
      let x = (dir > 0 ? -20 : 120) + (home - (dir > 0 ? -20 : 120)) * inK + drift * Math.max(0, f - enterEnd - delay);
      let o = inK;
      if (exitAt != null) {
        const ex = ramp(f, exitAt + delay / 2, dur, 0, 1, Easing.in(Easing.cubic));
        x += ex * 140 * dir; o *= 1 - ex * 0.3;
      }
      return <div key={i} style={{position: 'absolute', top: `${y}%`, left: `${x}%`, width: len, height: th,
        background: colors[i % colors.length], opacity: o, transform: `skewX(${SK}deg)`}} />;
    })}
  </AbsoluteFill>
);

// Horizontal slats that slide in from alternating sides with a stagger to assemble a solid block.
const Slats = ({f, n, colors, inStart, inDur = 16, outStart, outDur = 18, dirOut = 1, stagger = 2}) => (
  <AbsoluteFill style={{overflow: 'hidden'}}>
    {Array.from({length: n}).map((_, i) => {
      const side = i % 2 ? 1 : -1;
      const a = ramp(f, inStart + i * stagger, inStart + i * stagger + inDur, 0, 1, Easing.out(Easing.cubic));
      const b = ramp(f, outStart + i * stagger, outStart + i * stagger + outDur, 0, 1, Easing.in(Easing.cubic));
      const x = side * (1 - a) * 130 + dirOut * b * 130;
      return <div key={i} style={{position: 'absolute', left: `${-10 + x}%`, width: '120%', top: `${(i * 100) / n}%`, height: `${100 / n + 0.2}%`,
        background: colors[i % colors.length], transform: `skewX(${SK}deg)`}} />;
    })}
  </AbsoluteFill>
);

// Period transition (1 s): dash burst + slats cover the cut; crest+wordmark flash in the hold; slats exit right.
const PeriodWipe = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  const accent = team ? team.secondary : P.light;
  const base = team ? team.primary : P.navy;
  const hold = ramp(f, 20, 26) * (1 - ramp(f, 36, 42));
  return (
    <AbsoluteFill>
      <Slats f={f} n={9} colors={[P.navy, base, P.navy, P.black, base, P.navy, P.black, base, P.navy]} inStart={0} inDur={16} outStart={36} outDur={16} stagger={1} />
      <AbsoluteFill style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 40, opacity: hold, transform: `scale(${0.96 + 0.04 * hold})`}}>
        {team ? (<><Pic src={team.square} h={250} />{team.wordmark && <Pic src={team.wordmark} h={170} />}</>) : null}
      </AbsoluteFill>
      <Dashes f={f} colors={[accent, P.white, P.logoBlue, accent]} count={40} enterEnd={14} exitAt={34} dur={dur} />
    </AbsoluteFill>
  );
};

const PeriodCard = ({g, period, score, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 16, 14);
  const H = 76;
  const num = {fontSize: 46, fontWeight: 800};
  return (
    <div style={{position: 'absolute', left: '50%', bottom: 96, transform: 'translateX(-50%)', clipPath: wipeX(t)}}>
      <Slab>
        <Edge team={g.home} h={H} />
        <TeamChip team={g.home} h={H} logoH={62} />
        <Cell bg={TOK.score} w={80} h={H}><span style={num}>{score[0]}</span></Cell>
        <Cell bg={TOK.state} w="auto" h={H} style={{padding: '0 30px'}}><span style={{fontSize: 30, fontWeight: 900, fontStyle: 'italic', color: P.white, letterSpacing: 2}}>{period}</span></Cell>
        <Cell bg={TOK.score} w={80} h={H} style={{borderLeft: `2px solid ${P.light}`}}><span style={num}>{score[1]}</span></Cell>
        <TeamChip team={g.away} h={H} logoH={62} mirror />
        <Edge team={g.away} h={H} />
      </Slab>
    </div>
  );
};

// Open (3 s): dash burst → slats assemble the slab → crest drops in, wordmark mask-reveals with a glint,
// end caps grow then shrink (Sportsnet), matchup plate assembles from segments → slats + dashes exit.
const Open = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  const opp = team ? g[g.focus === 'home' ? 'away' : 'home'] : null;
  const main = team || g.home;
  const exitAt = dur - 26;
  const out = ramp(f, exitAt, dur, 0, 1, Easing.in(Easing.cubic));
  const crest = ramp(f, 22, 40, 0, 1, Easing.out(Easing.back(1.5)));
  const reveal = ramp(f, 30, 52, 0, 1, Easing.inOut(Easing.cubic));
  const glint = ramp(f, 58, 92, -30, 130, Easing.inOut(Easing.quad));
  const cap = ramp(f, 16, 30, 0, 1, Easing.out(Easing.cubic)) * (1 - 0.8 * ramp(f, 40, 110, 0, 1));
  const segs = [ramp(f, 56, 70), ramp(f, 60, 74), ramp(f, 64, 78)];
  const W = 1240, H = 290, TOP = 300;
  return (
    <AbsoluteFill style={{background: `repeating-linear-gradient(${90 + SK}deg, rgba(255,255,255,.025) 0 2px, transparent 2px 26px), radial-gradient(circle at 50% 45%, #15283f 0%, ${P.black} 72%)`,
      opacity: 1 - ramp(f, dur - 6, dur, 0, 1), overflow: 'hidden'}}>
      <Dashes f={f} colors={[main.secondary, P.white, P.logoBlue, P.grey]} count={54} enterEnd={24} exitAt={exitAt} dur={dur} />
      {/* main slab assembled from slats */}
      <div style={{position: 'absolute', left: (1920 - W) / 2, top: TOP, width: W, height: H, transform: `translateX(${out * 130}vw)`}}>
        <div style={{position: 'absolute', inset: 0, transform: `skewX(${SK}deg)`, overflow: 'hidden', outline: `3px solid ${P.white}`, outlineOffset: -1}}>
          <Slats f={f} n={6} colors={[main.primary, P.navy, main.primary, P.navy, main.primary, P.navy]} inStart={6} inDur={16} outStart={9999} stagger={2} />
          <div style={{position: 'absolute', inset: 0, background: `linear-gradient(180deg, ${main.primary} 0%, ${P.navy} 100%)`, opacity: ramp(f, 26, 40)}} />
        </div>
        {/* end caps: grow, then shrink to thin lines */}
        <div style={{position: 'absolute', top: 0, bottom: 0, left: -6, width: 8 + 36 * cap, background: main.secondary, transform: `skewX(${SK}deg)`, transformOrigin: 'bottom'}} />
        <div style={{position: 'absolute', top: 0, bottom: 0, right: -6, width: 8 + 36 * cap, background: main.secondary, transform: `skewX(${SK}deg)`, transformOrigin: 'bottom'}} />
        <div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 46}}>
          {team ? (<>
            <div style={{transform: `translateY(${(1 - crest) * -60}px) scale(${0.7 + 0.3 * crest})`, opacity: crest}}><Pic src={team.square} h={230} /></div>
            {team.wordmark && (
              <div style={{position: 'relative', clipPath: `inset(-20px ${(1 - reveal) * 100}% -20px -20px)`}}>
                <Pic src={team.wordmark} h={150} />
                <div style={{position: 'absolute', top: -30, bottom: -30, left: `${glint}%`, width: 80, transform: 'skewX(-20deg)',
                  background: 'linear-gradient(90deg, transparent, rgba(255,255,255,.8), transparent)', mixBlendMode: 'overlay'}} />
              </div>
            )}
          </>) : (
            <div style={{opacity: crest, display: 'flex', alignItems: 'center', gap: 60}}>
              <Pic src={g.home.square} h={200} /><span style={{color: P.white, fontSize: 64, fontWeight: 900, fontStyle: 'italic'}}>VS</span><Pic src={g.away.square} h={200} />
            </div>
          )}
        </div>
      </div>
      {/* matchup plate: three segments slide in from alternating sides */}
      <div style={{position: 'absolute', left: 0, right: 0, top: TOP + H + 44, display: 'flex', justifyContent: 'center', transform: `translateX(${out * -130}vw)`, opacity: ramp(f, 54, 60)}}>
        <Slab>
          <div style={{transform: `translateX(${(1 - segs[0]) * -400}px)`, opacity: segs[0]}}>
            <Cell bg={TOK.state} w={90} h={76}><span style={{color: P.white, fontSize: 32, fontWeight: 900, fontStyle: 'italic'}}>VS</span></Cell>
          </div>
          <div style={{transform: `translateX(${(1 - segs[1]) * 400}px)`, opacity: segs[1]}}>
            {team ? <TeamChip team={opp} h={76} logoH={60} abbr={false} /> : <Cell bg={TOK.info} w={260} h={76}><span style={{color: P.white, fontSize: 32, fontWeight: 800}}>GAME RECAP</span></Cell>}
          </div>
          {team && <div style={{transform: `translateX(${(1 - segs[1]) * 400}px)`, opacity: segs[1]}}>
            <Cell bg={TOK.info} w="auto" h={76} style={{padding: '0 34px'}}><span style={{color: P.white, fontSize: 36, fontWeight: 800}}>{opp.name}</span></Cell>
          </div>}
          <div style={{transform: `translateX(${(1 - segs[2]) * -400}px)`, opacity: segs[2]}}>
            <Cell bg={TOK.state} w="auto" h={76} style={{padding: '0 30px'}}><span style={{color: P.light, fontSize: 26, fontWeight: 700}}>{g.date}</span></Cell>
          </div>
        </Slab>
      </div>
    </AbsoluteFill>
  );
};

// FINAL: score slab + both teams' game-summary tables (always both, regardless of Perspective).
// Skaters with points or PIM (sorted by PTS, then G), then the goalie with saves.
const ScoringTable = ({team, data, t, delay}) => {
  const f = useCurrentFrame();
  const rows = data.skaters.filter((r) => r[2] + r[3] + r[4] > 0).sort((a, b) => (b[2] + b[3]) - (a[2] + a[3]) || b[2] - a[2] || b[4] - a[4]);
  const col = {width: 62, textAlign: 'center', fontVariantNumeric: 'tabular-nums'};
  const RowShell = ({i, num, name, children, tag}) => {
    const k = ramp(f, delay + i * 4, delay + i * 4 + 14);
    return (
      <div style={{display: 'flex', alignItems: 'center', height: 54, background: TOK.info, color: P.white,
        borderBottom: '1px solid rgba(255,255,255,.08)', opacity: k * t, transform: `translateX(${(1 - k) * -40}px)`}}>
        <div style={{width: 62, alignSelf: 'stretch', background: team.primary, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 24, fontWeight: 800, borderRight: `3px solid ${team.secondary}`}}>{num}</div>
        <div style={{flex: 1, paddingLeft: 18, fontSize: 25, fontWeight: 700, letterSpacing: .5, display: 'flex', alignItems: 'center', gap: 10}}>
          {name}{tag && <span style={{fontSize: 15, fontWeight: 800, color: TOK.label, letterSpacing: 2}}>{tag}</span>}
        </div>
        {children}
        <div style={{width: 14}} />
      </div>
    );
  };
  const [gn, gname, sv, ga] = data.goalie || [];
  return (
    <div style={{width: 660}}>
      <div style={{display: 'flex', alignItems: 'center', height: 56, background: team.primary, color: P.white}}>
        <div style={{width: 10, alignSelf: 'stretch', background: team.secondary}} />
        <div style={{display: 'flex', alignItems: 'center', gap: 12, paddingLeft: 16, flex: 1}}>
          <Pic src={team.square} h={42} />
          <span style={{fontSize: 22, fontWeight: 900, fontStyle: 'italic', letterSpacing: 2}}>{team.name}</span>
        </div>
        {['G', 'A', 'PTS', 'PIM'].map((h) => <div key={h} style={{...col, fontSize: 17, fontWeight: 800, letterSpacing: 2, opacity: .85}}>{h}</div>)}
        <div style={{width: 14}} />
      </div>
      {rows.map(([num, name, gl, as, pim], i) => (
        <RowShell key={name} i={i} num={num} name={name}>
          <div style={{...col, fontSize: 25, fontWeight: 600, color: P.light}}>{gl}</div>
          <div style={{...col, fontSize: 25, fontWeight: 600, color: P.light}}>{as}</div>
          <div style={{...col, fontSize: 27, fontWeight: 900}}>{gl + as}</div>
          <div style={{...col, fontSize: 25, fontWeight: 600, color: P.light}}>{pim}</div>
        </RowShell>
      ))}
      {data.goalie && (
        <RowShell i={rows.length} num={gn} name={gname} tag="G">
          <div style={{width: 248, display: 'flex', justifyContent: 'center', alignItems: 'baseline', gap: 10, fontVariantNumeric: 'tabular-nums'}}>
            <span style={{fontSize: 27, fontWeight: 900}}>{sv}</span><span style={{fontSize: 16, fontWeight: 800, color: P.light, letterSpacing: 2}}>SV</span>
            <span style={{fontSize: 22, color: P.grey}}>·</span>
            <span style={{fontSize: 25, fontWeight: 600, color: P.light}}>{(sv / (sv + ga)).toFixed(3).replace(/^0/, '')}</span><span style={{fontSize: 16, fontWeight: 800, color: P.light, letterSpacing: 2}}>SV%</span>
          </div>
        </RowShell>
      )}
    </div>
  );
};

const FinalCard = ({g, dur}) => {
  const f = useCurrentFrame();
  const t = inOut(f, dur, 20, 16);
  const H = 110;
  return (
    <AbsoluteFill style={{background: `rgba(10,10,10,${0.78 * t})`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column'}}>
      <div style={{clipPath: wipeX(t)}}>
        <Slab>
          <Edge team={g.home} h={H} w={16} />
          <TeamChip team={g.home} h={H} logoH={92} size={46} pad={26} />
          <Cell bg={TOK.score} w={130} h={H}><span style={{fontSize: 80, fontWeight: 900}}>{g.final.home}</span></Cell>
          <Cell bg={TOK.state} w={170} h={H}><span style={{fontSize: 38, fontWeight: 900, fontStyle: 'italic', color: P.white}}>FINAL</span></Cell>
          <Cell bg={TOK.score} w={130} h={H} style={{borderLeft: `2px solid ${P.light}`}}><span style={{fontSize: 80, fontWeight: 900}}>{g.final.away}</span></Cell>
          <TeamChip team={g.away} h={H} logoH={92} size={46} pad={26} mirror />
          <Edge team={g.away} h={H} w={16} />
        </Slab>
      </div>
      <div style={{display: 'flex', gap: 36, marginTop: 34, opacity: t}}>
        {['home', 'away'].map((c) => <ScoringTable key={c} team={g[c]} data={g.final.table[c]} t={t} delay={14} />)}
      </div>
    </AbsoluteFill>
  );
};

const Watermark = ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g[g.focus];
  return (
    <div style={{position: 'absolute', right: 46, bottom: 36, opacity: 0.55 * inOut(f, dur, 20, 20), filter: 'grayscale(1) brightness(0.9) contrast(1.1)'}}>
      <Pic src={team.wordmark || team.square} h={64} />
    </div>
  );
};

export const VariantD = {name: 'Ice Pak Slab', font: fontFamily, Bug, GoalCard, PenaltyCard, Flap, ReplayTag, PeriodWipe: makePeriodWipe(P), PeriodCard,
  Open: makeOpen(TeamChip, Slab, Cell, TOK, P), FinalCard, Watermark};
