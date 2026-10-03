// PROTOTYPE — Stinger kit for Variant D (round 5). Adds what broadcast stingers have and flat CSS lacks:
// 3D perspective + camera drift, directional motion blur (SVG filter), light streaks and flares,
// ice-shard particles flying through depth with depth-of-field blur, an impact flash + shockwave,
// line-drawn outlines, and a speed-ramped whip exit.
import React from 'react';
import {AbsoluteFill, useCurrentFrame, Easing, Img, staticFile} from 'remotion';
import {ramp} from './shared';

const SK = -16;
const BLUE = '#5AA4D0', NAVY = '#0D1B2A', WHITE = '#FFFFFF', BLACK = '#0A0A0A';
const rnd = (seed) => { const x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };
const Pic = ({src, h, style}) => <Img src={staticFile(src)} style={{height: h, width: 'auto', display: 'block', ...style}} />;

// Horizontal motion blur via an SVG filter (CSS blur() is isotropic). One filter per blur amount.
const MotionBlurDefs = ({ids}) => (
  <svg width="0" height="0" style={{position: 'absolute'}}>
    <defs>
      {ids.map(([id, v]) => (
        <filter key={id} id={id} x="-50%" y="-10%" width="200%" height="120%"><feGaussianBlur stdDeviation={`${Math.max(0.01, v)} 0`} /></filter>
      ))}
    </defs>
  </svg>
);

const Background = ({f, tint}) => {
  const lx = 30 + 40 * Math.sin(f / 70), ly = 40 + 12 * Math.cos(f / 55);
  return (
    <AbsoluteFill style={{background: `radial-gradient(ellipse at ${lx}% ${ly}%, ${tint}55 0%, transparent 45%),
      radial-gradient(circle at 50% 50%, #14263c 0%, ${BLACK} 75%)`}}>
      <AbsoluteFill style={{background: `repeating-linear-gradient(${90 + SK}deg, rgba(255,255,255,.028) 0 2px, transparent 2px 22px)`,
        transform: `translateX(${-f * 0.6}px)`}} />
      <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,.65) 100%)'}} />
    </AbsoluteFill>
  );
};

// A light streak that draws across the frame with a glowing head.
const Streak = ({f, start, dur, y, color = WHITE, thick = 3, dir = 1, len = 1.2}) => {
  const k = ramp(f, start, start + dur, 0, 1, Easing.inOut(Easing.cubic));
  const fade = 1 - ramp(f, start + dur * 0.7, start + dur * 1.6, 0, 1);
  if (f < start || fade <= 0) return null;
  const head = dir > 0 ? -10 + k * 120 : 110 - k * 120;
  return (
    <div style={{position: 'absolute', top: y, left: 0, right: 0, height: thick, opacity: fade}}>
      <div style={{position: 'absolute', height: thick, top: 0, left: `${dir > 0 ? head - 40 * len : head}%`, width: `${40 * len}%`,
        background: `linear-gradient(${dir > 0 ? 90 : 270}deg, transparent, ${color})`, boxShadow: `0 0 12px ${color}, 0 0 28px ${color}`}} />
      <div style={{position: 'absolute', top: -14 + thick / 2, left: `calc(${head}% - 14px)`, width: 28, height: 28, borderRadius: 14,
        background: `radial-gradient(circle, ${WHITE} 0%, ${color} 35%, transparent 70%)`}} />
    </div>
  );
};

// Ice shards: seeded polygons flying from depth to rest positions around the slab, with DOF blur,
// slow parallax drift in the hold, and a fling toward camera on exit.
const Shards = ({f, count = 26, inStart = 4, inDur = 26, exitAt, dur, colors, burstAt}) => (
  <AbsoluteFill style={{perspective: 1200, overflow: 'hidden'}}>
    {Array.from({length: count}).map((_, i) => {
      const r = (k) => rnd(i * 7 + k);
      const size = 14 + r(1) * (r(2) > 0.9 ? 90 : 46);
      const pts = [[0, 0], [1, 0.2 + r(3) * 0.3], [0.35 + r(4) * 0.3, 1]].map(([x, y]) => `${x * size},${y * size * (0.5 + r(5))}`).join(' ');
      const restX = r(6) * 1920, restY = r(7) * 1080;
      const z0 = -1800 - r(8) * 1200, zr = -400 + r(9) * 700;
      const delay = Math.round(r(10) * 10);
      const k = ramp(f, inStart + delay, inStart + delay + inDur, 0, 1, Easing.out(Easing.cubic));
      const burst = burstAt != null ? ramp(f, burstAt, burstAt + 18, 0, 1, Easing.out(Easing.quad)) : 0;
      const ex = exitAt != null ? ramp(f, exitAt + delay / 2, dur, 0, 1, Easing.in(Easing.cubic)) : 0;
      const cx = 960, cy = 540;
      const x = cx + (restX - cx) * k + (restX - cx) * 0.08 * burst + Math.sin((f + i * 13) / 40) * 12 + ex * (restX - cx) * 1.2;
      const y = cy + (restY - cy) * k + (restY - cy) * 0.08 * burst + Math.cos((f + i * 7) / 50) * 8 + ex * (restY - cy) * 1.2;
      const z = z0 + (zr - z0) * k + ex * 1400;
      const rot = r(11) * 360 + f * (r(12) - 0.5) * 1.2 + (1 - k) * 220;
      const blur = Math.min(14, Math.abs(z - 0) / 180) * (0.6 + r(13) * 0.4);
      const o = Math.min(1, k * 1.4) * (1 - ex * 0.6) * (0.35 + r(14) * 0.45) * (size > 60 ? 0.55 : 1);
      const col = colors[i % colors.length];
      return (
        <svg key={i} width={size} height={size * 1.6} style={{position: 'absolute', left: x, top: y, overflow: 'visible', opacity: o,
          transform: `translateZ(${z}px) rotate(${rot}deg)`, filter: `blur(${blur}px)`}}>
          <polygon points={pts} fill={col} fillOpacity={0.5} stroke={BLUE} strokeOpacity={0.9} strokeWidth={1.4} />
        </svg>
      );
    })}
  </AbsoluteFill>
);

// Small dashes (Sportsnet): fly in from both sides, drift, fling out.
const Dashes = ({f, colors, count = 50, enterEnd = 24, exitAt = null, dur = 180}) => (
  <AbsoluteFill style={{overflow: 'hidden', pointerEvents: 'none'}}>
    {Array.from({length: count}).map((_, i) => {
      const dir = rnd(i + 1) > 0.5 ? 1 : -1;
      const y = rnd(i + 7) * 100;
      const len = 10 + rnd(i + 3) * (rnd(i + 5) > 0.8 ? 150 : 50);
      const th = 2 + Math.round(rnd(i + 11) * 6);
      const home = 4 + rnd(i + 13) * 92;
      const delay = Math.round(rnd(i + 17) * 12);
      const drift = (0.02 + rnd(i + 19) * 0.07) * dir;
      const inK = ramp(f, delay, enterEnd + delay, 0, 1, Easing.out(Easing.cubic));
      let x = (dir > 0 ? -20 : 120) + (home - (dir > 0 ? -20 : 120)) * inK + drift * Math.max(0, f - enterEnd - delay);
      let o = inK * (0.5 + rnd(i + 23) * 0.5);
      if (exitAt != null) { const ex = ramp(f, exitAt + delay / 2, dur, 0, 1, Easing.in(Easing.cubic)); x += ex * 140 * dir; }
      return <div key={i} style={{position: 'absolute', top: `${y}%`, left: `${x}%`, width: len, height: th, background: colors[i % colors.length],
        opacity: o, transform: `skewX(${SK}deg)`, boxShadow: rnd(i + 29) > 0.8 ? `0 0 10px ${colors[i % colors.length]}` : 'none'}} />;
    })}
  </AbsoluteFill>
);

const Flash = ({f, at, strength = 0.45, dur = 10}) => {
  const o = f < at ? 0 : strength * (1 - ramp(f, at, at + dur, 0, 1, Easing.out(Easing.quad)));
  return <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 45%, ${WHITE} 0%, ${BLUE} 60%, transparent 100%)`, opacity: o, mixBlendMode: 'screen'}} />;
};

const Shockwave = ({f, at, color = BLUE, cx = 960, cy = 540}) => {
  if (f < at) return null;
  const k = ramp(f, at, at + 26, 0, 1, Easing.out(Easing.cubic));
  const w = 200 + k * 1800, h = w * 0.36;
  return (
    <div style={{position: 'absolute', left: cx - w / 2, top: cy - h / 2, width: w, height: h, borderRadius: '50%',
      border: `${6 * (1 - k) + 1}px solid ${color}`, opacity: 1 - k, boxShadow: `0 0 30px ${color}`, transform: `skewX(${SK}deg)`}} />
  );
};

// Parallelogram outline that draws on (SVG stroke-dash), then stays as a thin frame.
const DrawOn = ({f, start, dur, w, h, color = WHITE, width = 3}) => {
  const k = ramp(f, start, start + dur, 0, 1, Easing.inOut(Easing.cubic));
  const off = Math.tan((-SK * Math.PI) / 180) * h;
  const pts = `${off},0 ${w + off},0 ${w},${h} 0,${h} ${off},0`;
  const len = 2 * (w + Math.hypot(off, h));
  return (
    <svg width={w + off} height={h} style={{position: 'absolute', left: -off / 2, top: 0, overflow: 'visible'}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={width} strokeDasharray={len} strokeDashoffset={len * (1 - k)}
        style={{filter: `drop-shadow(0 0 6px ${color})`}} />
    </svg>
  );
};

// ---------------- OPEN (3 s) ----------------
export const makeOpen = (TeamChip, Slab, Cell, TOK, P) => ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  const opp = team ? g[g.focus === 'home' ? 'away' : 'home'] : null;
  const main = team || g.home;
  const exitAt = dur - 24;
  const W = 1300, H = 300, TOP = 310;
  // speed-ramped camera: fast in, slow drift, whip out
  const inK = ramp(f, 10, 34, 0, 1, Easing.out(Easing.exp));
  const drift = ramp(f, 34, exitAt, 0, 1, Easing.linear);
  const out = ramp(f, exitAt, dur, 0, 1, Easing.in(Easing.exp));
  const rotY = (1 - inK) * 55 + (4 - 8 * drift) * inK - out * 25;
  const z = (1 - inK) * -900 + drift * 60;
  const tx = out * 2400;
  const mbIn = (1 - ramp(f, 12, 30)) * 30, mbOut = out * 60;
  const crest = ramp(f, 22, 36, 0, 1, Easing.out(Easing.back(1.8)));
  const reveal = ramp(f, 30, 50, 0, 1, Easing.inOut(Easing.cubic));
  const glint = ramp(f, 60, 96, -30, 130, Easing.inOut(Easing.quad));
  const fill = ramp(f, 14, 30, 0, 1, Easing.out(Easing.cubic));
  const cap = ramp(f, 26, 36, 0, 1, Easing.out(Easing.cubic)) * (1 - 0.8 * ramp(f, 46, 120, 0, 1));
  const segs = [0, 1, 2].map((i) => ramp(f, 54 + i * 5, 68 + i * 5, 0, 1, Easing.out(Easing.back(1.4))));
  const IMPACT = 34;
  return (
    <AbsoluteFill style={{overflow: 'hidden', opacity: 1 - ramp(f, dur - 5, dur, 0, 1)}}>
      <MotionBlurDefs ids={[['mbIn', mbIn], ['mbOut', mbOut]]} />
      <Background f={f} tint={main.secondary} />
      <Streak f={f} start={0} dur={16} y={TOP - 30} color={main.secondary} thick={3} />
      <Streak f={f} start={4} dur={18} y={TOP + H + 22} color={WHITE} thick={2} dir={-1} />
      <Streak f={f} start={70} dur={30} y={TOP + H + 140} color={main.secondary} thick={2} len={0.6} />
      <Shards f={f} colors={[main.secondary, BLUE, '#9FD3F0', main.primary]} exitAt={exitAt} dur={dur} burstAt={IMPACT} />
      <Dashes f={f} colors={[main.secondary, WHITE, BLUE, '#6B6B6B']} exitAt={exitAt} dur={dur} />
      {/* 3D stage */}
      <AbsoluteFill style={{perspective: 1600, filter: out > 0 ? 'url(#mbOut)' : inK < 1 ? 'url(#mbIn)' : 'none'}}>
        <div style={{position: 'absolute', left: (1920 - W) / 2, top: TOP, width: W, height: H,
          transform: `translateX(${tx}px) translateZ(${z}px) rotateY(${rotY}deg)`, transformStyle: 'preserve-3d'}}>
          <div style={{position: 'absolute', inset: 0, transform: `skewX(${SK}deg)`, overflow: 'hidden', boxShadow: '0 30px 80px rgba(0,0,0,.6)'}}>
            <div style={{position: 'absolute', inset: 0, clipPath: `inset(0 ${(1 - fill) * 100}% 0 0)`,
              background: `linear-gradient(180deg, ${main.primary} 0%, ${NAVY} 100%)`}}>
              <div style={{position: 'absolute', inset: 0, background: `repeating-linear-gradient(${90 + SK}deg, rgba(255,255,255,.04) 0 2px, transparent 2px 14px)`}} />
              <div style={{position: 'absolute', top: 0, bottom: 0, left: `${fill * 100 - 4}%`, width: '4%', background: `linear-gradient(90deg, transparent, ${WHITE})`, opacity: 1 - fill}} />
              <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: '45%', background: 'linear-gradient(180deg, rgba(255,255,255,.14), transparent)'}} />
            </div>
          </div>
          <DrawOn f={f} start={8} dur={16} w={W} h={H} color={WHITE} />
          <div style={{position: 'absolute', top: 0, bottom: 0, left: -10, width: 8 + 40 * cap, background: main.secondary, transform: `skewX(${SK}deg)`, boxShadow: `0 0 20px ${main.secondary}`}} />
          <div style={{position: 'absolute', top: 0, bottom: 0, right: -10, width: 8 + 40 * cap, background: main.secondary, transform: `skewX(${SK}deg)`, boxShadow: `0 0 20px ${main.secondary}`}} />
          <div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 50}}>
            {team ? (<>
              <div style={{transform: `scale(${2.2 - 1.2 * crest})`, opacity: Math.min(1, crest * 2), filter: `blur(${(1 - crest) * 10}px) drop-shadow(0 0 ${18 * (1 - drift)}px ${main.secondary})`}}>
                <Pic src={team.square} h={240} />
              </div>
              {team.wordmark && (
                <div style={{position: 'relative', clipPath: `inset(-30px ${(1 - reveal) * 100}% -30px -30px)`, transform: `translateX(${(1 - reveal) * -60}px)`}}>
                  <Pic src={team.wordmark} h={150} />
                  <div style={{position: 'absolute', top: -40, bottom: -40, left: `${glint}%`, width: 110, transform: 'skewX(-20deg)',
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,.85), transparent)', mixBlendMode: 'overlay'}} />
                  <div style={{position: 'absolute', top: '50%', left: `${glint}%`, width: 220, height: 220, marginTop: -110, marginLeft: -110,
                    background: `radial-gradient(circle, rgba(255,255,255,${0.55 * Math.sin(Math.PI * Math.max(0, Math.min(1, (glint + 30) / 160)))}) 0%, transparent 60%)`}} />
                </div>
              )}
            </>) : (
              <div style={{opacity: crest, display: 'flex', alignItems: 'center', gap: 60}}>
                <Pic src={g.home.square} h={200} /><span style={{color: WHITE, fontSize: 64, fontWeight: 900, fontStyle: 'italic'}}>VS</span><Pic src={g.away.square} h={200} />
              </div>
            )}
          </div>
        </div>
      </AbsoluteFill>
      <Shockwave f={f} at={IMPACT} color={main.secondary} cy={TOP + H / 2} />
      <Flash f={f} at={IMPACT} strength={0.18} dur={7} />
      {/* matchup plate: segments flip in (rotateX) one after another */}
      <div style={{position: 'absolute', left: 0, right: 0, top: TOP + H + 48, display: 'flex', justifyContent: 'center', perspective: 900,
        transform: `translateX(${-out * 2400}px)`, filter: out > 0 ? 'url(#mbOut)' : 'none'}}>
        <div style={{display: 'flex', gap: 6}}>
          {[
            <Slab key="vs"><Cell bg={TOK.state} w={90} h={76}><span style={{color: WHITE, fontSize: 32, fontWeight: 900, fontStyle: 'italic'}}>VS</span></Cell></Slab>,
            team ? <Slab key="opp"><TeamChip team={opp} h={76} logoH={60} abbr={false} /><Cell bg={TOK.info} w="auto" h={76} style={{padding: '0 34px'}}><span style={{color: WHITE, fontSize: 36, fontWeight: 800}}>{opp.name}</span></Cell></Slab>
              : <Slab key="opp"><Cell bg={TOK.info} w={280} h={76}><span style={{color: WHITE, fontSize: 32, fontWeight: 800}}>GAME RECAP</span></Cell></Slab>,
            <Slab key="date"><Cell bg={TOK.state} w="auto" h={76} style={{padding: '0 30px'}}><span style={{color: P.light, fontSize: 26, fontWeight: 700}}>{g.date}</span></Cell></Slab>,
          ].map((el, i) => (
            <div key={i} style={{transform: `rotateX(${(1 - segs[i]) * 90}deg) translateY(${(1 - segs[i]) * 30}px)`, opacity: Math.min(1, segs[i] * 1.5), transformOrigin: 'top'}}>{el}</div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ---------------- PERIOD TRANSITION (1 s) ----------------
// Whip-in with motion blur → impact (flash + shockwave) with crest + wordmark over the cut → whip-out.
export const makePeriodWipe = (P) => ({g, dur}) => {
  const f = useCurrentFrame();
  const team = g.focus ? g[g.focus] : null;
  const main = team || g.home;
  const inK = ramp(f, 0, 18, 0, 1, Easing.out(Easing.exp));
  const outK = ramp(f, 32, dur, 0, 1, Easing.in(Easing.cubic));
  const x = (1 - inK) * -3700 + outK * 3700;
  const mb = (1 - inK) * 50 + outK * 60;
  const logo = ramp(f, 14, 24, 0, 1, Easing.out(Easing.back(1.6)));
  // pixel layout at rest (x = 0): navy body covers the frame incl. skew; accents lead and trail.
  const bands = [
    {c: main.secondary, l: -1260, w: 60}, {c: WHITE, l: -1180, w: 16}, {c: main.primary, l: -1150, w: 90},
    {c: NAVY, l: -1060, w: 3340, main: true},
    {c: main.primary, l: 2280, w: 110}, {c: WHITE, l: 2410, w: 18}, {c: main.secondary, l: 2450, w: 70},
  ];
  return (
    <AbsoluteFill style={{overflow: 'hidden'}}>
      <MotionBlurDefs ids={[['mbP', mb]]} />
      <AbsoluteFill style={{filter: 'url(#mbP)', transform: `translateX(${x}px)`}}>
        {bands.map((b, i) => (
          <div key={i} style={{position: 'absolute', top: -80, bottom: -80, left: b.l, width: b.w, background: b.c, transform: `skewX(${SK}deg)`}}>
            {b.main && <div style={{position: 'absolute', inset: 0, background: `repeating-linear-gradient(${90 + SK}deg, rgba(255,255,255,.035) 0 2px, transparent 2px 16px), linear-gradient(180deg, ${main.primary}88, transparent 60%)`}} />}
          </div>
        ))}
      </AbsoluteFill>
      <Dashes f={f} colors={[main.secondary, WHITE, BLUE]} count={36} enterEnd={12} exitAt={36} dur={dur} />
      <AbsoluteFill style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 44,
        opacity: logo, transform: `translateX(${outK * 3700}px) scale(${0.9 + 0.1 * logo})`, filter: `drop-shadow(0 0 18px ${main.secondary})`}}>
        {team ? (<><Pic src={team.square} h={250} />{team.wordmark && <Pic src={team.wordmark} h={160} />}</>) : null}
      </AbsoluteFill>
      <Shockwave f={f} at={20} color={main.secondary} />
      <Flash f={f} at={20} strength={0.16} dur={6} />
    </AbsoluteFill>
  );
};
