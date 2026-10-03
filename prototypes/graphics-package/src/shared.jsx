// PROTOTYPE — throwaway. Shared game data, timing helpers, and the storyboard
// (the decided Recap edit grammar) that every graphics variant plays through.
import React from 'react';
import {
  AbsoluteFill, Sequence, OffthreadVideo, staticFile, useCurrentFrame,
  interpolate, Easing, Img,
} from 'remotion';

export const FPS = 60;
export const s = (sec) => Math.round(sec * FPS);
export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};

// 0→1 on the way in, 1→0 on the way out.
export const inOut = (f, dur, inF = 14, outF = 12) =>
  Math.min(
    interpolate(f, [0, inF], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)}),
    interpolate(f, [dur - outF, dur], [1, 0], {...clamp, easing: Easing.in(Easing.cubic)}),
  );
export const ramp = (f, a, b, from = 0, to = 1, ease = Easing.out(Easing.cubic)) =>
  interpolate(f, [a, b], [from, to], {...clamp, easing: ease});

export const GAME = {
  league: 'SL',
  date: 'FEB 28, 2026',
  home: {name: 'ICE PAK', short: 'ICE PAK', abbr: 'ICE', color: '#12305E', accent: '#5FB8E6', logo: 'icepak.png'},
  away: {name: 'SALTY BOYS', short: 'SALTY', abbr: 'SLT', color: '#B5122B', accent: '#F4F4F4', logo: null},
  goal: {team: 'home', num: 17, scorer: 'B. SKATER', assists: ['C. SKATER', 'D. SKATER'], time: '1ST 8:42'},
  oppGoal: {team: 'away', num: 9, scorer: null, assists: [], time: '1ST 16:20'},
  penalty: {team: 'away', num: 4, infraction: 'SLASHING', minutes: 2, time: '1ST 11:05'},
  flap: {team: 'home', num: 22, scorer: 'A. SKATER', time: '2ND 3:30'},
  final: {
    home: 5, away: 2,
    leaders: {
      home: [['A. SKATER', '2G 1A'], ['B. SKATER', '1G 2A'], ['D. SKATER', '3A'], ['E. SKATER', '1G 1A']],
      away: [['#9', '1G'], ['#12', '1G'], ['#4', '1A']],
    },
  },
};

export const other = (t) => (t === 'home' ? 'away' : 'home');
export const scorerLine = (g) => (g.scorer ? `#${g.num} ${g.scorer}` : `#${g.num}`);

// Logo if the Team Config has one, else a styled wordmark (decided rule).
export const Mark = ({team, size = 60, wordStyle = {}}) =>
  team.logo ? (
    <Img src={staticFile(team.logo)} style={{height: size, width: 'auto', display: 'block'}} />
  ) : (
    <div style={{fontSize: size * 0.42, fontWeight: 800, color: '#fff', letterSpacing: 1, lineHeight: 1, whiteSpace: 'nowrap', ...wordStyle}}>
      {team.name}
    </div>
  );

const Clip = ({src, rate = 1, startFrom = 0, fadeIn = 0}) => {
  const f = useCurrentFrame();
  const o = fadeIn ? ramp(f, 0, fadeIn, 0, 1, Easing.linear) : 1;
  return (
    <AbsoluteFill style={{opacity: o}}>
      <OffthreadVideo src={staticFile(src)} playbackRate={rate} startFrom={startFrom} muted
        style={{width: '100%', height: '100%', objectFit: 'cover'}} />
    </AbsoluteFill>
  );
};

// Storyboard: open → live play → focus goal (card over celebration) → dissolve → 50% replay →
// dissolve → play + penalty → opponent goal (compact when focused) → period wipe + card over play →
// cut-goal score flap → FINAL.
export const T = {
  total: s(43),
  open: [0, 3.0],
  bug1: [2.9, 8.8], goal: [8.8, 13.9],
  replay: [14.3, 19.9],
  bug2: [20.4, 25.4], penalty: [21.2, 25.2], oppGoal: [25.4, 28.8],
  wipe: [29.0, 30.0], period: [30.0, 32.6],
  bug3: [32.6, 37.0], flap: [33.6, 37.0], flapScoreAt: 33.8,
  final: [37.2, 43.0],
};

const Seq = ({r, children}) => (
  <Sequence from={s(r[0])} durationInFrames={s(r[1] - r[0])} layout="none">{children}</Sequence>
);

export const Storyboard = ({V, game, perspective}) => {
  const focus = perspective === 'neutral' ? null : perspective;
  const g = {...game, focus};
  const d = (r) => s(r[1] - r[0]);
  const compactOpp = focus !== null && game.oppGoal.team !== focus;
  return (
    <AbsoluteFill style={{background: '#000', fontFamily: V.font}}>
      {/* footage */}
      <Sequence from={s(2.8)} durationInFrames={s(11.5)}><Clip src="clipB.mp4" /></Sequence>
      <Sequence from={s(14.0)} durationInFrames={s(6.3)}><Clip src="clipC.mp4" rate={0.5} fadeIn={18} /></Sequence>
      <Sequence from={s(20.0)} durationInFrames={s(9.5)}><Clip src="clipA.mp4" fadeIn={18} /></Sequence>
      <Sequence from={s(29.5)} durationInFrames={s(7.8)}><Clip src="clipC.mp4" /></Sequence>
      <Sequence from={s(37.0)} durationInFrames={s(6.0)}><Clip src="clipA.mp4" startFrom={s(2)} /></Sequence>

      {/* graphics */}
      {focus && <Sequence from={s(2.9)} durationInFrames={s(34.1)} layout="none"><V.Watermark g={g} dur={s(34.1)} /></Sequence>}
      <Seq r={T.bug1}><V.Bug g={g} dur={d(T.bug1)} period="1ST" score={() => [0, 0]} /></Seq>
      <Seq r={T.goal}><V.GoalCard g={g} goal={game.goal} score={[1, 0]} dur={d(T.goal)} compact={false} /></Seq>
      <Seq r={T.replay}><V.ReplayTag g={g} dur={d(T.replay)} /></Seq>
      <Seq r={T.bug2}><V.Bug g={g} dur={d(T.bug2)} period="1ST" score={() => [1, 0]} /></Seq>
      <Seq r={T.penalty}><V.PenaltyCard g={g} pen={game.penalty} dur={d(T.penalty)} /></Seq>
      <Seq r={T.oppGoal}><V.GoalCard g={g} goal={game.oppGoal} score={[1, 1]} dur={d(T.oppGoal)} compact={compactOpp} /></Seq>
      <Seq r={T.wipe}><V.PeriodWipe g={g} dur={d(T.wipe)} /></Seq>
      <Seq r={T.period}><V.PeriodCard g={g} period="2ND PERIOD" score={[1, 1]} dur={d(T.period)} /></Seq>
      <Seq r={T.bug3}>
        <V.Bug g={g} dur={d(T.bug3)} period="2ND"
          score={(f) => (f < s(T.flapScoreAt - T.bug3[0]) ? [1, 1] : [2, 1])}
          changeAt={s(T.flapScoreAt - T.bug3[0])} />
      </Seq>
      <Seq r={T.flap}><V.Flap g={g} flap={game.flap} dur={d(T.flap)} /></Seq>
      <Seq r={T.final}><V.FinalCard g={g} dur={d(T.final)} /></Seq>
      <Seq r={T.open}><V.Open g={g} dur={d(T.open)} /></Seq>
    </AbsoluteFill>
  );
};
