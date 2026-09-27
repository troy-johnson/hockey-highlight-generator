// PROTOTYPE — real-3D stingers (three.js via @remotion/three) for Variant D, in a "skating" language:
// a low tracking camera over a regulation rink, the slab hockey-stops with a shavings spray, frost grows
// over its face (old-stinger scrapes), bloom. Open carries the ICEPAK/HOCKEY lockup; the period
// transition wipes with snow walls and carries the crest. (Crystals/Streaks/Period3D are earlier passes.)
// Period: a wall of 3D blades that rotate closed over the cut (crest on the closed wall) and open again.
import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, useCurrentFrame, Easing, staticFile, delayRender, continueRender, OffthreadVideo, Sequence, useRemotionEnvironment} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import {EffectComposer, Bloom, Vignette} from '@react-three/postprocessing';
import * as THREE from 'three';
import {ramp} from './shared';
import {GAME_D} from './VariantD';
import {FrostLayer, Spray, SoftMist, SoftSnow} from './IceFX';
import {RinkFloor} from './Rink';
import {StreakSpray, SpeedStreaks2, PaintedSlab, outerShape, LensFrost} from './StreakFX';

const BLUE = '#5AA4D0', NAVY = '#0D1B2A', ICE = '#CFEFFF';
const rnd = (seed) => { const x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };
const W = 1920, H = 1080;

// Frame-driven camera (r3f has no makeDefault without drei).
const CameraRig = ({pos, fov}) => {
  const {camera} = useThree();
  camera.position.set(...pos); camera.fov = fov; camera.lookAt(0, 0, 0); camera.updateProjectionMatrix();
  return null;
};

// ThreeCanvas advances once per frame in an effect; the bloom composer sets its passes up afterwards,
// so re-advance after mount/frame change (pattern from the @remotion/three video-texture docs).
const ReAdvance = ({f, ready = true}) => {
  const {advance} = useThree();
  const {isRendering} = useRemotionEnvironment();
  useEffect(() => {
    const h = delayRender('re-advance');
    requestAnimationFrame(() => { advance(performance.now()); continueRender(h); });
  }, [f, ready, advance, isRendering]);
  return null;
};

const useTex = (src) => {
  const [tex, setTex] = useState(null);
  const [handle] = useState(() => delayRender(`texture ${src}`));
  useEffect(() => {
    new THREE.TextureLoader().load(staticFile(src), (t) => {
      t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; setTex(t); continueRender(handle);
    });
  }, [src, handle]);
  return tex;
};

// Faceted crystal: stretched, jittered octahedron — flat shaded with bright edges reads as ice.
const crystalGeo = (seed) => {
  const g = new THREE.OctahedronGeometry(1, 0);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    p.setXYZ(i, p.getX(i) * (0.7 + rnd(seed + i) * 0.6), p.getY(i) * (0.9 + rnd(seed + i * 3) * 0.9), p.getZ(i) * (0.7 + rnd(seed + i * 5) * 0.6));
  }
  g.computeVertexNormals();
  return g;
};

const Crystals = ({f, count = 34, impact, exitAt, dur}) => {
  const items = useMemo(() => Array.from({length: count}).map((_, i) => {
    const r = (k) => rnd(i * 11 + k);
    const geo = crystalGeo(i * 17);
    const angle = r(1) * Math.PI * 2, rad = 3.2 + r(2) * 6.5;
    return {
      geo, edges: new THREE.EdgesGeometry(geo),
      rest: [Math.cos(angle) * rad * 1.5, Math.sin(angle) * rad * 0.62, -3 + r(3) * 6],
      from: [Math.cos(angle) * 30, Math.sin(angle) * 14, -60 - r(4) * 40],
      scale: [0.22 + r(5) * 0.3, 0.35 + r(6) * 0.6, 0.22 + r(7) * 0.3],
      spin: [(r(8) - 0.5) * 0.04, (r(9) - 0.5) * 0.06, (r(10) - 0.5) * 0.03],
      rot0: [r(11) * 6, r(12) * 6, r(13) * 6], delay: Math.round(r(14) * 10),
    };
  }), [count]);
  return items.map((c, i) => {
    const k = ramp(f, 4 + c.delay, 34 + c.delay, 0, 1, Easing.out(Easing.cubic));
    const burst = ramp(f, impact, impact + 22, 0, 1, Easing.out(Easing.quad));
    const ex = ramp(f, exitAt + c.delay / 2, dur, 0, 1, Easing.in(Easing.cubic));
    const lerp = (a, b, t) => a + (b - a) * t;
    const pos = [0, 1, 2].map((j) => lerp(c.from[j], c.rest[j], k) + c.rest[j] * 0.12 * burst + (j === 2 ? ex * 22 : c.rest[j] * ex * 1.6));
    const rot = [0, 1, 2].map((j) => c.rot0[j] + c.spin[j] * f + (1 - k) * 4);
    const o = Math.min(1, k * 1.5);
    return (
      <group key={i} position={pos} rotation={rot} scale={c.scale}>
        <mesh geometry={c.geo}>
          <meshPhysicalMaterial color="#d8f3ff" emissive="#3aa0d8" emissiveIntensity={0.35} metalness={0} roughness={0.05}
            clearcoat={1} clearcoatRoughness={0.05} flatShading transparent opacity={0.3 * o} depthWrite={false} />
        </mesh>
        <lineSegments geometry={c.edges}>
          <lineBasicMaterial color="#e8fbff" transparent opacity={o} toneMapped={false} />
        </lineSegments>
      </group>
    );
  });
};

// Dash streaks at many depths (perspective gives natural parallax); emissive so bloom makes them glow.
const Streaks = ({f, count = 70, exitAt, dur}) => {
  const items = useMemo(() => Array.from({length: count}).map((_, i) => {
    const r = (k) => rnd(i * 13 + k + 500);
    return {y: (r(1) - 0.5) * 9, z: -14 + r(2) * 16, len: 0.2 + r(3) * (r(4) > 0.85 ? 3 : 0.9), speed: (0.05 + r(5) * 0.18) * (r(6) > 0.5 ? 1 : -1),
      x0: (r(7) - 0.5) * 30, color: r(8) > 0.6 ? '#ffffff' : BLUE, delay: Math.round(r(9) * 14)};
  }), [count]);
  return items.map((s, i) => {
    const inK = ramp(f, s.delay, s.delay + 20, 0, 1, Easing.out(Easing.cubic));
    const ex = ramp(f, exitAt, dur, 0, 1, Easing.in(Easing.cubic));
    const x = s.x0 * (0.3 + 0.7 * inK) + s.speed * f * 6 * (1 + ex * 8);
    const wrapped = ((x + 20) % 40 + 40) % 40 - 20;
    return (
      <mesh key={i} position={[wrapped, s.y, s.z]}>
        <boxGeometry args={[s.len, 0.035, 0.035]} />
        <meshBasicMaterial color={s.color} toneMapped={false} transparent opacity={inK * 0.9} />
      </mesh>
    );
  });
};

const slabShape = (w, h, skew) => {
  const s = new THREE.Shape();
  const o = Math.tan(skew) * h;
  s.moveTo(-w / 2 + o / 2, h / 2); s.lineTo(w / 2 + o / 2, h / 2); s.lineTo(w / 2 - o / 2, -h / 2); s.lineTo(-w / 2 - o / 2, -h / 2); s.closePath();
  return s;
};

// Parallelogram between centre-line x positions xa..xb, same slant as the slab.
const para = (xa, xb, h, skew) => {
  const s = new THREE.Shape(), o = Math.tan(skew) * h;
  s.moveTo(xa + o / 2, h / 2); s.lineTo(xb + o / 2, h / 2); s.lineTo(xb - o / 2, -h / 2); s.lineTo(xa - o / 2, -h / 2); s.closePath();
  return s;
};

// Slab body: navy/primary face with end caps built from the same slant, flush against the short sides, so
// the caps share the slab's lighting and read as one object. Thin lit rims replace the old 1-px edge lines.
const SlabBody = ({w, h, skew, depth, color, capColor, capW, glow = 0}) => {
  const o = Math.tan(skew) * h;
  const ext = (d, b) => ({depth: d, bevelEnabled: true, bevelThickness: b, bevelSize: b, bevelSegments: 4, curveSegments: 1});
  const face = useMemo(() => new THREE.ExtrudeGeometry(para(-w / 2, w / 2, h, skew), ext(depth, 0.05)), [w, h, skew, depth]);
  const caps = useMemo(() => [
    new THREE.ExtrudeGeometry(para(-w / 2 - capW, -w / 2 + 0.02, h, skew), ext(depth + 0.06, 0.045)),
    new THREE.ExtrudeGeometry(para(w / 2 - 0.02, w / 2 + capW, h, skew), ext(depth + 0.06, 0.045)),
  ], [w, h, skew, depth, capW]);
  return (
    <group position={[0, 0, -depth / 2 - 0.025]}>
      <mesh geometry={face}>
        <meshPhysicalMaterial color={color} metalness={0.45} roughness={0.34} clearcoat={0.6} clearcoatRoughness={0.25} />
      </mesh>
      {caps.map((g, i) => (
        <mesh key={i} geometry={g} position={[0, 0, -0.03]}>
          <meshPhysicalMaterial color={capColor} metalness={0.2} roughness={0.28} clearcoat={1} clearcoatRoughness={0.12}
            emissive={capColor} emissiveIntensity={0.18 + 0.9 * glow} />
        </mesh>
      ))}
      {[[h / 2, o / 2], [-h / 2, -o / 2]].map(([y, x], i) => (
        <mesh key={`r${i}`} position={[x, y, depth + 0.06]}>
          <boxGeometry args={[w, 0.028, 0.03]} />
          <meshStandardMaterial color="#e9f6ff" emissive="#cfeaff" emissiveIntensity={0.35 + 0.8 * glow} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );
};

const drift0 = (f, a, b) => ramp(f, a, b, 0, 1);
const OpenScene = ({f, dur, team}) => {
  const lockup = useTex(team.lockup);
  const IMPACT = 26, exitAt = dur - 26;
  const sw = 10, sh = 2.9, skew = (16 * Math.PI) / 180, capW = 0.34;
  const outer = useMemo(() => outerShape(sw, sh, skew, capW), []);
  // skating: tracking camera travels fast with the slab, brakes hard at the stop, speeds up on exit
  const travelAt = (t) => 70 * ramp(t, 0, IMPACT, 0, 1, Easing.out(Easing.quad)) + 3 * drift0(t, IMPACT, exitAt) + 60 * ramp(t, exitAt, dur, 0, 1, Easing.in(Easing.cubic));
  const travel = travelAt(f);
  // Hockey stop: slide in from the right, hard stop with a lean-back, spray off the leading bottom edge.
  const slide = ramp(f, 4, IMPACT, 0, 1, Easing.out(Easing.quad));
  const lean = ramp(f, IMPACT - 8, IMPACT, 0, 1) * (1 - ramp(f, IMPACT, IMPACT + 16, 0, 1, Easing.out(Easing.back(2.2))));
  const drift = ramp(f, IMPACT, exitAt, 0, 1);
  const out = ramp(f, exitAt, dur, 0, 1, Easing.in(Easing.exp));
  const slabPos = [24 * (1 - slide) + out * 28, 0.35, 0];
  const slabRot = [0, -0.35 * (1 - slide) + 0.05 - 0.1 * drift - out * 0.5, 0.1 * lean];
  const glow = ramp(f, IMPACT - 2, IMPACT + 2) * (1 - ramp(f, IMPACT + 4, IMPACT + 34));
  const logoK = ramp(f, IMPACT + 4, IMPACT + 24, 0, 1, Easing.out(Easing.cubic));
  const sweep = ramp(f, 56, 104, -7, 7, Easing.inOut(Easing.quad));
  const frost = ramp(f, IMPACT - 2, IMPACT + 46, 0, 0.36, Easing.out(Easing.cubic)) + ramp(f, IMPACT + 46, exitAt, 0, 0.06);
  const shake = f >= IMPACT ? Math.exp(-(f - IMPACT) / 5) * 0.12 : 0;
  const camPos = [Math.sin(f * 2.1) * shake, 1.0 + Math.cos(f * 2.7) * shake, 16 - 1.2 * drift];
  const bottom = -1.5;
  const emitters = useMemo(() => [
    // dense powder off the leading (left) bottom edge at the stop
    {t0: IMPACT - 6, t1: IMPACT + 6, origin: [-4.3, bottom, 0.4], spread: [3.2, 0.3, 0.9], v: [-8.5, 6.2, 3], vJit: [7, 4.5, 5], size: [0.018, 0.06], life: [0.8, 1.8], count: 2600, alpha: 0.5},
    // fine fast grains, wider fan
    {t0: IMPACT - 4, t1: IMPACT + 4, origin: [-4, bottom + 0.05, 0.3], spread: [3, 0.2, 0.6], v: [-13, 7.5, 4], vJit: [9, 6, 7], size: [0.01, 0.026], life: [0.5, 1.2], count: 1800, alpha: 0.8},
    // brighter granules
    {t0: IMPACT - 2, t1: IMPACT + 3, origin: [-4.2, bottom, 0.5], spread: [2, 0.15, 0.4], v: [-6.5, 7, 3.5], vJit: [5, 4, 4], size: [0.05, 0.1], life: [0.9, 1.6], count: 120, alpha: 0.9},
    // soft volume that hangs in the air
    {t0: IMPACT - 4, t1: IMPACT + 8, origin: [-3.6, bottom + 0.2, 0.5], spread: [4, 0.4, 0.8], v: [-4.5, 3, 2], vJit: [4, 2, 2], size: [0.35, 0.9], life: [1.2, 2.4], count: 300, alpha: 0.08},
  ], []);
  const puffs = useMemo(() => Array.from({length: 14}).map((_, i) => ({
    t0: IMPACT - 4 + rnd(i + 40) * 12, life: 1.4 + rnd(i + 41) * 1.2, x: -6 + rnd(i + 42) * 5.5, y: bottom + rnd(i + 43) * 0.9, z: 0.8 + rnd(i + 44) * 1.4,
    vx: -1.2 - rnd(i + 45) * 2.6, vy: 0.3 + rnd(i + 46) * 0.7, size: 1.6 + rnd(i + 47) * 2.4, opacity: 0.14 + rnd(i + 48) * 0.16, rot: rnd(i + 49) * 6,
  })), []);
  return (
    <>
      <color attach="background" args={['#04070d']} />
      <fog attach="fog" args={['#04070d', 12, 48]} />
      <CameraRig pos={camPos} fov={32} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[-6, 8, 10]} intensity={1.1} />
      <spotLight position={[0, 14, -4]} angle={0.5} penumbra={0.8} intensity={260} distance={40} color="#dff1ff" />
      <pointLight position={[sweep, 2.0, 2.4]} intensity={24} distance={7} color="#ffffff" />
      <pointLight position={[0, -3, -6]} intensity={80} distance={20} color={BLUE} />
      <pointLight position={[-5, 3, 5]} intensity={18} distance={10} color="#cfefff" />
      <SoftSnow f={f} />
      <RinkFloor travel={travel} floorY={-1.55} />
      <SpeedStreaks2 travel={travel} travelPrev={travelAt(f - 1.5)} floorY={-1.55} />
      <group position={slabPos} rotation={slabRot}>
        <PaintedSlab w={sw} h={sh} skew={skew} depth={0.45} color={team.primary} capColor={BLUE} capW={capW} glow={glow} />
        <FrostLayer shape={outer} progress={frost} z={0.275} front={1 - 0.85 * ramp(f, IMPACT + 30, IMPACT + 60)} />
        {lockup && (
          <mesh position={[0, 0, 0.31]} scale={[0.85 + 0.15 * logoK, 0.85 + 0.15 * logoK, 1]}>
            <planeGeometry args={[7.2, 7.2 * (468 / 1600)]} />
            <meshBasicMaterial map={lockup} transparent opacity={logoK} toneMapped={false} depthWrite={false} />
          </mesh>
        )}
      </group>
      <StreakSpray f={f} emitters={emitters} />
      <SoftMist f={f} puffs={puffs} />
      <EffectComposer multisampling={0} frameBufferType={THREE.UnsignedByteType}>
        <Bloom intensity={1.0} luminanceThreshold={0.6} luminanceSmoothing={0.25} mipmapBlur />
        <Vignette offset={0.3} darkness={0.7} />
      </EffectComposer>
      <ReAdvance f={f} ready={!!lockup} />
    </>
  );
};

export const Open3D = ({focus = 'home', dur = 180}) => {
  const f = useCurrentFrame();
  const g = GAME_D;
  const team = {...g[focus], lockup: 'icepak_lockup.png'};
  const opp = g[focus === 'home' ? 'away' : 'home'];
  const plate = [0, 1, 2].map((i) => ramp(f, 60 + i * 5, 74 + i * 5, 0, 1, Easing.out(Easing.back(1.4))));
  const out = ramp(f, dur - 26, dur, 0, 1, Easing.in(Easing.exp));
  const flash = f < 26 ? 0 : 0.12 * (1 - ramp(f, 26, 32));
  return (
    <AbsoluteFill style={{background: '#04070d', fontFamily: 'Inter'}}>
      <ThreeCanvas width={W} height={H} dpr={2} gl={{preserveDrawingBuffer: true, antialias: true}}><OpenScene f={f} dur={dur} team={team} /></ThreeCanvas>
      <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 45%, #fff 0%, ${BLUE} 55%, transparent 100%)`, opacity: flash, mixBlendMode: 'screen'}} />
      <div style={{position: 'absolute', left: 0, right: 0, top: 760, display: 'flex', justifyContent: 'center', gap: 8, perspective: 900,
        transform: `translateX(${-out * 2400}px)`}}>
        {[
          <div key="a" style={{background: NAVY, color: '#fff', height: 72, padding: '0 28px', display: 'flex', alignItems: 'center', fontWeight: 900, fontStyle: 'italic', fontSize: 30, transform: 'skewX(-16deg)', outline: '3px solid #fff'}}>VS</div>,
          <div key="b" style={{display: 'flex', transform: 'skewX(-16deg)', outline: '3px solid #fff'}}>
            <div style={{background: opp.primary, height: 72, padding: '0 20px', display: 'flex', alignItems: 'center'}}><img src={staticFile(opp.square)} style={{height: 56, transform: 'skewX(16deg)'}} /></div>
            <div style={{background: '#0A0A0A', color: '#fff', height: 72, padding: '0 30px', display: 'flex', alignItems: 'center', fontWeight: 800, fontSize: 34}}><span style={{transform: 'skewX(16deg)'}}>{opp.name}</span></div>
          </div>,
          <div key="c" style={{background: NAVY, color: '#D9D9D9', height: 72, padding: '0 28px', display: 'flex', alignItems: 'center', fontWeight: 700, fontSize: 26, transform: 'skewX(-16deg)', outline: '3px solid #fff'}}><span style={{transform: 'skewX(16deg)'}}>{g.date}</span></div>,
        ].map((el, i) => <div key={i} style={{transform: `rotateX(${(1 - plate[i]) * 90}deg)`, transformOrigin: 'top', opacity: Math.min(1, plate[i] * 1.5)}}>{el}</div>)}
      </div>
    </AbsoluteFill>
  );
};

// ---------- Period: 3D blinds over footage ----------
const BladesScene = ({f, dur, team}) => {
  const crest = useTex(team.square);
  const N = 12, bw = 19.5 / N, bh = 13;
  const close = (i) => ramp(f, i * 1.2, i * 1.2 + 16, 0, 1, Easing.out(Easing.cubic));
  const open = (i) => ramp(f, 34 + i * 1.2, 34 + i * 1.2 + 16, 0, 1, Easing.in(Easing.cubic));
  const allClosed = ramp(f, 18, 24) * (1 - ramp(f, 34, 38));
  const crestK = ramp(f, 16, 26, 0, 1, Easing.out(Easing.back(1.6))) * (1 - ramp(f, 34, 40));
  return (
    <>
      <CameraRig pos={[0, 0, 20]} fov={30} />
      <ambientLight intensity={0.35} />
      <directionalLight position={[-8, 6, 10]} intensity={1.6} />
      <pointLight position={[ramp(f, 14, 40, -10, 10), 2, 4]} intensity={90} distance={14} color="#ffffff" />
      {Array.from({length: N}).map((_, i) => {
        const x = -9.75 + bw * (i + 0.5);
        const ry = (1 - close(i)) * (Math.PI / 2) - open(i) * (Math.PI / 2);
        const visible = close(i) > 0 && open(i) < 1;
        if (!visible) return null;
        return (
          <group key={i} position={[x, 0, 0]} rotation={[0, ry, 0]}>
            <mesh><boxGeometry args={[bw * 1.02, bh, 0.12]} /><meshStandardMaterial color={i % 3 === 1 ? team.primary : NAVY} metalness={0.5} roughness={0.35} /></mesh>
            <mesh position={[bw / 2, 0, 0.07]}><boxGeometry args={[0.05, bh, 0.02]} /><meshBasicMaterial color={BLUE} toneMapped={false} /></mesh>
          </group>
        );
      })}
      <ReAdvance f={f} ready={!!crest} />
      {crest && crestK > 0 && (
        <mesh position={[0, 0, 0.4]} scale={[0.8 + 0.2 * crestK, 0.8 + 0.2 * crestK, 1]}>
          <planeGeometry args={[4.6, 4.6 * (1.05)]} />
          <meshBasicMaterial map={crest} transparent opacity={crestK * allClosed + crestK * 0.0001} toneMapped={false} />
        </mesh>
      )}
    </>
  );
};

export const Period3D = ({focus = 'home', dur = 60}) => {
  const f = useCurrentFrame();
  const team = GAME_D[focus];
  return (
    <AbsoluteFill>
      <ThreeCanvas width={W} height={H} gl={{alpha: true}} style={{background: 'transparent'}}><BladesScene f={f} dur={dur} team={team} /></ThreeCanvas>
    </AbsoluteFill>
  );
};

// Standalone review compositions: the open, and the period transition over a real cut (cut at frame 60).
export const Open3DReview = () => <Open3D />;
const reviewCut = (Wipe) => () => (
  <AbsoluteFill style={{background: '#000'}}>
    <Sequence durationInFrames={60}><OffthreadVideo src={staticFile('clipA.mp4')} startFrom={120} muted /></Sequence>
    <Sequence from={60}><OffthreadVideo src={staticFile('clipC.mp4')} muted /></Sequence>
    <Sequence from={30} durationInFrames={60}><Wipe /></Sequence>
  </AbsoluteFill>
);
export const Period3DReview = reviewCut((p) => <PeriodLens3D {...p} />);
export const PeriodSkate3DReview = reviewCut((p) => <PeriodSkate3D {...p} />);

// ---------- Period transition in the skating language ----------
// A snow wall sweeps in from the right over the old footage revealing the 3D rink; the crest slab
// hockey-stops with a spray and frosts over the cut; a second snow edge keeps travelling left and
// reveals the new period.
const PeriodSkateScene = ({f, dur, team}) => {
  const crest = useTex(team.square);
  const IMPACT = 24, exitAt = 42;
  const sw = 6.2, sh = 3.4, skew = (16 * Math.PI) / 180;
  const shape = useMemo(() => slabShape(sw, sh, skew), []);
  const slide = ramp(f, 6, IMPACT, 0, 1, Easing.out(Easing.quad));
  const lean = ramp(f, IMPACT - 6, IMPACT, 0, 1) * (1 - ramp(f, IMPACT, IMPACT + 12, 0, 1, Easing.out(Easing.back(2.2))));
  const out = ramp(f, exitAt, dur, 0, 1, Easing.in(Easing.cubic));
  const travel = 50 * ramp(f, 0, IMPACT, 0, 1, Easing.out(Easing.quad)) + 40 * out;
  const speed = f < IMPACT ? 1.2 * (1 - Math.pow(f / IMPACT, 3)) + 0.05 : 0.05 + 1.4 * out;
  const shake = f >= IMPACT ? Math.exp(-(f - IMPACT) / 4) * 0.14 : 0;
  const frost = ramp(f, IMPACT - 2, IMPACT + 18, 0, 0.34, Easing.out(Easing.cubic));
  const crestK = ramp(f, IMPACT - 2, IMPACT + 8, 0, 1, Easing.out(Easing.back(1.6)));
  const bottom = -1.5;
  const emitters = useMemo(() => [
    {t0: IMPACT - 5, t1: IMPACT + 4, origin: [-3.1, bottom, 0.4], spread: [2.2, 0.3, 0.9], v: [-8.5, 6.2, 3], vJit: [7, 4.5, 5], size: [0.02, 0.07], life: [0.7, 1.4], count: 1400, alpha: 0.45},
    {t0: IMPACT - 3, t1: IMPACT + 3, origin: [-2.8, bottom + 0.05, 0.3], spread: [2, 0.2, 0.6], v: [-13, 7.5, 4], vJit: [9, 6, 7], size: [0.01, 0.03], life: [0.5, 1.0], count: 900, alpha: 0.8},
    {t0: IMPACT - 2, t1: IMPACT + 2, origin: [-2.8, bottom, 0.5], spread: [1.6, 0.15, 0.4], v: [-6.5, 7, 3.5], vJit: [5, 4, 4], size: [0.07, 0.13], life: [0.8, 1.3], count: 60},
    {t0: IMPACT - 3, t1: IMPACT + 6, origin: [-2.6, bottom + 0.2, 0.5], spread: [3, 0.4, 0.8], v: [-4.5, 3, 2], vJit: [4, 2, 2], size: [0.35, 0.9], life: [1, 1.8], count: 180, alpha: 0.1},
  ], []);
  const puffs = useMemo(() => Array.from({length: 9}).map((_, i) => ({
    t0: IMPACT - 3 + rnd(i + 80) * 8, life: 1.1 + rnd(i + 81) * 0.8, x: -4 + rnd(i + 82) * 4, y: bottom + rnd(i + 83) * 0.9, z: 0.8 + rnd(i + 84) * 1.2,
    vx: -1.2 - rnd(i + 85) * 2.6, vy: 0.3 + rnd(i + 86) * 0.7, size: 1.4 + rnd(i + 87) * 2, opacity: 0.18 + rnd(i + 88) * 0.18, rot: rnd(i + 89) * 6,
  })), []);
  return (
    <>
      <color attach="background" args={['#04070d']} />
      <fog attach="fog" args={['#04070d', 12, 48]} />
      <CameraRig pos={[Math.sin(f * 2.1) * shake, 1.0 + Math.cos(f * 2.7) * shake, 16]} fov={32} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[-6, 8, 10]} intensity={1.1} />
      <spotLight position={[0, 14, -4]} angle={0.5} penumbra={0.8} intensity={260} distance={40} color="#dff1ff" />
      <pointLight position={[ramp(f, IMPACT, dur, -4, 4), 2.2, 2.6]} intensity={16} distance={7} color="#ffffff" />
      <SoftSnow f={f} count={180} />
      <RinkFloor travel={travel} floorY={-1.55} camFt0={150} />
      <SpeedStreaks2 travel={travel} travelPrev={travel - 1.5 * speed} floorY={-1.55} />
      <group position={[18 * (1 - slide) - 22 * out, 0.35, 0]} rotation={[0, -0.3 * (1 - slide) + 0.04, 0.1 * lean]}>
        <SlabBody w={sw} h={sh} skew={skew} depth={0.4} color={team.primary} capColor={team.secondary} capW={0.28} glow={ramp(f, IMPACT - 2, IMPACT + 2) * (1 - ramp(f, IMPACT + 2, IMPACT + 24))} />
        <FrostLayer shape={shape} progress={frost} z={0.235} front={1 - ramp(f, IMPACT + 10, IMPACT + 22)} />
        {crest && <mesh position={[0, 0, 0.27]} scale={[0.8 + 0.2 * crestK, 0.8 + 0.2 * crestK, 1]}>
          <planeGeometry args={[2.9, 2.9 * (5256 / 5000)]} />
          <meshBasicMaterial map={crest} transparent opacity={crestK} toneMapped={false} depthWrite={false} />
        </mesh>}
      </group>
      <Spray f={f} emitters={emitters} />
      <SoftMist f={f} puffs={puffs} />
      <EffectComposer multisampling={0} frameBufferType={THREE.UnsignedByteType}>
        <Bloom intensity={1.0} luminanceThreshold={0.55} luminanceSmoothing={0.2} mipmapBlur />
        <Vignette offset={0.3} darkness={0.7} />
      </EffectComposer>
      <ReAdvance f={f} ready={!!crest} />
    </>
  );
};

// Ragged snow edge that rides the mask boundary: fog band + flying flakes.
const SnowEdge = ({x, f, dir = -1, strength = 1}) => {
  if (strength <= 0) return null;
  return (
    <div style={{position: 'absolute', top: -40, bottom: -40, left: x - 260, width: 520, pointerEvents: 'none', opacity: strength}}>
      <div style={{position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 50% 60% at 50% 50%, rgba(240,248,255,.95) 0%, rgba(210,232,248,.7) 35%, rgba(180,215,240,0) 70%)', filter: 'blur(6px)'}} />
      {Array.from({length: 90}).map((_, i) => {
        const yy = rnd(i + 300) * 1160, off = (rnd(i + 301) - 0.5) * 460, sz = 2 + rnd(i + 302) * 7, trail = 10 + rnd(i + 303) * 60;
        const jitter = Math.sin(f * 0.9 + i) * 6;
        return <div key={i} style={{position: 'absolute', top: yy + jitter, left: 260 + off, width: sz + trail, height: sz, borderRadius: sz,
          background: `linear-gradient(${dir < 0 ? 90 : 270}deg, #ffffff, rgba(255,255,255,0))`, opacity: 0.5 + rnd(i + 304) * 0.5}} />;
      })}
    </div>
  );
};

export const PeriodSkate3D = ({focus = 'home', dur = 60}) => {
  const f = useCurrentFrame();
  const team = GAME_D[focus];
  const Wd = 1920;
  // left edge of the covered band slides right→left on the way in; right edge slides right→left on the way out
  const L = Wd * (1 - ramp(f, 0, 14, 0, 1.25, Easing.out(Easing.cubic)));
  const R = Wd * (1.15 - ramp(f, 42, dur, 0, 1.4, Easing.in(Easing.cubic)));
  const soft = 120;
  const mask = `linear-gradient(to right, transparent ${L - soft}px, black ${L}px, black ${R}px, transparent ${R + soft}px)`;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{WebkitMaskImage: mask, maskImage: mask}}>
        <ThreeCanvas width={W} height={H} dpr={2} gl={{preserveDrawingBuffer: true, antialias: true}}><PeriodSkateScene f={f} dur={dur} team={team} /></ThreeCanvas>
      </AbsoluteFill>
      <SnowEdge x={L} f={f} strength={1 - ramp(f, 12, 18)} />
      <SnowEdge x={R} f={f} strength={ramp(f, 40, 44) * (1 - ramp(f, 56, 60))} />
    </AbsoluteFill>
  );
};

// ---------- Period transition, round 3: snow on the lens ----------
// Different from the open: no slab and no rink. The game footage stays the background. A skater carves
// across low in frame and throws a hockey-stop spray at the camera; snow hits the glass and frost
// crystallises over the lens in about 14 frames, hiding the cut; the crest punches in on the frosted
// glass; the frost melts open from the centre onto the new period.
const LENS_SEEDS = [[0.06, 0.88], [0.3, 0.97], [0.62, 0.93], [0.94, 0.86], [0.97, 0.3], [0.04, 0.2], [0.45, 0.62], [0.72, 0.5]];
const PeriodLensScene = ({f}) => {
  const emitters = useMemo(() => [
    // carve sweeps right → left along the ice, throwing a spray wall at the lens
    {t0: 0, t1: 20, origin: [5.5, -2.4, -1], spread: [0.8, 0.3, 1.2], move: [-33, 0, 0], v: [3, 15, 22], vJit: [10, 9, 10], size: [0.01, 0.035], life: [0.4, 0.8], count: 11000, alpha: 0.9},
    {t0: 0, t1: 20, origin: [5.5, -2.4, -1], spread: [0.8, 0.3, 1.2], move: [-33, 0, 0], v: [2, 13, 20], vJit: [9, 8, 9], size: [0.035, 0.08], life: [0.4, 0.8], count: 2200, alpha: 0.7},
    {t0: 2, t1: 18, origin: [5, -2.4, -1], spread: [0.6, 0.2, 0.8], move: [-33, 0, 0], v: [2, 14, 20], vJit: [8, 6, 8], size: [0.08, 0.14], life: [0.4, 0.7], count: 140, alpha: 0.9},
    // soft snow cloud that billows up behind the carve
    {t0: 0, t1: 24, origin: [5, -2.0, -1.5], spread: [1.5, 0.6, 1.5], move: [-30, 0, 0], v: [2, 7, 10], vJit: [4, 4, 6], size: [0.5, 1.3], life: [0.6, 1.2], count: 480, alpha: 0.12},
  ], []);
  const progress = ramp(f, 11, 25, 0, 1.08, Easing.out(Easing.cubic));
  const clear = ramp(f, 37, 55, 0, 1.12, Easing.inOut(Easing.quad));
  return (
    <>
      <CameraRig pos={[0, 0, 8]} fov={40} />
      <StreakSpray f={f} emitters={emitters} drag={1.6} gravity={6} floor={false} maxR={46} hotTime={0.12} color={[0.55, 0.66, 0.78]} hot={[0.93, 0.97, 1.0]} />
      <LensFrost progress={progress} clear={clear} opacity={1 - ramp(f, 54, 60)} seeds={LENS_SEEDS} />
    </>
  );
};

export const PeriodLens3D = ({focus = 'home', dur = 60}) => {
  const f = useCurrentFrame();
  const team = GAME_D[focus];
  const crestIn = ramp(f, 19, 28, 0, 1, Easing.out(Easing.back(1.8)));
  const crestOut = ramp(f, 36, 44, 0, 1, Easing.in(Easing.cubic));
  const flash = ramp(f, 22, 25) * (1 - ramp(f, 25, 33));
  // the blur clears with the frost: a hole that opens from the centre
  const holeR = Math.max(1, ramp(f, 37, 55, 0, 1350, Easing.inOut(Easing.quad)));
  const hole = `radial-gradient(ellipse ${holeR * 1.1}px ${holeR * 0.75}px at 50% 50%, transparent 55%, black 100%)`;
  const cover = ramp(f, 11, 25, 0, 1, Easing.out(Easing.cubic)) * (1 - ramp(f, 42, 53, 0, 1, Easing.inOut(Easing.quad)));
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{backdropFilter: `blur(${(cover * 26).toFixed(2)}px) saturate(${1 - 0.4 * cover})`, WebkitBackdropFilter: `blur(${(cover * 26).toFixed(2)}px)`,
        WebkitMaskImage: hole, maskImage: hole}} />
      <ThreeCanvas width={W} height={H} dpr={2} gl={{alpha: true, preserveDrawingBuffer: true, antialias: true, premultipliedAlpha: true}} style={{position: 'absolute', inset: 0}}>
        <PeriodLensScene f={f} dur={dur} />
      </ThreeCanvas>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse at 50% 50%, rgba(255,255,255,.9), rgba(200,230,250,0) 70%)', opacity: flash * 0.5, mixBlendMode: 'screen'}} />
      <AbsoluteFill style={{alignItems: 'center', justifyContent: 'center'}}>
        <img src={staticFile(team.square)} style={{height: 420, opacity: crestIn * (1 - crestOut),
          transform: `scale(${0.7 + 0.3 * crestIn + 0.25 * crestOut})`, filter: `drop-shadow(0 0 3px rgba(13,27,42,.9)) drop-shadow(0 0 26px rgba(13,27,42,.55)) drop-shadow(0 10px 22px rgba(13,27,42,.45))`}} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
