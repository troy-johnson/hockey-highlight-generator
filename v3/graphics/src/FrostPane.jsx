import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, cancelRender, continueRender, delayRender, Easing, Img, staticFile, useCurrentFrame} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import * as THREE from 'three';

// Crest-led frost transition. One full-screen shader draws the pane, its
// growing frost, the bright leading edge, spray, and the ice crest, so every
// layer shares one deterministic, frame-driven material.
const vertexShader = `varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const fragmentShader = `precision highp float;
uniform vec2 res; uniform float paneX; uniform float frame; uniform float dur; uniform float sweep;
uniform sampler2D crest; uniform float hasCrest; uniform vec2 crestC; uniform vec2 crestS; uniform float crestA;
uniform vec3 teamLine; uniform vec3 teamGlow; uniform float hasTeam; uniform float replay; uniform float hold; uniform float vis;
varying vec2 vUv;
float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h21(i), h21(i + vec2(1, 0)), f.x), mix(h21(i + vec2(0, 1)), h21(i + 1.0), f.x), f.y); }
float fbm(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return v; }
vec2 vor(vec2 p) { vec2 i = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) { vec2 g = vec2(float(x), float(y));
    vec2 o = vec2(h21(i + g), h21(i + g + 31.7)); float d = length(g + o - f);
    if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d; }
  return vec2(d1, d2); }
float edgeFade(vec2 uv) { vec2 e = min(uv, 1.0 - uv); return smoothstep(0.0, 0.06, min(e.x, e.y)); }
float inBox(vec2 uv, float m) { return step(-m, uv.x) * step(-m, uv.y) * step(uv.x, 1.0 + m) * step(uv.y, 1.0 + m); }
float crestAlpha(vec2 uv, float bias) { return inBox(uv, 0.0) * texture2D(crest, uv, bias).a; }
void main() {
  vec2 p = vec2(vUv.x * res.x, (1.0 - vUv.y) * res.y);
  float W = res.x, H = res.y;
  vec2 l = vec2(p.x - paneX, p.y);
  float rag = fbm(vec2(p.y * 0.018, 3.0)) * 34.0 + fbm(vec2(p.y * 0.12, 9.0)) * 9.0;
  float d = min((W - rag) - l.x, l.x - rag * 0.8);
  float inside = max(smoothstep(0.0, 1.5, d), hold);
  float t = clamp(frame / dur, 0.0, 1.0);

  float g = clamp(l.x / W * 0.7 + l.y / H * 0.3, 0.0, 1.0);
  vec3 col = mix(vec3(0.098, 0.129, 0.165), vec3(0.055, 0.075, 0.098), smoothstep(0.0, 0.45, g));
  col = mix(col, vec3(0.027, 0.043, 0.063), smoothstep(0.45, 1.0, g));
  vec2 facet = vor(l / 170.0);
  col += vec3(0.03, 0.04, 0.05) * (h21(floor(l / 170.0 + facet)) - 0.5);
  col += vec3(0.55, 0.68, 0.78) * (1.0 - smoothstep(0.0, 0.035, facet.y - facet.x)) * 0.07;

  float reach = mix(70.0, W * 0.5, smoothstep(0.0, 0.75, t));
  float field = fbm(l * 0.011) * 0.65 + fbm(l * 0.047) * 0.35;
  float frostMask = 1.0 - smoothstep(reach * 0.35, reach, d - field * 140.0);
  float fine = fbm(l * 0.17);
  vec2 cell = vor(l / 34.0);
  float crystal = 1.0 - smoothstep(0.0, 0.07, cell.y - cell.x);
  float needles = smoothstep(0.62, 0.9, vnoise(vec2((l.x + l.y) * 0.09, (l.x - l.y) * 0.006)));
  float frost = frostMask * (0.18 + 0.42 * fine * crystal + 0.25 * needles * fine + 0.12 * fine);
  col = mix(col, vec3(0.78, 0.88, 0.95), clamp(frost * 0.62, 0.0, 0.7));
  float growFront = exp(-pow((d - field * 140.0 - reach * 0.82) / (reach * 0.08 + 6.0), 2.0)) * frostMask;
  col += vec3(0.7, 0.86, 1.0) * growFront * 0.16;

  float front = exp(-max(d, 0.0) / 13.0);
  col += vec3(0.82, 0.93, 1.0) * front * 0.8 + vec3(0.45, 0.65, 0.8) * exp(-max(d, 0.0) / 95.0) * 0.2 * fine;
  col = mix(col, teamLine, exp(-pow((d - 10.0) / 2.4, 2.0)) * hasTeam * 0.95);
  col += vec3(0.8, 0.93, 1.0) * exp(-pow((d - 2.5) / 1.1, 2.0));
  float streak = smoothstep(0.8, 0.96, vnoise(vec2(l.x * 0.0035, l.y * 0.32))) * exp(-max(d, 0.0) / 70.0);
  vec2 gc = floor(l / 21.0);
  float glint = step(0.986, h21(gc + floor(frame * 0.5) * 7.13)) * frostMask *
    (1.0 - smoothstep(0.0, 3.5, length(fract(l / 21.0) - 0.5) * 21.0));
  col += vec3(1.0) * (streak * 0.3 + glint * 0.85);
  float band = (p.x + p.y * 0.45) / (W + H * 0.45);
  col += vec3(0.86, 0.95, 1.0) * exp(-pow((band - mix(-0.1, 1.1, t)) / 0.05, 2.0)) * 0.1;

  float spray = smoothstep(0.82, 0.97, vnoise(vec2(p.y * 0.42, l.x * 0.012 + frame * 0.7))) *
    exp(min(d, 0.0) / 34.0) * (1.0 - inside) * vis;
  col = mix(vec3(0.85, 0.94, 1.0), col, inside);
  float A = max(inside, spray * 0.65);

  vec2 cu = (p - crestC) / crestS + 0.5;
  float ca = crestAlpha(cu, 0.0) * hasCrest;
  float shadow = edgeFade(cu - vec2(0.012, 0.022)) * texture2D(crest, cu - vec2(0.012, 0.022), 4.5).a * hasCrest;
  float glow = edgeFade(cu) * texture2D(crest, cu, 6.5).a * hasCrest;
  col *= 1.0 - shadow * 0.65 * crestA;
  col += teamGlow * glow * 0.45 * crestA * (1.0 - ca);
  A = max(A, max(shadow * 0.55, glow * 0.4) * crestA);
  vec3 cc = texture2D(crest, cu).rgb;
  vec2 lightStep = vec2(-2.5, -3.5) / crestS;
  float rim = clamp(ca - crestAlpha(cu - lightStep, 0.0), 0.0, 1.0);
  float shade = clamp(crestAlpha(cu + lightStep, 0.0) - ca, -1.0, 0.0);
  vec2 cl = cu * crestS;
  vec3 ice = mix(cc, vec3(0.85, 0.93, 1.0), fbm(cl * 0.05) * 0.14 + (1.0 - smoothstep(0.0, 0.06, vor(cl / 26.0).y - vor(cl / 26.0).x)) * 0.08);
  float crestBand = exp(-pow(((cu.x + cu.y * 0.55) / 1.55 - mix(-0.15, 1.15, smoothstep(0.15, 0.85, t))) / 0.07, 2.0));
  ice += vec3(0.92, 0.98, 1.0) * (rim * 0.9 + crestBand * 0.38 * ca) + teamGlow * 0.12 * ca;
  ice *= 1.0 + shade * 0.35;
  col = mix(col, ice, ca * crestA);
  A = mix(A, 1.0, ca * crestA);
  gl_FragColor = vec4(clamp(col, 0.0, 1.0) * A, A);
}`;

const rgb = hex => {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return new THREE.Vector3((n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255);
};

const PaneQuad = ({values}) => {
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader, transparent: true, blending: THREE.NoBlending,
    depthTest: false, depthWrite: false, toneMapped: false,
    uniforms: Object.fromEntries(['res', 'paneX', 'frame', 'dur', 'sweep', 'crest', 'hasCrest',
      'crestC', 'crestS', 'crestA', 'teamLine', 'teamGlow', 'hasTeam', 'replay', 'hold', 'vis']
      .map(key => [key, {value: null}])),
  }), []);
  const geometry = useMemo(() => new THREE.PlaneGeometry(2, 2), []);
  useEffect(() => () => {material.dispose(); geometry.dispose();}, [material, geometry]);
  for (const [key, value] of Object.entries(values)) material.uniforms[key].value = value;
  return <mesh geometry={geometry} material={material} frustumCulled={false}/>;
};

const blank = new THREE.DataTexture(new Uint8Array(4), 1, 1);
blank.needsUpdate = true;

export const FrostPane = ({vertical = false, replay = false, team}) => {
  const f = useCurrentFrame();
  const width = vertical ? 1080 : 1920;
  const height = vertical ? 1920 : 1080;
  const first = replay ? 7 : 12;
  const last = replay ? 9 : 17;
  const end = replay ? 15 : 29;
  const ease = Easing.inOut(Easing.cubic);
  const x = f < first ? -width * (1 - ease(f / first)) :
    f <= last ? 0 : width * ease((f - last) / (end - last));
  const asset = team ? (team.crest || team.logo) : null;
  const [crest, setCrest] = useState(null);
  const [handle] = useState(() => delayRender('Load transition crest'));
  useEffect(() => {
    if (!asset) {continueRender(handle); return undefined;}
    let live = true;
    const image = new Image();
    image.src = staticFile(asset);
    image.decode().then(() => {
      if (!live) return;
      const texture = new THREE.Texture(image);
      texture.flipY = false;
      texture.colorSpace = THREE.NoColorSpace;
      texture.anisotropy = 8;
      texture.needsUpdate = true;
      const ratio = image.naturalWidth / image.naturalHeight;
      setCrest({texture, aspect: Number.isFinite(ratio) && ratio > 0 ? ratio : 1});
    }).catch(error => cancelRender(error));
    return () => {live = false;};
  }, [asset, handle]);
  useEffect(() => {
    if (crest) continueRender(handle);
    return () => crest?.texture.dispose();
  }, [crest, handle]);
  const k = end / 15;
  const clamp = v => Math.max(0, Math.min(1, v));
  const crestIn = Easing.out(Easing.cubic)(clamp((f - k) / (4 * k)));
  const crestOut = Easing.in(Easing.cubic)(clamp((f - 11 * k) / (4 * k)));
  const crestScale = 1.12 - .12 * crestIn - .04 * crestOut;
  const aspect = crest?.aspect || 1;
  const crestH = Math.min(height * .6, width * .8 / aspect) * crestScale;
  const line = rgb(team?.line ?? team?.primary) || new THREE.Vector3(.5, .83, .94);
  // The crest glow follows the kit line, like the edge line (spec 003).
  const glow = line;
  const values = {
    res: new THREE.Vector2(width, height), paneX: x, frame: f, dur: end, sweep: f / end,
    crest: crest?.texture || blank, hasCrest: crest ? 1 : 0,
    crestC: new THREE.Vector2(width / 2 + x * .18, height / 2),
    crestS: new THREE.Vector2(crestH * aspect, crestH), crestA: crestIn * (1 - crestOut),
    teamLine: line, teamGlow: glow, hasTeam: team ? 1 : 0, replay: replay ? 1 : 0,
    hold: f >= first && f <= last ? 1 : 0, vis: Math.abs(x) < width - 1 ? 1 : 0,
  };
  return <AbsoluteFill style={{overflow: 'hidden', pointerEvents: 'none', zIndex: 35}}>
    <ThreeCanvas width={width} height={height} dpr={2}
      gl={{alpha: true, antialias: false, premultipliedAlpha: true}}
      style={{position: 'absolute', inset: 0}}>
      <PaneQuad values={values}/>
    </ThreeCanvas>
  </AbsoluteFill>;
};

export const PeriodRail = ({data, teams, duration, vertical}) => {
  const f = useCurrentFrame();
  const enter = Easing.out(Easing.cubic)(Math.max(0, Math.min(1, f / 12)));
  const exit = Easing.in(Easing.cubic)(Math.max(0, Math.min(1, (f - duration + 11) / 10)));
  const title = String(data.period).toUpperCase() === 'SO' ? 'SHOOTOUT' :
    ({1: '1st PERIOD', 2: '2nd PERIOD', 3: '3rd PERIOD'}[data.period] || 'OVERTIME');
  return <div style={{position: 'absolute', zIndex: 20, left: 96, bottom: vertical ? 430 : 74,
    width: vertical ? 840 : 1010, boxSizing: 'border-box', padding: '20px 24px',
    background: 'linear-gradient(#19212A, #0E1319)', color: '#EEF3F6',
    opacity: enter * (1 - exit), transform: `translateY(${(1 - enter) * 12 + exit * 8}px)`}}>
    <div style={{fontSize: vertical ? 34 : 30, fontWeight: 750, marginBottom: 14}}>{title}</div>
    <div style={{display: 'flex', gap: 20}}>{['home', 'away'].map((side, i) =>
      <div key={side} style={{flex: 1, display: 'flex', alignItems: 'center', gap: 14,
        background: '#232D38', padding: '8px 16px', borderBottom: `3px solid ${teams[side].line ?? teams[side].primary}`}}>
        {teams[side].logo && <Img src={staticFile(teams[side].logo)}
          style={{width: 48, height: 48, objectFit: 'contain'}}/>}
        <b style={{fontSize: 26, flex: 1}}>{teams[side].acronym}</b>
        <b style={{fontSize: 40, fontVariantNumeric: 'tabular-nums'}}>{data.score[i]}</b>
      </div>)}</div>
  </div>;
};
