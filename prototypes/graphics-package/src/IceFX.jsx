// PROTOTYPE — ice effects for the 3D stingers: rink shavings (skate-stop spray), mist puffs, drifting snow,
// and a procedural frost/skate-scrape layer that grows in from the slab edges (after the old stinger's look).
import React, {useMemo, useRef, useLayoutEffect} from 'react';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';

const rnd = (seed) => { const x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };

// ---- value noise / fbm (deterministic) ----
const hash2 = (x, y) => rnd(x * 127.1 + y * 311.7);
const vnoise = (x, y) => {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const s = (t) => t * t * (3 - 2 * t);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * s(xf) + (c - a) * s(yf) + (a - b - c + d) * s(xf) * s(yf);
};
const fbm = (x, y, oct = 4) => { let v = 0, amp = 0.5, fr = 1; for (let i = 0; i < oct; i++) { v += amp * vnoise(x * fr, y * fr); fr *= 2.03; amp *= 0.5; } return v; };

// Frost colour/alpha texture (scrapes + grain + crystalline veins) and a growth map (edges first).
export const makeFrost = (W = 1024, H = 300) => {
  const col = new Uint8Array(W * H * 4), grow = new Uint8Array(W * H * 4);
  // skate scrapes: long thin horizontal strokes, clustered in bands, slightly tilted
  const scrapes = Array.from({length: 190}).map((_, i) => ({
    y: rnd(i + 1) * H, x0: rnd(i + 2) * W * 1.1 - W * 0.1, len: W * (0.08 + rnd(i + 3) * 0.55), th: 0.6 + rnd(i + 4) * 3.2,
    tilt: (rnd(i + 5) - 0.5) * 0.04, a: 0.25 + rnd(i + 6) * 0.6,
  }));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W, v = y / H;
      const edge = Math.min(u, 1 - u) * (W / H) * 0.9 + 0 * v, edgeV = Math.min(v, 1 - v);
      const edgeDist = Math.min(edge, edgeV * 1.6);                      // 0 at the border
      const n = fbm(u * 14, v * 5);
      const ridge = 1 - Math.abs(fbm(u * 22 + 3, v * 9 + 7) * 2 - 1);    // crystalline veins
      let s = 0;
      for (const sc of scrapes) {
        const yy = sc.y + (x - sc.x0) * sc.tilt;
        if (x < sc.x0 || x > sc.x0 + sc.len) continue;
        const d = Math.abs(y - yy);
        if (d < sc.th * 2.2) {
          const rough = vnoise(x * 0.08, sc.y) * 0.8 + 0.2;
          s = Math.max(s, sc.a * rough * Math.max(0, 1 - d / (sc.th * 2.2)) * (0.5 + 0.5 * vnoise(x * 0.02, sc.y * 3)));
        }
      }
      const grain = hash2(x, y) > 0.82 ? 0.35 * hash2(x + 7, y + 3) : 0;
      const veins = Math.pow(ridge, 9) * 0.35;
      const a = Math.min(1, s * 0.9 + grain + veins * (0.4 + n * 0.6));
      const i4 = (y * W + x) * 4;
      col[i4] = 225; col[i4 + 1] = 242; col[i4 + 2] = 255; col[i4 + 3] = Math.round(a * 255);
      const g = Math.min(1, edgeDist * 1.25 + (n - 0.5) * 0.45 + 0.05);
      grow[i4] = grow[i4 + 1] = grow[i4 + 2] = Math.round(Math.max(0, g) * 255); grow[i4 + 3] = 255;
    }
  }
  const mk = (d) => { const t = new THREE.DataTexture(d, W, H, THREE.RGBAFormat); t.flipY = false; t.needsUpdate = true; t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter; return t; };
  const frostTex = mk(col); frostTex.colorSpace = THREE.SRGBColorSpace;
  return {frostTex, growthTex: mk(grow)};
};

const frostVert = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`;
const frostFrag = `
uniform sampler2D frostMap; uniform sampler2D growthMap; uniform float uProgress; uniform float uOpacity; uniform float uFront;
varying vec2 vUv;
void main(){
  vec4 c = texture2D(frostMap, vUv);
  float g = texture2D(growthMap, vUv).r;
  float m = smoothstep(uProgress, uProgress - 0.07, g);                 // frozen where growth < progress
  float front = smoothstep(0.05, 0.0, abs(g - uProgress)) * step(0.02, uProgress);
  float haze = m * 0.05;                                                 // milky frost film
  float faint = c.a * 0.28 * smoothstep(0.0, 0.25, uProgress);          // scrapes across the whole face (old stinger)
  float a = clamp(max(c.a * m, faint) + haze + front * 0.55 * uFront, 0.0, 1.0) * uOpacity;
  vec3 rgb = mix(c.rgb, vec3(1.6, 1.75, 1.9), front * uFront);                  // >1 so bloom catches the freezing front
  gl_FragColor = vec4(rgb, a);
}`;

// Parallelogram geometry with 0..1 UVs across its bounding box.
export const shapeGeoUV = (shape) => {
  const g = new THREE.ShapeGeometry(shape);
  g.computeBoundingBox();
  const {min, max} = g.boundingBox; const p = g.attributes.position; const uv = g.attributes.uv;
  for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) - min.x) / (max.x - min.x), (p.getY(i) - min.y) / (max.y - min.y));
  uv.needsUpdate = true;
  return g;
};

export const FrostLayer = ({shape, progress, opacity = 1, z = 0.3, front = 1}) => {
  const {frostTex, growthTex} = useMemo(() => makeFrost(), []);
  const geo = useMemo(() => shapeGeoUV(shape), [shape]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {frostMap: {value: frostTex}, growthMap: {value: growthTex}, uProgress: {value: 0}, uOpacity: {value: 1}, uFront: {value: 1}},
    vertexShader: frostVert, fragmentShader: frostFrag, transparent: true, depthWrite: false, toneMapped: false,
  }), [frostTex, growthTex]);
  mat.uniforms.uProgress.value = progress; mat.uniforms.uOpacity.value = opacity; mat.uniforms.uFront.value = front;
  return <mesh geometry={geo} material={mat} position={[0, 0, z]} />;
};

// ---- rink shavings: instanced snowy flakes with closed-form drag + gravity (deterministic per frame) ----
const flakeGeo = () => {
  const g = new THREE.IcosahedronGeometry(1, 0);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, p.getX(i) * (0.6 + rnd(i) * 0.8), p.getY(i) * 0.28, p.getZ(i) * (0.6 + rnd(i + 9) * 0.8));
  g.computeVertexNormals();
  return g;
};

// emitters: [{t0, t1, origin:[x,y,z], spread:[x,y,z], v:[vx,vy,vz], vJit:[...], size:[min,max], count}]
export const Shavings = ({f, fps = 60, emitters, gravity = 9, drag = 2.2}) => {
  const ref = useRef();
  const geo = useMemo(flakeGeo, []);
  const parts = useMemo(() => emitters.flatMap((e, ei) => Array.from({length: e.count}).map((_, i) => {
    const r = (k) => rnd(ei * 10007 + i * 31 + k);
    return {
      te: e.t0 + r(1) * (e.t1 - e.t0),
      p0: e.origin.map((o, j) => o + (r(2 + j) - 0.5) * e.spread[j]),
      v0: e.v.map((v, j) => v + (r(5 + j) - 0.5) * e.vJit[j]),
      size: e.size[0] + Math.pow(r(8), 2.2) * (e.size[1] - e.size[0]),
      rot0: [r(9) * 6, r(10) * 6, r(11) * 6], w: [(r(12) - 0.5) * 14, (r(13) - 0.5) * 14, (r(14) - 0.5) * 14],
      life: 1.1 + r(15) * 1.6,
    };
  })), [emitters]);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  useLayoutEffect(() => {
    const m = ref.current; if (!m) return;
    parts.forEach((p, i) => {
      const t = (f - p.te) / fps;
      if (t < 0 || t > p.life) { dummy.scale.setScalar(0); dummy.updateMatrix(); m.setMatrixAt(i, dummy.matrix); return; }
      const e = Math.exp(-drag * t), k = (1 - e) / drag;
      const x = p.p0[0] + p.v0[0] * k;
      const y = p.p0[1] + p.v0[1] * k - (gravity / drag) * t + (gravity / (drag * drag)) * (1 - e);
      const z = p.p0[2] + p.v0[2] * k;
      const fade = t < p.life * 0.55 ? 1 : 1 - (t - p.life * 0.55) / (p.life * 0.45);
      dummy.position.set(x, y, z);
      dummy.rotation.set(p.rot0[0] + p.w[0] * t, p.rot0[1] + p.w[1] * t, p.rot0[2] + p.w[2] * t);
      dummy.scale.setScalar(p.size * Math.max(0, fade));
      dummy.updateMatrix(); m.setMatrixAt(i, dummy.matrix);
    });
    m.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[geo, undefined, parts.length]} frustumCulled={false}>
      <meshStandardMaterial color="#f4fbff" emissive="#9fd3f0" emissiveIntensity={0.35} roughness={0.85} metalness={0} flatShading />
    </instancedMesh>
  );
};

// Soft mist puffs (billboard sprites) that bloom out of the spray and fade.
const softTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const S = 128, c = new Uint8Array(S * S * 4);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const d = Math.hypot(x - S / 2, y - S / 2) / (S / 2);
      const n = fbm(x / 18, y / 18);
      const a = Math.max(0, 1 - d) ** 1.8 * (0.55 + 0.45 * n);
      const i = (y * S + x) * 4; c[i] = 235; c[i + 1] = 245; c[i + 2] = 255; c[i + 3] = Math.round(a * 255);
    }
    t = new THREE.DataTexture(c, S, S, THREE.RGBAFormat); t.needsUpdate = true; t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
    return t;
  };
})();

export const Mist = ({f, fps = 60, puffs}) => {
  const tex = useMemo(softTex, []);
  return puffs.map((p, i) => {
    const t = (f - p.t0) / fps;
    if (t < 0 || t > p.life) return null;
    const k = 1 - Math.exp(-2 * t);
    const s = p.size * (0.5 + 1.2 * k);
    const o = p.opacity * Math.min(1, t * 6) * (1 - t / p.life);
    return (
      <sprite key={i} position={[p.x + p.vx * k, p.y + p.vy * k, p.z]} scale={[s, s * 0.7, 1]}>
        <spriteMaterial map={tex} transparent opacity={o} depthWrite={false} rotation={p.rot} />
      </sprite>
    );
  });
};

// Fine snow motes drifting slowly through the hold (depth parallax), instanced.
export const Snow = ({f, count = 260, area = [26, 14, 18], fall = 0.012}) => {
  const ref = useRef();
  const geo = useMemo(() => new THREE.IcosahedronGeometry(1, 0), []);
  const seeds = useMemo(() => Array.from({length: count}).map((_, i) => ({
    x: (rnd(i + 3) - 0.5) * area[0], y: (rnd(i + 5) - 0.5) * area[1], z: -area[2] + rnd(i + 7) * area[2] + 4,
    s: 0.015 + rnd(i + 9) * 0.04, sp: 0.5 + rnd(i + 11), ph: rnd(i + 13) * 6,
  })), [count]);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  useLayoutEffect(() => {
    const m = ref.current; if (!m) return;
    seeds.forEach((p, i) => {
      const y = ((p.y - f * fall * p.sp + area[1] / 2) % area[1] + area[1]) % area[1] - area[1] / 2;
      dummy.position.set(p.x + Math.sin(f / 40 + p.ph) * 0.3, y, p.z);
      dummy.scale.setScalar(p.s); dummy.updateMatrix(); m.setMatrixAt(i, dummy.matrix);
    });
    m.instanceMatrix.needsUpdate = true;
  });
  return (
    <instancedMesh ref={ref} args={[geo, undefined, count]} frustumCulled={false}>
      <meshBasicMaterial color="#dff3ff" toneMapped={false} transparent opacity={0.55} />
    </instancedMesh>
  );
};

// ---- "skating fast": ice floor rushing under a low tracking camera, rink lines, speed streaks ----
const makeIceFloorTex = () => {
  const S = 512, c = new Uint8Array(S * S * 4);
  const cuts = Array.from({length: 90}).map((_, i) => ({x: rnd(i + 900) * S, w: 0.5 + rnd(i + 901) * 2.2, a: 0.08 + rnd(i + 902) * 0.28, curve: (rnd(i + 903) - 0.5) * 0.25}));
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const n = fbm(x / 60, y / 60, 3);
    let v = 0.72 + (n - 0.5) * 0.22;
    for (const k of cuts) { const cx = k.x + Math.sin(y / S * Math.PI * 2) * k.curve * 40; const d = Math.abs(x - cx); if (d < k.w) v += k.a * (1 - d / k.w); }
    v = Math.min(1, v);
    const i = (y * S + x) * 4; c[i] = Math.round(190 * v); c[i + 1] = Math.round(215 * v); c[i + 2] = Math.round(235 * v); c[i + 3] = 255;
  }
  const t = new THREE.DataTexture(c, S, S, THREE.RGBAFormat); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(4, 10);
  t.colorSpace = THREE.SRGBColorSpace; t.minFilter = THREE.LinearMipmapLinearFilter; t.generateMipmaps = true; t.anisotropy = 8; t.needsUpdate = true;
  return t;
};

// travel: distance skated so far (world units); speed: current speed (0..1.5) for streak stretch.
export const IceRush = ({travel, speed, floorY = -1.7}) => {
  const tex = useMemo(makeIceFloorTex, []);
  tex.offset.y = travel / 12;   // 120-unit floor, repeat 10 → 12 units per tile
  const lines = [{z0: -34, c: '#1f5fd6', w: 1.1}, {z0: -62, c: '#d6262f', w: 1.1}, {z0: -90, c: '#1f5fd6', w: 1.1}];
  const streaks = useMemo(() => Array.from({length: 70}).map((_, i) => ({
    x: (rnd(i + 700) - 0.5) * 34, y: floorY + 0.05 + rnd(i + 701) * 9, z0: -rnd(i + 702) * 90, len: 0.6 + rnd(i + 703) * 2.2,
    c: rnd(i + 704) > 0.55 ? '#ffffff' : '#5AA4D0', side: Math.abs((rnd(i + 700) - 0.5) * 34) > 5 || rnd(i + 701) > 0.55,
  })).filter((s) => s.side), [floorY]);
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY, -50]}>
        <planeGeometry args={[80, 120]} />
        <meshStandardMaterial map={tex} roughness={0.22} metalness={0.15} color="#d8ecf8" />
      </mesh>
      {lines.map((l, i) => {
        const z = ((l.z0 + travel) % 120 + 120) % 120 - 110;
        return (
          <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY + 0.01, z]}>
            <planeGeometry args={[80, l.w]} />
            <meshStandardMaterial color={l.c} roughness={0.3} transparent opacity={0.85} />
          </mesh>
        );
      })}
      {streaks.map((s, i) => {
        const z = ((s.z0 + travel * 1.4) % 90 + 90) % 90 - 80;
        const stretch = s.len * (0.3 + speed * 5);
        return (
          <mesh key={`s${i}`} position={[s.x, s.y, z]}>
            <boxGeometry args={[0.03, 0.03, stretch]} />
            <meshBasicMaterial color={s.c} toneMapped={false} transparent opacity={Math.min(1, 0.15 + speed)} />
          </mesh>
        );
      })}
    </>
  );
};

// ---- round 2: soft GPU particles (no facets, no floor clipping) ----
// Spray: rink-shaving powder as soft round points. Positions come from closed-form drag + gravity in the
// vertex shader; particles that reach the ice settle on it instead of passing through. Each particle
// draws a short trail of sub-samples inside a 1/120 s shutter, so fast grains read as motion-blurred.
const sprayVert = `
attribute vec3 p0; attribute vec3 v0; attribute vec4 meta; attribute float alpha; // meta: te (s), life (s), size, lag (0..1)
uniform float uTime; uniform float uGravity; uniform float uDrag; uniform float uFloor; uniform float uScale; uniform float uShutter;
varying float vAlpha; varying float vHot;
void main(){
  float t = uTime - meta.x - meta.w * uShutter;
  if (t < 0.0 || t > meta.y) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; vAlpha = 0.0; return; }
  float e = exp(-uDrag * t), k = (1.0 - e) / uDrag;
  vec3 p = p0 + v0 * k;
  p.y += -(uGravity / uDrag) * t + (uGravity / (uDrag * uDrag)) * (1.0 - e);
  float landed = step(p.y, uFloor + meta.z * 0.5);
  p.y = max(p.y, uFloor + meta.z * 0.5);
  float life = t / meta.y;
  float fade = smoothstep(0.0, 0.04, t) * (1.0 - smoothstep(0.45, 1.0, life)) * (1.0 - 0.85 * landed);
  vAlpha = fade * (1.0 - meta.w * 0.8) * alpha;
  vHot = 1.0 - smoothstep(0.0, 0.35, t);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.5, meta.z * uScale / -mv.z);
  vAlpha *= clamp(gl_PointSize / 1.5, 0.0, 1.0);
}`;
const sprayFrag = `
uniform float uOpacity; varying float vAlpha; varying float vHot;
void main(){
  float r = length(gl_PointCoord - 0.5) * 2.0;
  float a = pow(1.0 - smoothstep(0.0, 1.0, r), 1.6) * vAlpha * uOpacity;
  if (a < 0.003) discard;
  vec3 col = mix(vec3(0.84, 0.92, 1.0), vec3(1.02, 1.05, 1.08), vHot);
  gl_FragColor = vec4(col, a);
}`;

// emitters: [{t0, t1 (frames), origin, spread, v, vJit, size:[min,max] (world units), count, life:[min,max] (s), alpha}]
export const Spray = ({f, fps = 60, emitters, floorY = -1.55, gravity = 7, drag = 2.6, trail = 3}) => {
  const {gl, camera, size} = useThree();
  const geo = useMemo(() => {
    const P = [], V = [], M = [], A = [];
    emitters.forEach((e, ei) => {
      for (let i = 0; i < e.count; i++) {
        const r = (k) => rnd(ei * 10007 + i * 31 + k);
        // bias launch direction into a fan: faster grains fly lower and further
        const fast = Math.pow(r(20), 0.7);
        const te = (e.t0 + r(1) * (e.t1 - e.t0)) / fps;
        const p0 = e.origin.map((o, j) => o + (r(2 + j) - 0.5) * e.spread[j]);
        const v0 = e.v.map((v, j) => (v + (r(5 + j) - 0.5) * e.vJit[j]) * (0.55 + 0.6 * fast));
        const sz = e.size[0] + Math.pow(r(8), 2.4) * (e.size[1] - e.size[0]);
        const life = e.life[0] + r(15) * (e.life[1] - e.life[0]);
        for (let l = 0; l < trail; l++) { P.push(...p0); V.push(...v0); M.push(te, life, sz, l / trail); A.push(e.alpha ?? 1); }
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));   // unused by the shader; keeps three happy
    g.setAttribute('p0', new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute('v0', new THREE.Float32BufferAttribute(V, 3));
    g.setAttribute('meta', new THREE.Float32BufferAttribute(M, 4));
    g.setAttribute('alpha', new THREE.Float32BufferAttribute(A, 1));
    return g;
  }, [emitters, fps, trail]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {uTime: {value: 0}, uGravity: {value: gravity}, uDrag: {value: drag}, uFloor: {value: floorY}, uScale: {value: 1}, uShutter: {value: 1 / 120}, uOpacity: {value: 1}},
    vertexShader: sprayVert, fragmentShader: sprayFrag, transparent: true, depthWrite: false, toneMapped: false,
  }), [gravity, drag, floorY]);
  mat.uniforms.uTime.value = f / fps;
  mat.uniforms.uScale.value = (size.height * gl.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
  return <points geometry={geo} material={mat} frustumCulled={false} renderOrder={5} />;
};

// Soft cloud texture: radial falloff that reaches zero well inside the quad, so no square edges show.
const cloudTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const S = 256, c = new Uint8Array(S * S * 4);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const dx = (x - S / 2) / (S / 2), dy = (y - S / 2) / (S / 2);
      const d = Math.hypot(dx, dy);
      const n = fbm(x / 34 + 3, y / 34 + 7, 5);
      const edge = 1 - THREE.MathUtils.smoothstep(d, 0.15 + 0.35 * n, 0.92);
      const a = Math.max(0, edge) ** 2 * (0.6 + 0.4 * n);
      const i = (y * S + x) * 4; c[i] = 236; c[i + 1] = 245; c[i + 2] = 255; c[i + 3] = Math.round(a * 255);
    }
    t = new THREE.DataTexture(c, S, S, THREE.RGBAFormat); t.needsUpdate = true; t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
    return t;
  };
})();

// Mist puffs: camera-facing sprites that ignore the depth buffer, so the ice and the slab never cut a
// straight edge through them. Puffs rise off the ice as they grow.
export const SoftMist = ({f, fps = 60, puffs, floorY = -1.55}) => {
  const tex = useMemo(cloudTex, []);
  return puffs.map((p, i) => {
    const t = (f - p.t0) / fps;
    if (t < 0 || t > p.life) return null;
    const k = 1 - Math.exp(-2.2 * t);
    const s = p.size * (0.45 + 1.3 * k);
    const o = p.opacity * Math.min(1, t * 5) * Math.pow(1 - t / p.life, 1.5);
    const y = Math.max(p.y + p.vy * k, floorY + s * 0.18);
    return (
      <sprite key={i} position={[p.x + p.vx * k, y, p.z]} scale={[s * 1.4, s * 0.8, 1]} renderOrder={6}>
        <spriteMaterial map={tex} transparent opacity={o} depthWrite={false} depthTest={false} rotation={p.rot + t * 0.2} toneMapped={false} />
      </sprite>
    );
  });
};

// Drifting snow motes as soft points (replaces the faceted instanced motes).
const snowVert = `
attribute vec4 seed; // x, y, z, size
uniform float uF; uniform float uFall; uniform vec3 uArea; uniform float uScale;
varying float vA;
void main(){
  float sp = 0.5 + fract(seed.x * 13.7);
  float y = mod(seed.y - uF * uFall * sp + uArea.y * 0.5, uArea.y) - uArea.y * 0.5;
  vec3 p = vec3(seed.x + sin(uF / 40.0 + seed.z) * 0.3, y, seed.z);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = max(1.2, seed.w * uScale / -mv.z);
  vA = clamp(gl_PointSize / 1.2, 0.0, 1.0) * smoothstep(40.0, 10.0, -mv.z);
}`;
const snowFrag = `
uniform float uOpacity; varying float vA;
void main(){
  float r = length(gl_PointCoord - 0.5) * 2.0;
  float a = pow(1.0 - smoothstep(0.0, 1.0, r), 2.0) * vA * uOpacity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(0.9, 0.96, 1.0, a);
}`;
export const SoftSnow = ({f, count = 260, area = [26, 14, 18], fall = 0.012, opacity = 0.7}) => {
  const {gl, camera, size} = useThree();
  const geo = useMemo(() => {
    const S = [], P = [];
    for (let i = 0; i < count; i++) {
      S.push((rnd(i + 3) - 0.5) * area[0], (rnd(i + 5) - 0.5) * area[1], -area[2] + rnd(i + 7) * area[2] + 4, 0.03 + rnd(i + 9) * 0.07);
      P.push(0, 0, 0);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute('seed', new THREE.Float32BufferAttribute(S, 4));
    return g;
  }, [count]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {uF: {value: 0}, uFall: {value: fall}, uArea: {value: new THREE.Vector3(...area)}, uScale: {value: 1}, uOpacity: {value: opacity}},
    vertexShader: snowVert, fragmentShader: snowFrag, transparent: true, depthWrite: false, toneMapped: false,
  }), [fall, opacity]);
  mat.uniforms.uF.value = f;
  mat.uniforms.uScale.value = (size.height * gl.getPixelRatio()) / (2 * Math.tan((camera.fov * Math.PI) / 360));
  return <points geometry={geo} material={mat} frustumCulled={false} renderOrder={4} />;
};
