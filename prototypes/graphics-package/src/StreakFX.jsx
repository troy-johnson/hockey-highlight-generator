// PROTOTYPE — round 3 polish: motion-blurred soft particles, a single painted slab (body + caps as one
// surface), and a lens-frost layer for the period transition.
//
// Streak particles are camera-facing quads stretched along each particle's screen-space motion during a
// shutter interval (the "stretched billboard" technique). The fragment shader draws a soft capsule, so a
// fast grain reads as a smooth blur line, never as a row of dots or a hard 1-px line.
import React, {useMemo} from 'react';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';

const rnd = (seed) => { const x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };
const hash2 = (x, y) => rnd(x * 127.1 + y * 311.7);
const vnoise = (x, y) => {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const s = (t) => t * t * (3 - 2 * t);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * s(xf) + (c - a) * s(yf) + (a - b - c + d) * s(xf) * s(yf);
};
const fbm = (x, y, oct = 4) => { let v = 0, amp = 0.5, fr = 1; for (let i = 0; i < oct; i++) { v += amp * vnoise(x * fr, y * fr); fr *= 2.03; amp *= 0.5; } return v; };

// ---------- shared quad builder ----------
const quadHead = `
attribute vec2 corner;
uniform vec2 uRes; uniform float uScale; uniform float uMinR; uniform float uMaxR;
varying vec2 vLocal; varying float vLen; varying float vR; varying float vA;
void buildQuad(vec3 pNow, vec3 pPrev, float worldSize, float alpha){
  vec4 mvA = modelViewMatrix * vec4(pNow, 1.0);
  vec4 mvB = modelViewMatrix * vec4(pPrev, 1.0);
  if (mvA.z > -0.3) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vA = 0.0; return; }
  mvB.z = min(mvB.z, -0.3);
  vec4 cA = projectionMatrix * mvA, cB = projectionMatrix * mvB;
  vec2 sA = cA.xy / cA.w * uRes * 0.5, sB = cB.xy / cB.w * uRes * 0.5;
  vec2 d = sA - sB; float len = min(length(d), uRes.y * 0.35);
  vec2 n = len > 0.001 ? normalize(d) : vec2(1.0, 0.0), pp = vec2(-n.y, n.x);
  float r = worldSize * uScale / -mvA.z;
  float small = clamp(r / uMinR, 0.0, 1.0);
  float big = clamp(uMaxR / max(r, 0.001), 0.0, 1.0);          // defocused blobs near the lens stay faint
  r = clamp(r, uMinR, uMaxR * 3.0);
  float along = mix(-len - r, r, corner.x);
  vec2 s = sA + n * along + pp * corner.y * r;
  gl_Position = vec4(s / (uRes * 0.5) * cA.w, cA.z, cA.w);
  vLocal = vec2(along, corner.y * r); vLen = len; vR = r;
  vA = alpha * small * small * big * pow(clamp(2.0 * r / (len + 2.0 * r), 0.0, 1.0), 0.55);
}`;
const quadFrag = `
uniform vec3 uColor; uniform vec3 uHotColor; uniform float uOpacity;
varying vec2 vLocal; varying float vLen; varying float vR; varying float vA; varying float vHot;
void main(){
  float x = vLocal.x;
  float dx = x > 0.0 ? x : (x < -vLen ? -vLen - x : 0.0);
  float d = length(vec2(dx, vLocal.y)) / vR;
  float tail = vLen > 0.5 ? mix(0.35, 1.0, clamp((x + vLen) / vLen, 0.0, 1.0)) : 1.0;
  float a = pow(1.0 - smoothstep(0.0, 1.0, d), 1.7) * vA * tail * uOpacity;
  if (a < 0.002) discard;
  gl_FragColor = vec4(mix(uColor, uHotColor, vHot), a);
}`;

// Expand per-particle attributes into 4-corner quads with an index buffer.
const quadGeometry = (count, attrs) => {
  const g = new THREE.BufferGeometry();
  const corner = new Float32Array(count * 8), pos = new Float32Array(count * 12), idx = new Uint32Array(count * 6);
  for (let i = 0; i < count; i++) {
    corner.set([0, -1, 1, -1, 1, 1, 0, 1], i * 8);
    idx.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('corner', new THREE.BufferAttribute(corner, 2));
  for (const [name, size, arr] of attrs) {
    const out = new Float32Array(count * 4 * size);
    for (let i = 0; i < count; i++) for (let k = 0; k < 4; k++) for (let j = 0; j < size; j++) out[(i * 4 + k) * size + j] = arr[i * size + j];
    g.setAttribute(name, new THREE.BufferAttribute(out, size));
  }
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
};

const useQuadUniforms = (mat, {minR = 0.9, maxR = 40} = {}) => {
  const {gl, camera, size} = useThree();
  const pr = gl.getPixelRatio();
  mat.uniforms.uRes.value.set(size.width * pr, size.height * pr);
  mat.uniforms.uScale.value = (size.height * pr) / (2 * Math.tan((camera.fov * Math.PI) / 360));
  mat.uniforms.uMinR.value = minR * pr;
  mat.uniforms.uMaxR.value = maxR * pr;
};
const baseUniforms = (color, hot) => ({
  uRes: {value: new THREE.Vector2(1, 1)}, uScale: {value: 1}, uMinR: {value: 1}, uMaxR: {value: 40}, uOpacity: {value: 1},
  uColor: {value: new THREE.Color(...color)}, uHotColor: {value: new THREE.Color(...hot)},
});

// ---------- spray: closed-form drag + gravity, settles on the ice ----------
const sprayVert = quadHead + `
attribute vec3 p0; attribute vec3 v0; attribute vec4 meta; // te (s), life (s), size, alpha
uniform float uTime; uniform float uShutter; uniform float uGravity; uniform float uDrag; uniform float uFloor; uniform float uHotTime;
varying float vHot;
vec3 at(float t){
  float e = exp(-uDrag * t), k = (1.0 - e) / uDrag;
  vec3 p = p0 + v0 * k;
  p.y += -(uGravity / uDrag) * t + (uGravity / (uDrag * uDrag)) * (1.0 - e);
  p.y = max(p.y, uFloor + meta.z * 0.5);
  return p;
}
void main(){
  float t = uTime - meta.x;
  vHot = 0.0;
  if (t < 0.0 || t > meta.y) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vA = 0.0; return; }
  float life = t / meta.y;
  vec3 pNow = at(t);
  float landed = step(pNow.y, uFloor + meta.z * 0.5 + 0.001);
  float fade = smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.4, 1.0, life)) * (1.0 - 0.8 * landed);
  vHot = 1.0 - smoothstep(0.0, uHotTime, t);
  buildQuad(pNow, at(max(0.0, t - uShutter)), meta.z, meta.w * fade);
}`;

// emitters: [{t0, t1 (frames), origin, spread, v, vJit, size:[min,max], life:[min,max] (s), count, alpha, move:[dx,dy,dz] per s}]
export const StreakSpray = ({f, fps = 60, emitters, floorY = -1.55, gravity = 7, drag = 2.6, shutter = 0.5,
  color = [0.82, 0.9, 1.0], hot = [1.0, 1.02, 1.05], hotTime = 0.3, minR = 0.9, maxR = 40, floor = true}) => {
  const geo = useMemo(() => {
    const P = [], V = [], M = [];
    emitters.forEach((e, ei) => {
      for (let i = 0; i < e.count; i++) {
        const r = (k) => rnd(ei * 10007 + i * 31 + k);
        const fast = Math.pow(r(20), 0.7);
        const u = r(1);
        const te = (e.t0 + u * (e.t1 - e.t0)) / fps;
        const mv = e.move || [0, 0, 0];
        const p0 = e.origin.map((o, j) => o + (r(2 + j) - 0.5) * e.spread[j] + mv[j] * u * (e.t1 - e.t0) / fps);
        const v0 = e.v.map((v, j) => (v + (r(5 + j) - 0.5) * e.vJit[j]) * (0.55 + 0.6 * fast));
        const sz = e.size[0] + Math.pow(r(8), 2.4) * (e.size[1] - e.size[0]);
        const life = e.life[0] + r(15) * (e.life[1] - e.life[0]);
        P.push(...p0); V.push(...v0); M.push(te, life, sz, e.alpha ?? 1);
      }
    });
    return quadGeometry(P.length / 3, [['p0', 3, P], ['v0', 3, V], ['meta', 4, M]]);
  }, [emitters, fps]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {...baseUniforms(color, hot), uTime: {value: 0}, uShutter: {value: shutter / fps}, uGravity: {value: gravity}, uDrag: {value: drag},
      uFloor: {value: floor ? floorY : -1e6}, uHotTime: {value: hotTime}},
    vertexShader: sprayVert, fragmentShader: quadFrag, transparent: true, depthWrite: false, toneMapped: false,
  }), []);
  mat.uniforms.uTime.value = f / fps;
  useQuadUniforms(mat, {minR, maxR});
  return <mesh geometry={geo} material={mat} frustumCulled={false} renderOrder={5} />;
};

// ---------- speed streaks: rink-relative lines that stretch with camera speed ----------
const speedVert = quadHead + `
attribute vec4 seed; // x, y, z0, size
uniform float uTravel; uniform float uTravelPrev; uniform float uFade;
varying float vHot;
float zAt(float tr){ return mod(seed.z + tr * 1.4, 90.0) - 80.0; }
void main(){
  vHot = step(0.55, fract(seed.x * 7.13));
  float zN = zAt(uTravel), zP = zAt(uTravelPrev);
  if (zP > zN) zP = zN;                                           // wrapped this frame: no streak
  float edge = smoothstep(-80.0, -70.0, zN) * (1.0 - smoothstep(4.0, 12.0, zN));
  buildQuad(vec3(seed.x, seed.y, zN), vec3(seed.x, seed.y, zP), seed.w, uFade * edge);
}`;
export const SpeedStreaks2 = ({travel, travelPrev, floorY = -1.55, count = 70, opacity = 0.8}) => {
  const geo = useMemo(() => {
    const S = [];
    for (let i = 0; i < count; i++) {
      const x = (rnd(i + 700) - 0.5) * 34, y = floorY + 0.4 + rnd(i + 701) * 9;
      if (!(Math.abs(x) > 5 || y > 2)) continue;
      S.push(x, y, -rnd(i + 702) * 90, 0.02 + rnd(i + 703) * 0.035);
    }
    return quadGeometry(S.length / 4, [['seed', 4, S]]);
  }, [count, floorY]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {...baseUniforms([0.35, 0.64, 0.82], [1, 1, 1]), uTravel: {value: 0}, uTravelPrev: {value: 0}, uFade: {value: 1}},
    vertexShader: speedVert, fragmentShader: quadFrag, transparent: true, depthWrite: false, toneMapped: false,
  }), []);
  mat.uniforms.uTravel.value = travel; mat.uniforms.uTravelPrev.value = travelPrev; mat.uniforms.uFade.value = opacity;
  useQuadUniforms(mat, {minR: 0.8, maxR: 6});
  return <mesh geometry={geo} material={mat} frustumCulled={false} renderOrder={3} />;
};

// ---------- painted slab: body and end caps on one surface ----------
export const para = (xa, xb, h, skew) => {
  const s = new THREE.Shape(), o = Math.tan(skew) * h;
  s.moveTo(xa + o / 2, h / 2); s.lineTo(xb + o / 2, h / 2); s.lineTo(xb - o / 2, -h / 2); s.lineTo(xa - o / 2, -h / 2); s.closePath();
  return s;
};
const shade = (hex, k) => { const c = new THREE.Color(hex); const t = k > 0 ? new THREE.Color('#ffffff') : new THREE.Color('#000000'); return '#' + c.lerp(t, Math.abs(k)).getHexString(); };

const paintSlab = ({w, h, skew, capW, color, capColor}) => {
  const o = Math.tan(skew) * h, W2 = w / 2 + capW;
  const minX = -W2 - o / 2, maxX = W2 + o / 2, bw = maxX - minX;
  const CW = 2048, CH = Math.round((CW * h) / bw);
  const X = (x) => ((x - minX) / bw) * CW, Y = (y) => ((h / 2 - y) / h) * CH;
  const mk = () => { const c = document.createElement('canvas'); c.width = CW; c.height = CH; return c; };
  const poly = (g, xa, xb) => { g.beginPath(); g.moveTo(X(xa + o / 2), Y(h / 2)); g.lineTo(X(xb + o / 2), Y(h / 2)); g.lineTo(X(xb - o / 2), Y(-h / 2)); g.lineTo(X(xa - o / 2), Y(-h / 2)); g.closePath(); };
  const brushed = (g, n, seed) => {
    for (let i = 0; i < n; i++) {
      const y = rnd(seed + i) * CH, x0 = rnd(seed + i + 0.3) * CW - CW * 0.2, len = CW * (0.1 + rnd(seed + i + 0.6) * 0.6);
      g.fillStyle = rnd(seed + i + 0.9) > 0.5 ? `rgba(255,255,255,${0.015 + rnd(seed + i + 1.2) * 0.035})` : `rgba(0,0,0,${0.02 + rnd(seed + i + 1.5) * 0.05})`;
      g.fillRect(x0, y, len, 0.6 + rnd(seed + i + 1.8) * 1.4);
    }
  };
  const map = mk(), g = map.getContext('2d');
  // body
  let gr = g.createLinearGradient(0, 0, 0, CH);
  gr.addColorStop(0, shade(color, 0.16)); gr.addColorStop(0.45, color); gr.addColorStop(1, shade(color, -0.35));
  g.fillStyle = gr; g.fillRect(0, 0, CW, CH);
  brushed(g, 1400, 11);
  // caps, same slant, with a machined seam where they meet the body
  for (const [xa, xb, seam, hi] of [[-W2, -w / 2, -w / 2, -w / 2 - 0.012], [w / 2, W2, w / 2, w / 2 + 0.012]]) {
    g.save(); poly(g, xa, xb); g.clip();
    gr = g.createLinearGradient(0, 0, 0, CH);
    gr.addColorStop(0, shade(capColor, 0.3)); gr.addColorStop(0.5, capColor); gr.addColorStop(1, shade(capColor, -0.3));
    g.fillStyle = gr; g.fillRect(0, 0, CW, CH);
    brushed(g, 500, 97 + xa);
    g.restore();
    g.lineWidth = 5; g.strokeStyle = 'rgba(0,0,0,0.55)';
    g.beginPath(); g.moveTo(X(seam + o / 2), Y(h / 2)); g.lineTo(X(seam - o / 2), Y(-h / 2)); g.stroke();
    g.lineWidth = 2.5; g.strokeStyle = 'rgba(255,255,255,0.7)';
    g.beginPath(); g.moveTo(X(hi + o / 2), Y(h / 2)); g.lineTo(X(hi - o / 2), Y(-h / 2)); g.stroke();
  }
  // soft top highlight and bottom shade across the whole piece so body and caps share one light
  gr = g.createLinearGradient(0, 0, 0, CH);
  gr.addColorStop(0, 'rgba(255,255,255,0.18)'); gr.addColorStop(0.08, 'rgba(255,255,255,0)'); gr.addColorStop(0.85, 'rgba(0,0,0,0)'); gr.addColorStop(1, 'rgba(0,0,0,0.25)');
  g.fillStyle = gr; g.fillRect(0, 0, CW, CH);
  // emissive mask: caps only
  const em = mk(), e = em.getContext('2d');
  e.fillStyle = '#000'; e.fillRect(0, 0, CW, CH); e.fillStyle = '#fff';
  poly(e, -W2, -w / 2 - 0.01); e.fill(); poly(e, w / 2 + 0.01, W2); e.fill();
  const t1 = new THREE.CanvasTexture(map); t1.colorSpace = THREE.SRGBColorSpace; t1.anisotropy = 16;
  const t2 = new THREE.CanvasTexture(em); t2.anisotropy = 16;
  return {map: t1, emissiveMap: t2, outer: para(-W2, W2, h, skew)};
};

// UVs from x/y across the bounding box, for every face (side walls pick up the edge colours).
const boxUV = (g) => {
  g.computeBoundingBox();
  const {min, max} = g.boundingBox, p = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < p.count; i++) uv.setXY(i, (p.getX(i) - min.x) / (max.x - min.x), (p.getY(i) - min.y) / (max.y - min.y));
  uv.needsUpdate = true;
  return g;
};

export const PaintedSlab = ({w, h, skew, depth, color, capColor, capW, glow = 0, children}) => {
  const {map, emissiveMap, outer} = useMemo(() => paintSlab({w, h, skew, capW, color, capColor}), [w, h, skew, capW, color, capColor]);
  const geo = useMemo(() => boxUV(new THREE.ExtrudeGeometry(outer, {depth, bevelEnabled: true, bevelThickness: 0.05, bevelSize: 0.05, bevelSegments: 5, curveSegments: 1})), [outer, depth]);
  const o = Math.tan(skew) * h, W2 = w / 2 + capW;
  return (
    <group>
      <mesh geometry={geo} position={[0, 0, -depth / 2 - 0.025]}>
        <meshPhysicalMaterial map={map} emissiveMap={emissiveMap} emissive={capColor} emissiveIntensity={0.12 + 0.9 * glow}
          metalness={0.4} roughness={0.36} clearcoat={0.7} clearcoatRoughness={0.2} />
      </mesh>
      {[[h / 2, o / 2], [-h / 2, -o / 2]].map(([y, x], i) => (
        <mesh key={i} position={[x, y, depth / 2 + 0.03]}>
          <boxGeometry args={[W2 * 2, 0.022, 0.024]} />
          <meshStandardMaterial color="#e9f6ff" emissive="#cfeaff" emissiveIntensity={0.25 + 0.7 * glow} roughness={0.3} />
        </mesh>
      ))}
      {children}
    </group>
  );
};
export const outerShape = (w, h, skew, capW) => para(-(w / 2 + capW), w / 2 + capW, h, skew);

// ---------- lens frost: frost crystallising on the camera glass ----------
// Crystal map: ridged noise at several scales plus needle crystals that branch off seed points.
// Growth map: distance to the nearest seed (where snow hit the glass), broken up with noise.
export const makeLensFrost = (W = 960, H = 540, seeds) => {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  // needle crystals (fern-like): a trunk with short side branches, drawn additively
  g.globalCompositeOperation = 'lighter'; g.lineCap = 'round';
  // fine feathery crystals, dense toward the frame edges and the snow-hit seeds, never a clip-art fern
  const needle = (x, y, ang, len, depth, k, al) => {
    const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
    g.strokeStyle = `rgba(255,255,255,${al})`; g.lineWidth = 0.35 + depth * 0.25;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x2, y2); g.stroke();
    if (depth <= 0) return;
    const nb = 4 + Math.floor(rnd(k) * 5);
    for (let i = 1; i <= nb; i++) {
      const t = i / (nb + 1), bx = x + (x2 - x) * t, by = y + (y2 - y) * t;
      for (const side of [-1, 1]) if (rnd(k + i * 7 + side) > 0.25) needle(bx, by, ang + side * (1.0 + (rnd(k + i + side) - 0.5) * 0.5), len * (0.22 + rnd(k + i * 3) * 0.2) * (1 - t * 0.6), depth - 1, k * 1.7 + i * 3 + side, al * 0.8);
    }
  };
  for (let i = 0; i < 2600; i++) {
    let x, y;
    if (i % 3 === 0) { const s = seeds[i % seeds.length]; x = s[0] * W + (rnd(i + 40) - 0.5) * W * 0.3; y = s[1] * H + (rnd(i + 41) - 0.5) * H * 0.3; }
    else { const e = Math.pow(rnd(i + 44), 2.2) * 0.32, side = Math.floor(rnd(i + 45) * 4), t = rnd(i + 46);
      [x, y] = [[e * W, t * H], [(1 - e) * W, t * H], [t * W, e * H], [t * W, (1 - e) * H]][side]; }
    const edgeD = Math.min(x / W, 1 - x / W, y / H, 1 - y / H);
    needle(x, y, rnd(i + 42) * Math.PI * 2, 5 + rnd(i + 43) * 22 * (1 - edgeD), 2, i + 1, 0.05 + 0.1 * rnd(i + 47));
  }
  g.globalCompositeOperation = 'source-over';
  const soft = document.createElement('canvas'); soft.width = W; soft.height = H;
  const sg = soft.getContext('2d'); sg.filter = 'blur(0.5px)'; sg.drawImage(c, 0, 0);
  const needles = sg.getImageData(0, 0, W, H).data;
  const col = new Uint8Array(W * H * 4), grow = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const u = x / W, v = y / H;
    const cloud = fbm(u * 3 + 9, v * 2, 4);                            // smooth milky density, no ridges
    const i4 = (y * W + x) * 4;
    const sparkle = hash2(x, y) > 0.985 ? 0.5 * hash2(x + 3, y + 1) : 0;
    const a = Math.min(1, needles[i4] / 255 * 1.6 + sparkle);
    col[i4] = 240; col[i4 + 1] = 248; col[i4 + 2] = 255; col[i4 + 3] = Math.round(a * 255);
    let dmin = 9;
    for (const s of seeds) dmin = Math.min(dmin, Math.hypot((u - s[0]) * (W / H), v - s[1]));
    const gv = Math.min(1, Math.max(0, dmin * 0.85 + (fbm(u * 5 + 2, v * 5, 4) - 0.5) * 0.3));
    const rad = Math.min(1, Math.hypot((u - 0.5) * (W / H), v - 0.5) / 1.02 + (fbm(u * 6 + 1, v * 6 + 4, 3) - 0.5) * 0.25);
    grow[i4] = Math.round(gv * 255); grow[i4 + 1] = Math.round(Math.max(0, rad) * 255); grow[i4 + 2] = Math.round(cloud * 255); grow[i4 + 3] = 255;
  }
  const mk = (d) => { const t = new THREE.DataTexture(d, W, H, THREE.RGBAFormat); t.flipY = false; t.needsUpdate = true; t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter; return t; };
  return {frost: mk(col), grow: mk(grow)};
};

const lensFrag = `
uniform sampler2D frost; uniform sampler2D grow; uniform float uProgress; uniform float uClear; uniform float uOpacity;
varying vec2 vUv;
void main(){
  vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 c = texture2D(frost, uv); vec3 gm = texture2D(grow, uv).rgb;
  float m = smoothstep(uProgress, uProgress - 0.12, gm.r);                 // frozen where growth < progress
  float front = smoothstep(0.06, 0.0, abs(gm.r - uProgress)) * step(0.01, uProgress) * (1.0 - step(1.0, uProgress));
  float keep = smoothstep(uClear - 0.1, uClear + 0.02, gm.g);              // melt opens from the centre outward
  float edge = smoothstep(0.25, 0.95, gm.g);
  float haze = m * (0.42 + 0.22 * gm.b + 0.3 * edge);
  float a = clamp(haze + c.a * m * 0.85 + front * 0.12, 0.0, 1.0) * keep * uOpacity;
  vec3 rgb = mix(vec3(0.78, 0.87, 0.95), vec3(1.0), clamp(c.a * 1.2 + front, 0.0, 1.0));
  gl_FragColor = vec4(rgb, a);
}`;
export const LensFrost = ({progress, clear, opacity = 1, seeds}) => {
  const {camera, size} = useThree();
  const {frost, grow} = useMemo(() => makeLensFrost(960, 540, seeds), [seeds]);
  const mat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {frost: {value: frost}, grow: {value: grow}, uProgress: {value: 0}, uClear: {value: 0}, uOpacity: {value: 1}},
    vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: lensFrag, transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
  }), [frost, grow]);
  mat.uniforms.uProgress.value = progress; mat.uniforms.uClear.value = clear; mat.uniforms.uOpacity.value = opacity;
  // a quad that exactly fills the view, 1 unit in front of the camera
  const hgt = 2 * Math.tan((camera.fov * Math.PI) / 360), wid = hgt * (size.width / size.height);
  const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
  const pos = camera.position.clone().add(dir);
  return (
    <mesh position={pos} quaternion={camera.quaternion} material={mat} renderOrder={20} frustumCulled={false}>
      <planeGeometry args={[wid, hgt]} />
    </mesh>
  );
};
