// PROTOTYPE — a regulation rink surface (200 × 85 ft, 1 unit = 1 ft) drawn with Canvas 2D:
// markings painted "under the ice" (softened, desaturated), then skate cuts in every direction,
// snow scuffs and a frosty sheen on top. The camera skates along the rink's long axis (−z).
import React, {useMemo} from 'react';
import * as THREE from 'three';

const rnd = (seed) => { const x = Math.sin(seed * 999.13) * 43758.5453; return x - Math.floor(x); };
const PX = 20;                 // px per ft
const LEN = 200, WID = 85;     // ft

const drawRink = () => {
  const c = document.createElement('canvas');
  c.width = WID * PX; c.height = LEN * PX;
  const g = c.getContext('2d');
  const X = (ft) => (ft + WID / 2) * PX;     // lateral: −42.5..42.5 ft
  const Y = (ft) => ft * PX;                 // along the rink: 0..200 ft
  // base ice
  const base = g.createLinearGradient(0, 0, c.width, 0);
  base.addColorStop(0, '#cfdbe3'); base.addColorStop(0.5, '#e4edf2'); base.addColorStop(1, '#cfdbe3');
  g.fillStyle = base; g.fillRect(0, 0, c.width, c.height);

  // ---- markings (under the ice) ----
  g.save();
  g.filter = 'blur(2.2px) saturate(0.7)';
  g.globalAlpha = 0.72;
  const RED = '#c23b44', BLUE = '#2f5fb3';
  const band = (y, h, col) => { g.fillStyle = col; g.fillRect(0, Y(y) - (h * PX) / 2, c.width, h * PX); };
  const circle = (x, y, r, col, w = 2 / 12) => { g.strokeStyle = col; g.lineWidth = w * PX; g.beginPath(); g.arc(X(x), Y(y), r * PX, 0, Math.PI * 2); g.stroke(); };
  const dot = (x, y, r, col) => { g.fillStyle = col; g.beginPath(); g.arc(X(x), Y(y), r * PX, 0, Math.PI * 2); g.fill(); };
  band(11, 2 / 12, RED); band(189, 2 / 12, RED);             // goal lines
  band(75, 1, BLUE); band(125, 1, BLUE);                     // blue lines
  band(100, 1, RED);                                         // centre red line
  circle(0, 100, 15, BLUE); dot(0, 100, 0.5, BLUE);          // centre circle + dot
  for (const ey of [31, 169]) {                              // end-zone faceoff circles, hash marks, dots
    for (const ex of [-22, 22]) {
      circle(ex, ey, 15, RED); dot(ex, ey, 1, RED);
      g.fillStyle = RED;
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
        g.fillRect(X(ex + sx * 2.9) - 1, Y(ey + sy * 15) - (sy > 0 ? 0 : 2 * PX), 2 / 12 * PX, 2 * PX);   // hash marks
      }
    }
  }
  for (const ny of [80, 120]) for (const nx of [-22, 22]) dot(nx, ny, 1, RED);   // neutral-zone dots
  // goal creases
  g.fillStyle = 'rgba(90,160,220,0.55)';
  for (const [cy, dir] of [[11, 1], [189, -1]]) { g.beginPath(); g.arc(X(0), Y(cy), 6 * PX, dir > 0 ? 0 : Math.PI, dir > 0 ? Math.PI : Math.PI * 2); g.fill(); }
  g.restore();

  // ---- skate cuts: curves in every direction, denser near circles and the middle ----
  g.save();
  g.lineCap = 'round';
  for (let i = 0; i < 2600; i++) {
    const r = (k) => rnd(i * 17 + k);
    const hot = r(1) < 0.35;                                  // bias some cuts toward faceoff circles
    const cx = hot ? [-22, 22, 0][Math.floor(r(2) * 3)] + (r(3) - 0.5) * 34 : (r(3) - 0.5) * WID;
    const cy = hot ? [31, 100, 169, 80, 120][Math.floor(r(4) * 5)] + (r(5) - 0.5) * 36 : r(5) * LEN;
    const len = 2 + r(6) * (r(7) > 0.85 ? 28 : 10);
    const ang = r(8) * Math.PI * 2, bend = (r(9) - 0.5) * 1.6;
    const x0 = X(cx), y0 = Y(cy);
    const x1 = x0 + Math.cos(ang) * len * PX, y1 = y0 + Math.sin(ang) * len * PX;
    const mx = (x0 + x1) / 2 - Math.sin(ang) * bend * len * PX * 0.5, my = (y0 + y1) / 2 + Math.cos(ang) * bend * len * PX * 0.5;
    const light = r(10) > 0.3;
    g.strokeStyle = light ? `rgba(255,255,255,${0.12 + r(11) * 0.35})` : `rgba(120,140,155,${0.08 + r(11) * 0.18})`;
    g.lineWidth = 0.6 + r(12) * 2.2;
    g.beginPath(); g.moveTo(x0, y0); g.quadraticCurveTo(mx, my, x1, y1); g.stroke();
    if (r(13) > 0.8) {                                         // paired edge cut (both blades)
      g.beginPath(); g.moveTo(x0 + 6, y0 + 6); g.quadraticCurveTo(mx + 6, my + 6, x1 + 6, y1 + 6); g.stroke();
    }
  }
  // snow scuffs (stops) and a frosty sheen
  for (let i = 0; i < 160; i++) {
    const r = (k) => rnd(i * 29 + 5000 + k);
    const x = X((r(1) - 0.5) * WID), y = Y(r(2) * LEN), rad = (1 + r(3) * 5) * PX;
    const grd = g.createRadialGradient(x, y, 0, x, y, rad);
    grd.addColorStop(0, `rgba(255,255,255,${0.25 + r(4) * 0.3})`); grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.save(); g.translate(x, y); g.rotate(r(5) * Math.PI); g.scale(1.8, 0.6); g.translate(-x, -y);
    g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.fill(); g.restore();
  }
  g.restore();
  return c;
};

let cached = null;
const rinkTexture = () => {
  if (cached) return cached;
  const t = new THREE.CanvasTexture(drawRink());
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 16; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
  cached = t;
  return t;
};

// Rink plane. Canvas top (ft 0) maps to the far end (−z). The camera sits at z = 16; camFt0 is the rink
// position (ft) under the camera at travel = 0, and skating forward moves the floor toward +z.
export const RinkFloor = ({travel, floorY = -1.55, camFt0 = 190}) => {
  const tex = useMemo(rinkTexture, []);
  const zc = 16 + LEN / 2 - camFt0 + travel;
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY, zc]}>
      <planeGeometry args={[WID, LEN]} />
      <meshStandardMaterial map={tex} color="#8ea2b2" roughness={0.42} metalness={0} />
    </mesh>
  );
};

// Speed streaks toward the camera (unchanged idea, kept here with the rink).
export const SpeedStreaks = ({travel, speed, floorY = -1.55}) => {
  const streaks = useMemo(() => Array.from({length: 60}).map((_, i) => ({
    x: (rnd(i + 700) - 0.5) * 34, y: floorY + 0.4 + rnd(i + 701) * 9, z0: -rnd(i + 702) * 90, len: 0.6 + rnd(i + 703) * 2.2,
    c: rnd(i + 704) > 0.55 ? '#ffffff' : '#5AA4D0',
  })).filter((s) => Math.abs(s.x) > 5 || s.y > 2), [floorY]);
  return streaks.map((s, i) => {
    const z = ((s.z0 + travel * 1.4) % 90 + 90) % 90 - 80;
    return (
      <mesh key={i} position={[s.x, s.y, z]}>
        <boxGeometry args={[0.03, 0.03, s.len * (0.3 + speed * 5)]} />
        <meshBasicMaterial color={s.c} toneMapped={false} transparent opacity={Math.min(1, 0.1 + speed)} />
      </mesh>
    );
  });
};
