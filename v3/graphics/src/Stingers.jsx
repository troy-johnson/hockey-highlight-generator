import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, cancelRender, continueRender, delayRender, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import * as THREE from 'three';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};
const fontCss = `@font-face {font-family: Inter; src: url('${staticFile('Inter.ttf')}'); font-weight: 100 900;}`;

// The face and its frost use one texture on the slab's shared outline.
function useFace(team, wipe, tokens) {
  const [texture, setTexture] = useState(null);
  const [handle] = useState(() => delayRender('Load stinger face'));
  useEffect(() => {
    let disposed = false;
    let result;
    async function load() {
      const font = new FontFace('StingerInter', `url('${staticFile('Inter.ttf')}')`, {weight: '900'});
      await font.load();
      document.fonts.add(font);
      let logo;
      if (team?.logo) {
        logo = new Image();
        logo.src = staticFile(team.logo);
        await logo.decode();
      }
      const canvas = document.createElement('canvas');
      canvas.width = 1536; canvas.height = 512;
      const c = canvas.getContext('2d');
      c.fillStyle = team?.primary ?? tokens.state; c.fillRect(0, 0, 1536, 512);
      c.fillStyle = team?.secondary ?? tokens.label; c.fillRect(0, 0, 100, 512); c.fillRect(1436, 0, 100, 512);
      const frost = c.createLinearGradient(0, 0, 0, 512);
      frost.addColorStop(0, 'rgba(240,250,255,.95)'); frost.addColorStop(.14, 'rgba(220,245,255,.05)');
      frost.addColorStop(.86, 'rgba(220,245,255,.05)'); frost.addColorStop(1, 'rgba(240,250,255,.9)');
      c.fillStyle = frost; c.fillRect(0, 0, 1536, 512);
      c.fillStyle = '#FFFFFF'; c.textAlign = 'center'; c.textBaseline = 'middle';
      if (logo) {
        const size = wipe ? 370 : 250;
        const ratio = Math.min(size / logo.width, size / logo.height);
        c.drawImage(logo, (wipe ? 768 : 310) - logo.width * ratio / 2,
          256 - logo.height * ratio / 2, logo.width * ratio, logo.height * ratio);
      }
      if (!wipe || !logo) {
        const x = logo ? 920 : 768;
        const title = team?.name?.replace(/\s+/g, '').toUpperCase() ?? 'HOCKEY';
        c.font = '900 148px StingerInter';
        c.fillText(title, x, 225, logo ? 890 : 1220);
        if (!wipe) {
          c.font = '900 62px StingerInter'; c.fillStyle = '#DDEFF9';
          c.fillText(team ? 'HOCKEY' : 'GAME RECAP', x, 356);
        }
      }
      // Deterministic frost remains attached to the face during every transform.
      for (let i = 0; i < 320; i++) {
        const x = (i * 443) % 1536;
        const y = i % 2 ? (i * 13) % 48 : 512 - (i * 17) % 48;
        c.fillStyle = `rgba(255,255,255,${.12 + i % 4 * .1})`;
        c.fillRect(x, y, 2 + i % 5, 2 + i % 3);
      }
      result = new THREE.CanvasTexture(canvas); result.colorSpace = THREE.SRGBColorSpace;
      if (!disposed) setTexture(result);
      else result.dispose();
    }
    load().catch(cancelRender);
    return () => {disposed = true; result?.dispose();};
  }, [handle, team, wipe, tokens]);
  useEffect(() => {if (texture) continueRender(handle);}, [texture, handle]);
  return texture;
}

function slabShape() {
  const shape = new THREE.Shape();
  shape.moveTo(-6, -2); shape.lineTo(5, -2); shape.lineTo(6, 2); shape.lineTo(-5, 2); shape.closePath();
  return shape;
}

function Slab3D({team, tokens, texture, position, rotation, scale = 1}) {
  const [body, face] = useMemo(() => {
    const shape = slabShape();
    const body = new THREE.ExtrudeGeometry(shape, {depth: .55, bevelEnabled: true,
      bevelThickness: .09, bevelSize: .08, bevelSegments: 3, steps: 1});
    const face = new THREE.ShapeGeometry(shape);
    const positions = face.attributes.position; const uv = face.attributes.uv;
    for (let i = 0; i < positions.count; i++) uv.setXY(i, (positions.getX(i) + 6) / 12, (positions.getY(i) + 2) / 4);
    return [body, face];
  }, []);
  useEffect(() => () => {body.dispose(); face.dispose();}, [body, face]);
  return <group {...{position, rotation, scale}}>
    <mesh geometry={body}><meshStandardMaterial color={team?.secondary ?? tokens.label} metalness={.4} roughness={.42}/></mesh>
    {texture && <mesh geometry={face} position={[0, 0, .66]}>
      <meshStandardMaterial map={texture} roughness={.7} metalness={.08}/></mesh>}
  </group>;
}

function roundedShape(width, height, radius) {
  const shape = new THREE.Shape();
  shape.moveTo(-width / 2 + radius, -height / 2); shape.lineTo(width / 2 - radius, -height / 2);
  shape.quadraticCurveTo(width / 2, -height / 2, width / 2, -height / 2 + radius);
  shape.lineTo(width / 2, height / 2 - radius);
  shape.quadraticCurveTo(width / 2, height / 2, width / 2 - radius, height / 2);
  shape.lineTo(-width / 2 + radius, height / 2);
  shape.quadraticCurveTo(-width / 2, height / 2, -width / 2, height / 2 - radius);
  shape.lineTo(-width / 2, -height / 2 + radius);
  shape.quadraticCurveTo(-width / 2, -height / 2, -width / 2 + radius, -height / 2);
  return shape;
}

function Rink() {
  const resources = useMemo(() => {
    // One scene unit is ten feet: 200 × 85 feet, with 28-foot corners.
    const ice = new THREE.ShapeGeometry(roundedShape(20, 8.5, 2.8));
    const outline = roundedShape(20.4, 8.9, 3);
    outline.holes.push(new THREE.Path(roundedShape(20, 8.5, 2.8).getPoints(64)));
    const boards = new THREE.ExtrudeGeometry(outline, {depth: .42, bevelEnabled: false});
    const canvas = document.createElement('canvas'); canvas.width = 2000; canvas.height = 850;
    const c = canvas.getContext('2d'); c.fillStyle = '#E4F0F3'; c.fillRect(0, 0, 2000, 850);
    const line = (x, color, width) => {c.strokeStyle = color; c.lineWidth = width;
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, 850); c.stroke();};
    line(1000, '#D64455', 10); line(750, '#4089C4', 12); line(1250, '#4089C4', 12);
    line(110, '#D64455', 3); line(1890, '#D64455', 3);
    const circle = (x, y, r, color) => {c.strokeStyle = color; c.lineWidth = 3;
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke();};
    circle(1000, 425, 150, '#4089C4');
    for (const x of [310, 1690]) for (const y of [205, 645]) {
      circle(x, y, 150, '#D64455'); c.fillStyle = '#D64455'; c.beginPath(); c.arc(x, y, 6, 0, Math.PI * 2); c.fill();
    }
    for (const x of [110, 1890]) {
      c.fillStyle = '#91C6E1'; c.beginPath(); c.arc(x, 425, 60, x < 1000 ? -Math.PI / 2 : Math.PI / 2,
        x < 1000 ? Math.PI / 2 : Math.PI * 1.5); c.fill();
    }
    const uv = ice.attributes.uv; const positions = ice.attributes.position;
    for (let i = 0; i < positions.count; i++) uv.setXY(i, (positions.getX(i) + 10) / 20, (positions.getY(i) + 4.25) / 8.5);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    return {ice, boards, texture};
  }, []);
  useEffect(() => () => Object.values(resources).forEach(r => r.dispose()), [resources]);
  return <group rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
    <mesh geometry={resources.ice}><meshStandardMaterial map={resources.texture} roughness={.28}/></mesh>
    <mesh geometry={resources.boards}><meshStandardMaterial color="#F5F8FB" roughness={.55}/></mesh>
  </group>;
}

// Screen-space snow and mist cannot intersect scene geometry or expose depth edges.
function Snow({wipe, frame: f}) {
  const strength = wipe ? interpolate(f, [0, 8, 19, 29], [0, 1, 1, 0], clamp) :
    interpolate(f, [0, 20, 32, 74], [0, 0, 1, 0], clamp);
  return <AbsoluteFill style={{overflow: 'hidden', pointerEvents: 'none', opacity: strength}}>
    <AbsoluteFill style={{background: wipe ? 'rgba(231,246,253,.88)' : 'transparent',
      filter: 'blur(16px)'}}/>
    {Array.from({length: 72}, (_, i) => {
      const x = ((i * 347 + f * (wipe ? 90 : 35)) % 2600) - 340;
      const y = ((i * 191 + f * (i % 3 - 1) * 7) % 1300) - 110;
      return <div key={i} style={{position: 'absolute', left: x, top: y,
        width: 60 + i % 7 * 20, height: 9 + i % 4 * 5, borderRadius: '50%',
        background: '#F1FBFF', opacity: .25 + i % 3 * .1,
        filter: `blur(${8 + i % 4 * 4}px)`, transform: 'rotate(-12deg)'}}/>;
    })}
    <div style={{position: 'absolute', left: '8%', top: '48%', width: '86%', height: '24%',
      background: 'radial-gradient(ellipse,rgba(235,250,255,.65),transparent 70%)', filter: 'blur(35px)'}}/>
  </AbsoluteFill>;
}

export function Stinger({data, teams, tokens, duration, wipe = false}) {
  const frame = useCurrentFrame();
  const nominal = wipe ? 30 : 75;
  const f = frame * (nominal - 1) / Math.max(1, duration - 1);
  const team = data.team ? teams[data.team] : null;
  const texture = useFace(team, wipe, tokens);
  const enter = interpolate(f, wipe ? [0, 10, 19, 29] : [0, 27, 38, 74],
    wipe ? [-18, 0, 0, 18] : [-19, .6, 0, 0], clamp);
  const yaw = interpolate(f, wipe ? [0, 10, 19, 29] : [0, 22, 34, 74],
    wipe ? [-.6, -.08, .08, .6] : [.05, .45, -.06, -.06], clamp);
  const opacity = interpolate(f, [0, 3, nominal - 7, nominal - 1], [0, 1, 1, 0], clamp);
  return <AbsoluteFill style={{opacity, background: wipe ? 'transparent' :
    'radial-gradient(ellipse at 50% 30%,#24455B,#07131F 75%)'}}>
    <style>{fontCss}</style>
    {/* The lens snow passes behind the crest and in front of the rink. */}
    {wipe && <Snow wipe frame={f}/>}
    {texture && <ThreeCanvas width={1920} height={1080} camera={{position: [0, 8, 22], fov: 40}}
      gl={{alpha: true, antialias: true}} style={{position: 'absolute', inset: 0}}>
      <ambientLight intensity={1.4}/><directionalLight position={[-6, 12, 14]} intensity={3}/>
      <pointLight position={[7, 3, 5]} intensity={55} color="#91D3FA"/>
      {!wipe && <Rink/>}
      <Slab3D team={team} tokens={tokens} texture={texture} position={[enter, wipe ? 2 : 1.7, 2]}
        rotation={[-.08, yaw, -.035]} scale={wipe ? .66 : 1}/>
    </ThreeCanvas>}
    {!wipe && <Snow frame={f}/>}
  </AbsoluteFill>;
}

export const OpenStinger = props => <Stinger data={{team: props.focusSide}} teams={props.teams}
  tokens={props.tokens} duration={props.durationFrames}/>;
export const PeriodWipe = props => <Stinger data={{team: props.focusSide}} teams={props.teams}
  tokens={props.tokens} duration={props.durationFrames} wipe/>;
