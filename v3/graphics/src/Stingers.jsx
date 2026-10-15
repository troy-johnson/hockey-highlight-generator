import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, cancelRender, continueRender, delayRender, Easing, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {Reflector} from 'three/addons/objects/Reflector.js';
import {BladeCut, ContactShadow, IceBloom, IceSpray, openingMotion, SnowFlurries} from './StingerEffects.jsx';
import {FrostPane} from './FrostPane.jsx';
import {RegionalMatchup} from './IcePakGraphics';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};
const fontCss = `@font-face {font-family: Inter; src: url('${staticFile('Inter.ttf')}'); font-weight: 100 900;}`;

// The face and its frost use one texture on the slab's shared outline.
function useFace(team, wipe, tokens) {
  const [texture, setTexture] = useState(null);
  const [handle] = useState(() => delayRender('Load stinger face'));
  useEffect(() => {
    if (wipe) {continueRender(handle); return;}
    let disposed = false;
    let result;
    async function load() {
      const font = new FontFace('StingerInter', `url('${staticFile('Inter.ttf')}')`, {weight: '900'});
      await font.load();
      document.fonts.add(font);
      let logo;
      const faceAsset = wipe ? (team?.logo || team?.wordmark) : (team?.lockup || team?.wordmark || team?.logo);
      if (faceAsset) {
        logo = new Image();
        logo.src = staticFile(faceAsset);
        await logo.decode();
      }
      const canvas = document.createElement('canvas');
      canvas.width = 1536; canvas.height = 512;
      const c = canvas.getContext('2d');
      const snapshot = () => {
        const copy = document.createElement('canvas'); copy.width = 1536; copy.height = 512;
        copy.getContext('2d').drawImage(canvas, 0, 0);
        const texture = new THREE.CanvasTexture(copy); texture.colorSpace = THREE.SRGBColorSpace;
        return texture;
      };
      c.fillStyle = team?.primary ?? tokens.state; c.fillRect(0, 0, 1536, 512);
      c.fillStyle = team?.secondary ?? tokens.label; c.fillRect(0, 0, 100, 512); c.fillRect(1436, 0, 100, 512);
      const paint = snapshot();
      c.clearRect(0, 0, 1536, 512);
      c.fillStyle = '#FFFFFF'; c.textAlign = 'center'; c.textBaseline = 'middle';
      if (logo) {
        const ratio = Math.min((wipe ? 370 : 1200) / logo.width, (wipe ? 370 : 330) / logo.height);
        c.drawImage(logo, 768 - logo.width * ratio / 2,
          256 - logo.height * ratio / 2, logo.width * ratio, logo.height * ratio);
      }
      if (!logo) {
        const x = logo ? 920 : 768;
        const title = team?.name?.replace(/\s+/g, '').toUpperCase() ?? 'HOCKEY';
        c.font = '900 148px StingerInter';
        c.fillText(title, x, 225, logo ? 890 : 1220);
        if (!wipe) {
          c.font = '900 62px StingerInter'; c.fillStyle = '#DDEFF9';
          c.fillText(team ? 'HOCKEY' : 'GAME RECAP', x, 356);
        }
      }
      const brand = snapshot();
      c.clearRect(0, 0, 1536, 512);
      for (let i = 0; i < 190; i++) {
        const n = k => {const v = Math.sin(i * 97.3 + k * 173.1) * 43758.5453; return v - Math.floor(v);};
        const x = n(1) * 1536, y = n(2) * 512, length = 80 + n(3) * 650;
        c.lineWidth = .5 + n(4) * 2; c.strokeStyle = `rgba(230,245,255,${.12 + n(5) * .35})`;
        c.beginPath(); c.moveTo(x, y); c.lineTo(x + length, y + (n(6) - .5) * 18); c.stroke();
      }
      // Crystal clusters grow inward from the painted edges.
      for (let i = 0; i < 1600; i++) {
        const r = k => {const n = Math.sin(i * 127.1 + k * 311.7) * 43758.5453; return n - Math.floor(n);};
        const x = r(1) * 1536;
        const depth = r(2) ** 2 * 170;
        const y = i % 2 ? depth : 512 - depth;
        const angle = r(3) * Math.PI * 2, length = 5 + r(4) * 18;
        c.strokeStyle = `rgba(235,250,255,${.08 + r(5) * .25})`;
        c.lineWidth = .4 + r(6) * .6; c.beginPath(); c.moveTo(x, y);
        c.lineTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length);
        for (let branch = 1; branch <= 4; branch++) {
          const t = branch / 5, bx = x + Math.cos(angle) * length * t, by = y + Math.sin(angle) * length * t;
          for (const side of [-1, 1]) {
            const branchLength = length * (1 - t) * .4;
            c.moveTo(bx, by); c.lineTo(bx + Math.cos(angle + side) * branchLength,
              by + Math.sin(angle + side) * branchLength);
          }
        }
        c.stroke();
      }
      const frost = snapshot();
      result = {paint, brand, frost, dispose: () => [paint, brand, frost].forEach(t => t.dispose())};
      if (!disposed) setTexture(result);
      else result.dispose();
    }
    load().catch(cancelRender);
    return () => {disposed = true; result?.dispose();};
  }, [handle, team, wipe, tokens]);
  useEffect(() => {if (texture) continueRender(handle);}, [texture, handle]);
  return texture;
}

const HASH = `float h21(vec2 p) {vec3 q = fract(vec3(p.xyx) * .1031); q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);}
float noise(vec2 p) {vec2 cell = floor(p), t = fract(p); t = t*t*(3.0-2.0*t);
  return mix(mix(h21(cell),h21(cell+vec2(1,0)),t.x), mix(h21(cell+vec2(0,1)),h21(cell+vec2(1,1)),t.x),t.y);}`;
const VERTEX = `varying vec2 vUv; void main() {vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);}`;

function slabShape() {
  const shape = new THREE.Shape();
  shape.moveTo(-5, 1.45); shape.lineTo(5, 1.45); shape.lineTo(5, -1.45); shape.lineTo(-5, -1.45); shape.closePath();
  return shape;
}

function Slab3D({team, tokens, texture, position, rotation, frame: f, scale = 1, sweep = 0}) {
  const [body, face, strip, rim, brandFace] = useMemo(() => {
    const shape = slabShape();
    const body = new THREE.ExtrudeGeometry(shape, {depth: .55, bevelEnabled: true,
      bevelThickness: .09, bevelSize: .08, bevelSegments: 3, steps: 1});
    const face = new THREE.ShapeGeometry(shape);
    const positions = face.attributes.position; const uv = face.attributes.uv;
    for (let i = 0; i < positions.count; i++) uv.setXY(i, (positions.getX(i) + 5) / 10, (positions.getY(i) + 1.45) / 2.9);
    return [body, face, new THREE.BoxGeometry(10.1, .16, .12), new THREE.EdgesGeometry(body, 35), new THREE.PlaneGeometry(10, 10 / 3)];
  }, []);
  useEffect(() => () => {body.dispose(); face.dispose(); strip.dispose(); rim.dispose(); brandFace.dispose();}, [body, face, strip, rim, brandFace]);
  const reveal = interpolate(f, [43, 62], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const frost = interpolate(f, [24, 46], [0, 1.05], clamp);
  const breakthrough = interpolate(f, [44, 67], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const glow = f >= 26 ? Math.exp(-(f - 26) / 8) : 0;
  const primary = team?.primary ?? tokens.state;
  return <group {...{position, rotation, scale}}>
    <mesh geometry={body}><meshPhysicalMaterial color="#19212A" metalness={.25} roughness={.38}
      clearcoat={.7} clearcoatRoughness={.2}/></mesh>
    <mesh geometry={strip} position={[0, -1.41, .64]}>
      <meshPhysicalMaterial color={primary} emissive={primary} emissiveIntensity={.35 + glow * .9}
        metalness={.2} roughness={.3} clearcoat={.6}/></mesh>
    <lineSegments geometry={rim}><lineBasicMaterial color={new THREE.Color('#CFEFFF').multiplyScalar(1 + glow * 2)}
      transparent opacity={.14 + glow * .4}/></lineSegments>
    <mesh geometry={face} position={[0, 0, .66]}>
      <meshPhysicalMaterial color="#141B23" roughness={.32} metalness={.2} clearcoat={.8} clearcoatRoughness={.15}/></mesh>
    {texture && <mesh geometry={brandFace} position={[0, .06, .7]} scale={1.04 - .04 * reveal} renderOrder={3}>
      <shaderMaterial transparent depthWrite={false} toneMapped={false}
        uniforms={{map: {value: texture.brand}, opacity: {value: reveal}, sweep: {value: sweep}}}
        vertexShader={VERTEX}
        fragmentShader={`uniform sampler2D map; uniform float opacity; uniform float sweep; varying vec2 vUv;
          void main() {vec4 c = texture2D(map, vUv); vec2 o = vec2(.0016, .0048);
            float lit = clamp(c.a - texture2D(map, vUv + vec2(o.x, -o.y)).a, 0.0, 1.0);
            float shade = clamp(c.a - texture2D(map, vUv + vec2(-o.x, o.y)).a, 0.0, 1.0);
            float band = exp(-pow((vUv.x + (1.0 - vUv.y) * .35 - (sweep * 1.7 - .35)) * 7.0, 2.0)) * step(.001, sweep);
            vec3 col = c.rgb * (1.0 - shade * .45) + vec3(.85, .95, 1.0) * (lit * .55 + band * c.a * .55);
            gl_FragColor = vec4(col, c.a * opacity);
            #include <colorspace_fragment>
          }`}/></mesh>}
    {texture && <mesh geometry={face} position={[0, 0, .715]} renderOrder={3.5}>
      <shaderMaterial transparent depthWrite={false} toneMapped={false} uniforms={{map: {value: texture.frost}, growth: {value: frost}, breakthrough: {value: breakthrough}}}
        vertexShader={VERTEX}
        fragmentShader={`uniform sampler2D map; uniform float growth; uniform float breakthrough; varying vec2 vUv;
          ${HASH}
          void main() {vec4 color = texture2D(map, vUv);
            vec2 a = vUv * vec2(3.448, 1.0);
            float edge = min(min(vUv.x, 1.0-vUv.x)*3.448, min(vUv.y, 1.0-vUv.y));
            float contact = length(a) / 3.59;
            float field = max(0.0, contact + (noise(vUv*vec2(19,7))-.5)*.12 + (noise(vUv*vec2(77,29))-.5)*.035);
            float frozen = 1.0 - smoothstep(growth-.02, growth+.02, field);
            float grain = noise(vUv*vec2(310,110));
            float needles = smoothstep(.62, .95, noise(vec2(atan(a.y, a.x) * 60.0, length(a) * 6.0)));
            float front = (1.0-smoothstep(.0,.014,abs(field-growth))) * smoothstep(0.0,.05,growth) * (1.0-smoothstep(1.0,1.05,growth)) * smoothstep(.2,.7,grain);
            vec2 p = (vUv-.5)*vec2(3.448,1.0);
            float fracture = length(p) + .055*sin(atan(p.y,p.x)*9.0) + (noise(vUv*vec2(45,17))-.5)*.055;
            float reach = breakthrough*1.94;
            float cleared = (1.0-smoothstep(reach-.035,reach+.035,fracture))*smoothstep(0.0,.08,breakthrough);
            float crack = (1.0-smoothstep(.0,.022,abs(fracture-reach)))*smoothstep(.0,.15,breakthrough)*(1.0-smoothstep(.8,1.0,breakthrough));
            float edgeResidue = 1.0-smoothstep(.035,.11,edge);
            float coat = max(color.a*.6, .3 + grain*.12 + needles*.18)*frozen;
            float alpha = (coat*(1.0-cleared*.97)+front*.22+crack*.22)*smoothstep(0.0,.025,growth);
            alpha = max(alpha, color.a*edgeResidue*.45*smoothstep(.5,1.0,breakthrough));
            gl_FragColor = vec4(mix(vec3(.68,.8,.88),vec3(.98,1.08,1.12),front+crack*.5+needles*.2),clamp(alpha,0.0,.78));
          }`}/></mesh>}
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

function CameraRig({frame: f, drift, push = 0, vertical = false}) {
  const {camera} = useThree();
  const shake = f >= 26 ? Math.exp(-(f - 26) / 5) * .06 : 0;
  const settle = interpolate(f, [0, 26], [1, 0], {...clamp, easing: Easing.out(Easing.quad)});
  const base = (vertical ? 36 : 16) - 1.2 * drift;
  // Exit: dolly into the lockup so the cut lands inside the brand.
  camera.position.set(Math.sin(f * 2.1) * shake - settle * 1.5, (.6 + .4 * (1 - settle)) * (1 - push) + Math.cos(f * 2.7) * shake,
    base - push * (base - 1.6));
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  return null;
}

function Rink({travel}) {
  const {gl} = useThree();
  const resources = useMemo(() => {
    // One scene unit is one foot, as in the approved skating scene.
    const ice = new THREE.ShapeGeometry(roundedShape(200, 85, 28));
    const outline = roundedShape(200.4, 85.4, 28.2);
    outline.holes.push(new THREE.Path(roundedShape(200, 85, 28).getPoints(64)));
    const boards = new THREE.ExtrudeGeometry(outline, {depth: .42, bevelEnabled: false});
    const canvas = document.createElement('canvas'); canvas.width = 4000; canvas.height = 1700;
    const c = canvas.getContext('2d'); c.scale(2, 2); c.fillStyle = '#E4F0F3'; c.fillRect(0, 0, 2000, 850);
    c.save(); c.filter = 'blur(1.1px) saturate(.7)'; c.globalAlpha = .72;
    const line = (x, color, width) => {c.strokeStyle = color; c.lineWidth = width;
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, 850); c.stroke();};
    line(1000, '#D64455', 10); line(750, '#4089C4', 12); line(1250, '#4089C4', 12);
    line(110, '#D64455', 3); line(1890, '#D64455', 3);
    const circle = (x, y, r, color) => {c.strokeStyle = color; c.lineWidth = 3;
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke();};
    circle(1000, 425, 150, '#4089C4');
    for (const x of [310, 1690]) for (const y of [205, 645]) {
      circle(x, y, 150, '#D64455'); c.fillStyle = '#D64455'; c.beginPath(); c.arc(x, y, 6, 0, Math.PI * 2); c.fill();
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) c.fillRect(x + sx * 150, y + sy * 29, 20, 2);
    }
    for (const x of [110, 1890]) {
      c.fillStyle = '#91C6E1'; c.beginPath(); c.arc(x, 425, 60, x < 1000 ? -Math.PI / 2 : Math.PI / 2,
        x < 1000 ? Math.PI / 2 : Math.PI * 1.5); c.fill();
    }
    c.restore(); c.lineCap = 'round';
    const surface = document.createElement('canvas'); surface.width = 4000; surface.height = 1700;
    const s = surface.getContext('2d'); s.scale(2, 2); s.fillStyle = '#AAAAAA'; s.fillRect(0, 0, 2000, 850);
    for (let i = 0; i < 3600; i++) {
      const r = k => {const n = Math.sin(i*127.1 + k*311.7)*43758.5453; return n-Math.floor(n);};
      const x = r(1)*2000, y = r(2)*850, angle = r(3)*Math.PI*2, length = 20 + r(4)*(r(10)>.85 ? 220 : 90);
      const endX = x + Math.cos(angle)*length, endY = y + Math.sin(angle)*length;
      const bend = (r(5)-.5)*length*.8;
      c.strokeStyle = r(6) > .2 ? `rgba(255,255,255,${.08+r(7)*.18})` : `rgba(120,140,155,${.03+r(7)*.06})`;
      c.lineWidth = .2+r(8)*.45;
      s.strokeStyle = `rgba(50,50,50,${.12+r(7)*.22})`; s.lineWidth = .35+r(8)*.65;
      for (const offset of r(9) > .75 ? [0, 3] : [0]) {
        c.beginPath(); c.moveTo(x+offset,y+offset);
        c.quadraticCurveTo((x+endX)/2-Math.sin(angle)*bend+offset,
          (y+endY)/2+Math.cos(angle)*bend+offset,endX+offset,endY+offset); c.stroke();
        s.beginPath(); s.moveTo(x+offset,y+offset);
        s.quadraticCurveTo((x+endX)/2-Math.sin(angle)*bend+offset,
          (y+endY)/2+Math.cos(angle)*bend+offset,endX+offset,endY+offset); s.stroke();
      }
    }
    for (let i = 0; i < 160; i++) {
      const r = k => {const n=Math.sin(i*173.1+k*79.7)*43758.5453;return n-Math.floor(n);};
      const x=r(1)*2000,y=r(2)*850,radius=15+r(3)*45;
      const scuff=c.createRadialGradient(x,y,0,x,y,radius);
      scuff.addColorStop(0,'rgba(255,255,255,.3)'); scuff.addColorStop(1,'rgba(255,255,255,0)');
      c.fillStyle=scuff;c.save();c.translate(x,y);c.rotate(i*2.4);c.scale(1.8,.55);
      c.beginPath();c.arc(0,0,radius,0,Math.PI*2);c.fill();c.restore();
    }
    const uv = ice.attributes.uv; const positions = ice.attributes.position;
    for (let i = 0; i < positions.count; i++) uv.setXY(i, (positions.getX(i) + 100) / 200, (positions.getY(i) + 42.5) / 85);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = gl.capabilities.getMaxAnisotropy();
    const surfaceMap = new THREE.CanvasTexture(surface); surfaceMap.anisotropy = texture.anisotropy;
    const roughCanvas = document.createElement('canvas'); roughCanvas.width = surface.width; roughCanvas.height = surface.height;
    const roughContext = roughCanvas.getContext('2d'); roughContext.filter = 'invert(1)'; roughContext.drawImage(surface, 0, 0);
    const roughnessMap = new THREE.CanvasTexture(roughCanvas); roughnessMap.anisotropy = texture.anisotropy;
    const shader = {
      uniforms: {color: {value: new THREE.Color('#9DB4C4')}, tDiffuse: {value: null}, textureMatrix: {value: null}},
      vertexShader: Reflector.ReflectorShader.vertexShader,
      fragmentShader: `uniform sampler2D tDiffuse; varying vec4 vUv;
        void main() {vec2 p=vUv.xy/vUv.w; float grazing=smoothstep(.15,.85,p.y); vec2 d=vec2(.0015,.0025)*(1.0+grazing*1.5);
          vec3 reflection=texture2D(tDiffuse,p).rgb*.4;
          reflection+=(texture2D(tDiffuse,p+d).rgb+texture2D(tDiffuse,p-d).rgb
            +texture2D(tDiffuse,p+vec2(d.x,-d.y)).rgb+texture2D(tDiffuse,p+vec2(-d.x,d.y)).rgb)*.15;
          gl_FragColor=vec4(reflection,.06+.12*grazing);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    };
    const reflection = new Reflector(ice, {textureWidth: 1024, textureHeight: 512, clipBias: .003,
      multisample: 0, shader});
    reflection.position.z = .007; reflection.material.transparent = true; reflection.material.depthWrite = false;
    reflection.renderOrder = 1;
    return {ice, boards, texture, surfaceMap, roughnessMap, reflection};
  }, [gl]);
  useEffect(() => () => Object.values(resources).forEach(r => r.dispose()), [resources]);
  return <group rotation={[-Math.PI / 2, 0, Math.PI / 2]} position={[0, -1.55, -74 + travel]}>
    <mesh geometry={resources.ice}><meshPhysicalMaterial map={resources.texture} color="#9AABB7"
      bumpMap={resources.surfaceMap} bumpScale={.025} roughnessMap={resources.roughnessMap} roughness={.72}
      clearcoat={.35} clearcoatRoughness={.28}/></mesh>
    <primitive object={resources.reflection}/>
    <mesh geometry={resources.boards}><meshStandardMaterial color="#F5F8FB" roughness={.55}/></mesh>
  </group>;
}

function MatchupRail({data, teams, team, opponent, f, vertical, accent = '#7FD3F0'}) {
  // One graphite plate in the Black Ice family: focus team, VS, opponent, date.
  const rise = i => interpolate(f, [62 + i * 3, 76 + i * 3], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const line = interpolate(f, [62, 80], [0, 1], {...clamp, easing: Easing.out(Easing.cubic)});
  const first = data.team ? team : teams.home;
  const second = data.team ? opponent : teams.away;
  const cell = (t, i, key) => t && <div key={key} style={{display: 'flex', alignItems: 'center', gap: 18,
    height: vertical ? 92 : 84, padding: '0 28px', background: '#19212A', borderBottom: `3px solid ${t.line ?? t.primary}`,
    opacity: rise(i), transform: `translateY(${(1 - rise(i)) * 24}px)`}}>
    {t.logo && <Img src={staticFile(t.logo)} style={{height: vertical ? 60 : 54, width: vertical ? 60 : 54,
      objectFit: 'contain'}}/>}
    <span style={{fontSize: vertical ? 40 : 36, fontWeight: 750, color: '#EEF3F6', whiteSpace: 'nowrap'}}>{t.name}</span>
  </div>;
  const vs = <div key="vs" style={{display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: vertical ? 56 : 84, padding: '0 22px', fontSize: 22, fontWeight: 700, letterSpacing: 4,
    color: '#9FB2C1', background: '#0E1319', opacity: rise(1), transform: `translateY(${(1 - rise(1)) * 24}px)`}}>VS</div>;
  const date = data.date && <div key="date" style={{display: 'flex', alignItems: 'center', justifyContent: 'center',
    height: vertical ? 56 : 84, padding: '0 26px', fontSize: 24, fontWeight: 700, letterSpacing: 2, color: '#9FB2C1',
    background: '#0E1319', borderBottom: `2px solid ${accent}`, opacity: rise(3),
    transform: `translateY(${(1 - rise(3)) * 24}px)`}}>{data.date}</div>;
  return <div style={{position: 'absolute', fontFamily: 'Inter, Arial, sans-serif',
    ...(vertical ? {left: 120, width: 840, bottom: 430} : {left: 0, right: 0, top: 800}),
    display: 'flex', justifyContent: 'center'}}>
    <div style={{position: 'relative', display: 'flex', flexDirection: vertical ? 'column' : 'row',
      alignItems: 'stretch', width: vertical ? '100%' : 'auto', opacity: rise(0),
      background: 'linear-gradient(180deg,#19212A,#0E1319)', boxShadow: '0 18px 40px rgba(0,0,0,.45)'}}>
      <div style={{position: 'absolute', left: 0, top: -1, height: 1, width: `${line * 100}%`, background: accent}}/>
      {[cell(first, 0, 'first'), vs, cell(second, 2, 'second'), date]}
    </div></div>;
}

export function Stinger({data, teams, tokens, duration, wipe = false}) {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const vertical = height > width;
  const nominal = wipe ? 30 : 135;
  const f = frame * (nominal - 1) / Math.max(1, duration - 1);
  const team = data.team ? teams[data.team] : null;
  const opponent = teams[data.team === 'away' ? 'home' : 'away'];
  const texture = useFace(team, wipe, tokens);
  if (wipe) return <FrostPane vertical={vertical} team={team}/>;
  const {enter, yaw, lean, travel, drift, y, push} = openingMotion(f);
  // Full-length openings exit through the lockup and a short ice flash;
  // legacy short openings fade out over their last frames.
  const opacity = duration < nominal
    ? interpolate(f, [0, 6, nominal - 10, nominal - 1], [0, 1, 1, 0], clamp)
    : interpolate(f, [0, 6], [0, 1], clamp);
  const sweep = interpolate(f, [60, 70], [-7, 7], {...clamp, easing: Easing.inOut(Easing.quad)});
  const sweepOn = interpolate(f, [58, 61, 69, 72], [0, 1, 1, 0], clamp);
  return <AbsoluteFill style={{zIndex: 30, background: '#04070D'}}>
    <style>{fontCss}</style>
    <AbsoluteFill style={{opacity}}>
    {texture && <ThreeCanvas width={width} height={height} dpr={2}
      camera={{position: vertical ? [0, 1, 36] : [0, 1, 16], fov: 32}}
      gl={{alpha: true, antialias: true}} style={{position: 'absolute', inset: 0}}>
      <CameraRig frame={f} drift={drift} push={duration < nominal ? 0 : push} vertical={vertical}/>
      <color attach="background" args={['#04070D']}/>
      <fog attach="fog" args={['#04070D', vertical ? 30 : 12, vertical ? 70 : 48]}/>
      <ambientLight intensity={interpolate(f, [0, 8], [.08, .3], clamp)}/>
      <directionalLight position={[-6, 8, 10]} intensity={1.1}/>
      <spotLight position={[0, 14, -4]} angle={.5} penumbra={.8} intensity={260} distance={40} color="#DFF1FF"/>
      <pointLight position={[sweep, 2, 2.4]} intensity={60 * sweepOn} distance={8} color="#FFFFFF"/>
      <pointLight position={[0, -3, -6]} intensity={80} distance={20} color="#5AA4D0"/>
      <pointLight position={[-5, 3, 5]} intensity={18} distance={10} color="#CFEFFF"/>
      <Rink travel={travel}/>
      <BladeCut frame={f} travel={travel}/>
      <ContactShadow position={[enter, y, 0]} yaw={yaw} lean={lean}/>
      <Slab3D team={team} tokens={tokens} texture={texture} frame={f} position={[enter, y, 0]}
        rotation={[0, yaw, lean]} sweep={sweep * sweepOn}/>
      <SnowFlurries frame={f}/><IceSpray frame={f}/><IceBloom frame={f}/>
    </ThreeCanvas>}
    <AbsoluteFill style={{pointerEvents: 'none',
      background: 'radial-gradient(ellipse at 50% 45%, transparent 32%, rgba(0,0,0,.65) 100%)'}}/>
    <AbsoluteFill style={{pointerEvents: 'none', overflow: 'hidden',
      opacity: interpolate(f, [0, 12, 26, 40], [0, .35, .1, 0], clamp)}}>
      {Array.from({length: 16}, (_, i) => <div key={i} style={{position: 'absolute',
        left: ((i * 347 - travel * 80) % 2600 + 2600) % 2600 - 340,
        top: (vertical ? 260 : 160) + i * (vertical ? 70 : 52), width: 200 + i % 5 * 100, height: 1 + i % 2,
        background: 'linear-gradient(90deg,transparent,rgba(170,220,255,.5),transparent)', filter: 'blur(2px)'}}/>)}
    </AbsoluteFill>
    <AbsoluteFill style={{opacity: interpolate(f, [112, 122], [1, 0], clamp),
      transform: `translateY(${interpolate(f, [112, 122], [0, 30], {...clamp, easing: Easing.in(Easing.cubic)})}px)`}}>
      <MatchupRail data={data} teams={teams} team={team} opponent={opponent} f={f} vertical={vertical}
        accent={tokens?.accent}/>
    </AbsoluteFill>
    {duration >= nominal && <AbsoluteFill style={{pointerEvents: 'none', background: '#EAF4FA',
      opacity: interpolate(f, [126, 134], [0, 1], {...clamp, easing: Easing.in(Easing.quad)})}}/>}
    </AbsoluteFill>
  </AbsoluteFill>;
}

// The matchup rail alone, over a pre-rendered opening and the first play (spec 003).
// Its local frame 0 is the rail's first rise frame; it exits over its last 10 frames.
export function OpeningRail({data, teams, tokens, duration}) {
  const {width, height} = useVideoConfig();
  return <AbsoluteFill style={{zIndex: 30, pointerEvents: 'none'}}>
    <style>{fontCss}</style>
    <RegionalMatchup data={data} teams={teams} duration={duration} vertical={height > width}/>
  </AbsoluteFill>;
}

export const OpenStinger = props => <Stinger data={{team: props.focusSide, date: props.date}} teams={props.teams}
  tokens={props.tokens} duration={props.durationFrames}/>;
export const PeriodWipe = props => <Stinger data={{team: props.focusSide}} teams={props.teams}
  tokens={props.tokens} duration={props.durationFrames} wipe/>;
