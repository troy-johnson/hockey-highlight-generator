import React, {useEffect, useLayoutEffect, useMemo} from 'react';
import {useFrame, useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import {Easing, interpolate} from 'remotion';

const random = seed => {const n = Math.sin(seed * 127.13) * 43758.5453; return n - Math.floor(n);};

export function openingMotion(frame) {
  const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};
  const slide = interpolate(frame, [4, 26], [0, 1], {...clamp, easing: Easing.out(Easing.quad)});
  const drift = interpolate(frame, [26, 108], [0, 1], clamp);
  // The opening exits by pushing the camera through the lockup, not by sliding the slab away.
  const push = interpolate(frame, [114, 134], [0, 1], {...clamp, easing: Easing.inOut(Easing.quad)});
  const travel = 70 * interpolate(frame, [0, 26], [0, 1], {...clamp, easing: Easing.out(Easing.quad)}) + 3 * drift;
  const lean = .1 * interpolate(frame, [18, 26], [0, 1], clamp) *
    (1 - interpolate(frame, [26, 42], [0, 1], {...clamp, easing: Easing.out(Easing.back(2.2))}));
  // Lift the slab so its lowest beveled corner stays above the ice during the roll.
  const y = -1.55 + 1.53 * Math.cos(lean) + 5.5 * Math.abs(Math.sin(lean)) + .015;
  return {enter: 24 * (1 - slide), yaw: (-.35 * (1 - slide) + .18 - .06 * drift) * (1 - push),
    y, lean, travel, drift, push};
}

export function BladeCut({frame, travel}) {
  const resources = useMemo(() => {
    const points = Array.from({length: 23}, (_, i) => {
      const motion = openingMotion(i + 4);
      const point = new THREE.Vector3(-5.5, -1.53, .3).applyEuler(new THREE.Euler(0, motion.yaw, motion.lean));
      return new THREE.Vector3(point.x + motion.enter, 0, point.z + 74 - motion.travel);
    });
    const curve = new THREE.CatmullRomCurve3(points);
    const grooves = [-.07, .07].map(() => new THREE.TubeGeometry(curve, 512, .008, 4, false));
    return {curve, grooves, lengths: curve.getLengths(1024)};
  }, []);
  const progress = Math.max(0, Math.min(1, (frame - 4) / 22));
  const head = resources.curve.getPoint(progress);
  const sample = progress * 1024, index = Math.floor(sample);
  const length = THREE.MathUtils.lerp(resources.lengths[index], resources.lengths[Math.min(index + 1, 1024)], sample - index);
  const drawn = Math.ceil(length / resources.lengths[1024] * 512);
  resources.grooves.forEach(g => g.setDrawRange(0, drawn * 24));
  const impact = frame >= 26 ? Math.exp(-(frame - 26) / 10) : 0;
  useEffect(() => () => resources.grooves.forEach(g => g.dispose()), [resources]);
  return <group position={[0, -1.525, -74 + travel]}>
    {resources.grooves.map((g, i) => <mesh key={i} geometry={g} position={[0, 0, i ? .07 : -.07]}>
      <meshBasicMaterial color={new THREE.Color('#CFEFFF').multiplyScalar(.65 + impact * .25)}
        transparent opacity={progress ? .25 + impact * .25 : 0} depthWrite={false}/></mesh>)}
    <pointLight position={[head.x, .12, head.z]} intensity={progress < 1 && progress > 0 ? 3 : impact * 5}
      distance={2.5} color="#CFEFFF"/>
  </group>;
}

export function ContactShadow({position, yaw, lean}) {
  const resources = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 128;
    const c = canvas.getContext('2d'); c.scale(1, .5);
    const gradient = c.createRadialGradient(128, 128, 10, 128, 128, 128);
    gradient.addColorStop(0, 'rgba(0,0,0,.65)'); gradient.addColorStop(.55, 'rgba(0,0,0,.3)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)'); c.fillStyle = gradient; c.fillRect(0, 0, 256, 256);
    return {texture: new THREE.CanvasTexture(canvas), geometry: new THREE.PlaneGeometry(12, 3)};
  }, []);
  useEffect(() => () => {resources.texture.dispose(); resources.geometry.dispose();}, [resources]);
  const roll = Math.min(1, Math.abs(lean) / .1);
  return <group position={[position[0], -1.52, position[2] + .3]} rotation={[0, yaw, 0]}>
    <mesh geometry={resources.geometry} rotation={[-Math.PI / 2, 0, 0]}
      position={[-Math.sign(lean) * 4 * roll, 0, 0]} scale={[1 - .45 * roll, 1, 1]}>
      <meshBasicMaterial map={resources.texture} transparent opacity={1 - .35 * roll} depthWrite={false}/>
    </mesh>
  </group>;
}

export function SnowFlurries({frame}) {
  const {size, gl, camera} = useThree();
  const resources = useMemo(() => {
    const seeds = new Float32Array(320 * 4);
    for (let i = 0; i < 320; i++) seeds.set([
      (random(i * 11 + 1) - .5) * 30, random(i * 11 + 2) * 15,
      -16 + random(i * 11 + 3) * 27, .025 + random(i * 11 + 4) ** 2 * .09], i * 4);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(320 * 3), 3));
    geometry.setAttribute('seed', new THREE.BufferAttribute(seeds, 4));
    const material = new THREE.ShaderMaterial({transparent: true, depthWrite: false, toneMapped: false,
      uniforms: {frame: {value: 0}, pixels: {value: 1}},
      vertexShader: `attribute vec4 seed; uniform float frame; uniform float pixels; varying float alpha;
        void main() {vec3 p = seed.xyz;
          p.x += sin(frame * .025 + seed.y) * .6 + frame * .008;
          p.y = mod(seed.y - frame * .035, 12.0) - 1.45;
          vec4 eye = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * eye;
          gl_PointSize = clamp(seed.w * pixels / max(-eye.z, .3), 1.0, 36.0);
          alpha = .55 * smoothstep(.3, 2.0, -eye.z) * (1.0 - smoothstep(12.0, 40.0, -eye.z));}`,
      fragmentShader: `varying float alpha;
        void main() {float radius = length(gl_PointCoord - .5) * 2.0;
          float a = (1.0 - smoothstep(.1, 1.0, radius)) * alpha;
          if (a < .01) discard; gl_FragColor = vec4(.85, .94, 1.0, a);}`});
    return {geometry, material, viewport: new THREE.Vector4()};
  }, []);
  resources.material.uniforms.frame.value = frame;
  resources.material.uniforms.pixels.value = size.height * gl.getPixelRatio() / (2 * Math.tan(camera.fov * Math.PI / 360));
  useEffect(() => () => {resources.geometry.dispose(); resources.material.dispose();}, [resources]);
  return <points geometry={resources.geometry} material={resources.material} frustumCulled={false} renderOrder={4}
    onBeforeRender={(renderer, scene, viewCamera) => {
      renderer.getCurrentViewport(resources.viewport);
      resources.material.uniforms.pixels.value = resources.viewport.w * viewCamera.projectionMatrix.elements[5] / 2;
      resources.material.uniformsNeedUpdate = true;
    }}/>;
}

// Particle positions depend on the frame, not on browser playback time.
export function IceSpray({frame}) {
  const resources = useMemo(() => {
    const count = 3200, positions = new Float32Array(count * 3);
    const previous = new Float32Array(count * 3), radii = new Float32Array(count);
    const alpha = new Float32Array(count);
    const geometry = new THREE.BufferGeometry();
    // Each particle uses a camera-facing quad stretched over half a frame.
    const corners = new Float32Array(count * 8), indices = new Uint32Array(count * 6);
    for (let i = 0; i < count; i++) {
      corners.set([-1, -1, 1, -1, 1, 1, -1, 1], i * 8);
      indices.set([i*4, i*4+1, i*4+2, i*4, i*4+2, i*4+3], i*6);
    }
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.addGroup(0, 2300 * 6, 0); geometry.addGroup(2300 * 6, 900 * 6, 1);
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 12), 3));
    geometry.setAttribute('corner', new THREE.BufferAttribute(corners, 2));
    geometry.setAttribute('head', new THREE.BufferAttribute(new Float32Array(count * 12), 3));
    geometry.setAttribute('tail', new THREE.BufferAttribute(new Float32Array(count * 12), 3));
    geometry.setAttribute('radius', new THREE.BufferAttribute(new Float32Array(count * 4), 1));
    geometry.setAttribute('opacity', new THREE.BufferAttribute(new Float32Array(count * 4), 1));
    const material = new THREE.ShaderMaterial({transparent: true, depthWrite: false,
      uniforms: {resolution: {value: new THREE.Vector2(1920, 1080)}},
      vertexShader: `attribute vec2 corner; attribute vec3 head; attribute vec3 tail;
        attribute float radius; attribute float opacity; uniform vec2 resolution;
        varying vec2 local; varying float fade;
        void main() {
          vec4 a = projectionMatrix * modelViewMatrix * vec4(head, 1.0);
          vec4 b = projectionMatrix * modelViewMatrix * vec4(tail, 1.0);
          if (a.w <= .3) {gl_Position = vec4(2.0, 2.0, 2.0, 1.0); local = corner; fade = 0.0; return;}
          if (b.w <= .3) b = a;
          vec2 start = a.xy / a.w, end = b.xy / b.w;
          vec2 motion = (start - end) * resolution;
          vec2 dir = length(motion) > .01 ? normalize(motion) : vec2(1.0, 0.0);
          float blur = min(length(motion), 200.0);
          float size = clamp(radius * resolution.y / max(a.w, .3), 1.0, 80.0);
          vec2 offset = dir * corner.x * (size + blur * .25) + vec2(-dir.y, dir.x) * corner.y * size;
          gl_Position = vec4((start + offset * 2.0 / resolution) * a.w, a.z, a.w);
          local = corner; fade = a.w > .3 ? opacity * size / (size + blur * .25) : 0.0;
        }`,
      fragmentShader: `varying vec2 local; varying float fade;
        void main() {float a = pow(1.0 - smoothstep(.1, 1.0, length(local)), 1.5) * fade;
          if (a < .002) discard; gl_FragColor = vec4(.88, .96, 1.04, a);}`});
    const mistMaterial = material.clone(); mistMaterial.depthTest = false;
    mistMaterial.uniforms = material.uniforms;
    return {geometry, material, mistMaterial, count, positions, previous, radii, alpha};
  }, []);
  const {size, gl} = useThree();
  resources.material.uniforms.resolution.value.set(size.width * gl.getPixelRatio(), size.height * gl.getPixelRatio());
  for (let i = 0; i < resources.count; i++) {
    const r = k => random(i * 23 + k);
    const mist = i >= 2300;
    const birth = 20 + r(1) * 10, age = (frame - birth) / 30;
    const life = mist ? 1.3 + r(2) : .6 + r(2) * 1.3;
    const vx = -3 - r(3) * (mist ? 4 : 12), vy = 1 + r(4) * (mist ? 3 : 9);
    const vz = -2 + r(5) * 12, drag = mist ? 3 : 1.8;
    const at = t => {
      const time = Math.max(0, t), travel = (1 - Math.exp(-drag * time)) / drag;
      return [-4.7 + r(6) * 1.7 + vx * travel,
        Math.max(-1.5, -1.35 + vy * travel - (mist ? 1 : 5) * time * time),
        .3 + r(7) * 1.2 + vz * travel];
    };
    resources.positions.set(at(age), i * 3); resources.previous.set(at(age - 1/60), i * 3);
    resources.radii[i] = mist ? .15 + r(8) * .45 : .008 + r(8) ** 3 * .12;
    resources.alpha[i] = age >= 0 && age < life ? Math.min(1, age * 20) * (1 - age / life) ** 2 * (mist ? .13 : .8) : 0;
    for (let corner = 0; corner < 4; corner++) {
      const v = i * 4 + corner;
      resources.geometry.attributes.head.array.set(resources.positions.subarray(i*3, i*3+3), v*3);
      resources.geometry.attributes.tail.array.set(resources.previous.subarray(i*3, i*3+3), v*3);
      resources.geometry.attributes.radius.array[v] = resources.radii[i];
      resources.geometry.attributes.opacity.array[v] = resources.alpha[i];
    }
  }
  for (const key of ['head', 'tail', 'radius', 'opacity']) resources.geometry.attributes[key].needsUpdate = true;
  useEffect(() => () => {resources.geometry.dispose(); resources.material.dispose(); resources.mistMaterial.dispose();}, [resources]);
  return <mesh geometry={resources.geometry} material={[resources.material, resources.mistMaterial]} frustumCulled={false} renderOrder={5}/>;
}

export function IceBloom({frame}) {
  const {gl, scene, camera, size} = useThree();
  const resources = useMemo(() => {
    const target = new THREE.WebGLRenderTarget(size.width * gl.getPixelRatio(), size.height * gl.getPixelRatio(),
      {samples: 4, type: THREE.HalfFloatType});
    const composer = new EffectComposer(gl, target);
    const render = new RenderPass(scene, camera);
    const bloom = new UnrealBloomPass(new THREE.Vector2(size.width * gl.getPixelRatio(), size.height * gl.getPixelRatio()), .35, .65, 1.1);
    const output = new OutputPass();
    composer.addPass(render); composer.addPass(bloom); composer.addPass(output);
    return {composer, render, bloom, output};
  }, [gl, scene, camera]);
  useLayoutEffect(() => {resources.composer.setPixelRatio(gl.getPixelRatio());
    resources.composer.setSize(size.width, size.height);}, [resources, gl, size.width, size.height]);
  useEffect(() => () => {resources.render.dispose(); resources.bloom.dispose(); resources.output.dispose();
    resources.composer.dispose();}, [resources]);
  resources.bloom.strength = .3 + (frame >= 26 ? .25 * Math.exp(-(frame - 26) / 7) : 0);
  useFrame(() => resources.composer.render(0), 1);
  return null;
}
