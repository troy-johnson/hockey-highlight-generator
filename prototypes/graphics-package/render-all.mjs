// PROTOTYPE — renders review stills (default) or full MP4s (`node render-all.mjs video`).
import path from 'node:path';
import fs from 'node:fs';
import {bundle} from '@remotion/bundler';
import {selectComposition, renderStill, renderMedia} from '@remotion/renderer';

const mode = process.argv[2] || 'stills';
const only = process.argv[3];
const out = path.resolve('out'); fs.mkdirSync(out, {recursive: true});
const serveUrl = await bundle({entryPoint: path.resolve('src/index.jsx')});
const ids = [...['A-Broadcast', 'B-Glass', 'C-IceEdge', 'D-IcePakSlab'].flatMap((v) => [`${v}-Focus`, `${v}-Neutral`]), 'Stinger3D-Open', 'Stinger3D-Period'].filter((i) => !only || i.includes(only));
const STILL_SEC = [1.8, 6, 10.5, 12.5, 16, 23, 27, 29.5, 31.3, 35, 40];
for (const id of ids) {
  const composition = await selectComposition({serveUrl, id});
  if (mode === 'video') {
    await renderMedia({serveUrl, composition, codec: 'h264', outputLocation: `${out}/${id}.mp4`, concurrency: 6, chromiumOptions: {gl: 'angle'}});
    console.log('video', id);
  } else {
    for (const sec of STILL_SEC) {
      await renderStill({serveUrl, composition, frame: Math.round(sec * 60), output: `${out}/${id}_${String(sec).replace('.', '_')}.png`, scale: 0.5});
    }
    console.log('stills', id);
  }
}
