import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle} from '@remotion/bundler';
import {renderMedia, selectComposition} from '@remotion/renderer';

const [propsPath, outputDir, range] = process.argv.slice(2);
if (!propsPath || !outputDir) throw new Error('Usage: node render.mjs props.json output-dir [first:last]');
const props = JSON.parse(fs.readFileSync(propsPath, 'utf8'));
if (props.schemaVersion !== 1) throw new Error('Unsupported graphics props schema');
const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(outputDir, 'public');
fs.mkdirSync(publicDir, {recursive: true});
// Keep local fonts available without a network request during rendering.
for (const name of fs.readdirSync(path.join(here, 'public'))) {
  fs.copyFileSync(path.join(here, 'public', name), path.join(publicDir, name));
}
const serveUrl = await bundle({entryPoint: path.join(here, 'src/index.jsx'), publicDir});
const composition = await selectComposition({serveUrl, id: 'RecapGraphics', inputProps: props});
const [first, last] = range ? range.split(':').map(Number) : [0, props.durationFrames - 1];
if (!Number.isInteger(first) || !Number.isInteger(last) || first < 0 || last < first || last >= props.durationFrames) {
  throw new Error('Invalid render frame range');
}
const files = [];
// Short clips allow alpha input through ffconcat.
for (let start = first; start <= last; start += 300) {
  const end = Math.min(start + 299, last);
  const name = `clip-${String(start).padStart(8, '0')}.mov`;
  await renderMedia({serveUrl, composition, inputProps: props,
    outputLocation: path.join(outputDir, name), frameRange: [start, end],
    codec: 'prores', proResProfile: '4444', imageFormat: 'png', pixelFormat: 'yuva444p10le',
    scale: 2, concurrency: 2});
  files.push(name);
  console.log(`[graphics] rendered frames ${start}-${end}`);
}
fs.writeFileSync(path.join(outputDir, 'clips.ffconcat'),
  'ffconcat version 1.0\n' + files.map(name => `file '${name}'\n`).join(''));
