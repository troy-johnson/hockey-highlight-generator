import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {bundle} from '@remotion/bundler';
import {renderMedia, selectComposition} from '@remotion/renderer';

const [propsPath, outputDir, range, id = 'RecapGraphics'] = process.argv.slice(2);
if (!propsPath || !outputDir) throw new Error('Usage: node render.mjs props.json output-dir [first:last|--validate] [composition]');
const props = JSON.parse(fs.readFileSync(propsPath, 'utf8'));
if (props.schemaVersion !== 1) throw new Error('Unsupported graphics props schema');
if (![props.width, props.height, props.durationFrames].every(n => Number.isInteger(n) && n > 0) || props.fps !== 30) {
  throw new Error('Graphics require positive integer dimensions and frames at 30 fps');
}
if (!['RecapGraphics', 'OpenStinger', 'PeriodWipe'].includes(id)) throw new Error('Unknown graphics composition');
if (![null, 'home', 'away'].includes(props.focusSide)) throw new Error('Invalid Focus Team side');
for (const side of ['home', 'away']) {
  const team = props.teams?.[side];
  if (!team || typeof team.name !== 'string' || typeof team.acronym !== 'string' ||
      ![team.primary, team.secondary].every(c => /^#[0-9a-f]{6}$/i.test(c))) throw new Error('Invalid graphics team');
  if (team.logo !== null && (typeof team.logo !== 'string' || !team.logo ||
      /[/\\]/.test(team.logo) || ['.', '..'].includes(team.logo))) throw new Error('Logo must be a public asset basename');
}
if (!['score', 'state', 'info', 'hero', 'label'].every(k => /^#[0-9a-f]{6}$/i.test(props.tokens?.[k]))) {
  throw new Error('Invalid graphics tokens');
}
if (!Array.isArray(props.events)) throw new Error('Graphics events must be an array');
for (const event of props.events) {
  if (!['scorebug', 'goal', 'penalty', 'period', 'final', 'open_stinger', 'period_wipe'].includes(event.kind)) {
    throw new Error('Unknown graphics event');
  }
  if (!Number.isInteger(event.startFrame) || event.startFrame < 0 ||
      !Number.isInteger(event.durationFrames) || event.durationFrames <= 0) throw new Error('Invalid graphics event frames');
  if (event.startFrame + event.durationFrames > props.durationFrames) throw new Error('Event exceeds graphics duration');
  if (!event.data || typeof event.data !== 'object') throw new Error('Missing graphics event data');
  if (['open_stinger', 'period_wipe'].includes(event.kind) && ![null, 'home', 'away'].includes(event.data.team)) {
    throw new Error('Invalid stinger team side');
  }
}
const [first, last] = range && range !== '--validate' ? range.split(':').map(Number) : [0, props.durationFrames - 1];
if ((range && range !== '--validate' && !/^\d+:\d+$/.test(range)) ||
    !Number.isInteger(first) || !Number.isInteger(last) || first < 0 || last < first || last >= props.durationFrames) {
  throw new Error('Invalid render frame range');
}
if (range === '--validate') {
  console.log(`[graphics] valid ${id} props: ${props.durationFrames} frames`);
  process.exit(0);
}
const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(outputDir, 'public');
fs.mkdirSync(publicDir, {recursive: true});
// Keep local fonts available without a network request during rendering.
for (const name of fs.readdirSync(path.join(here, 'public'))) {
  fs.copyFileSync(path.join(here, 'public', name), path.join(publicDir, name));
}
const serveUrl = await bundle({entryPoint: path.join(here, 'src/index.jsx'), publicDir});
const chromiumOptions = {gl: 'angle'};
const composition = await selectComposition({serveUrl, id, inputProps: props, chromiumOptions});
const files = [];
// Short clips allow alpha input through ffconcat.
for (let start = first; start <= last; start += 300) {
  const end = Math.min(start + 299, last);
  const name = `clip-${String(start).padStart(8, '0')}.mov`;
  await renderMedia({serveUrl, composition, inputProps: props,
    outputLocation: path.join(outputDir, name), frameRange: [start, end],
    codec: 'prores', proResProfile: '4444', imageFormat: 'png', pixelFormat: 'yuva444p10le',
    scale: 2, concurrency: 1, chromiumOptions});
  files.push(name);
  console.log(`[graphics] rendered frames ${start}-${end}`);
}
fs.writeFileSync(path.join(outputDir, 'clips.ffconcat'),
  'ffconcat version 1.0\n' + files.map(name => `file '${name}'\n`).join(''));
