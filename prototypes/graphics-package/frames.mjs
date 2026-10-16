// PROTOTYPE — render specific seconds of one composition: node frames.mjs <id> <sec,sec,...>
import path from 'node:path';
import {bundle} from '@remotion/bundler';
import {selectComposition, renderStill} from '@remotion/renderer';
const [id, secs] = process.argv.slice(2);
const serveUrl = await bundle({entryPoint: path.resolve('src/index.jsx')});
const composition = await selectComposition({serveUrl, id});
for (const sec of secs.split(',').map(Number)) {
  await renderStill({serveUrl, composition, frame: Math.round(sec * 60), output: `out/f_${id}_${sec}.png`, scale: 0.5, chromiumOptions: {gl: 'angle'}});
}
