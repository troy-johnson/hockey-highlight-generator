// Select ink from the displayed fill, including the fade back to graphite.
export const scoreFlash = (color, strength) => {
  const base = [35, 43, 52];
  const rgb = [1, 3, 5].map((offset, i) => Math.round(parseInt(color.slice(offset, offset + 2), 16) * strength + base[i] * (1 - strength)));
  const linear = rgb.map(v => v / 255 <= .04045 ? v / 255 / 12.92 : ((v / 255 + .055) / 1.055) ** 2.4);
  const light = linear[0] * .2126 + linear[1] * .7152 + linear[2] * .0722;
  return {background: `rgb(${rgb.join(',')})`, color: light > .179 ? '#0E1319' : '#EEF3F6'};
};
