import React from 'react';
import {useCurrentFrame} from 'remotion';

export const SkateCutAccent = ({delay = 0}) => {
  const frame = useCurrentFrame();
  const reveal = Math.max(0, Math.min(1, (frame - delay - 4) / 18));
  return <svg viewBox="0 0 400 20" preserveAspectRatio="none" aria-hidden="true"
    style={{position: 'absolute', bottom: -10, left: 16, width: 180, maxWidth: '70%',
      height: 12, overflow: 'visible', pointerEvents: 'none', opacity: .55 * reveal}}>
    {[0, 3].map(offset => <path key={offset} d={`M400 ${6 + offset} Q220 ${17 + offset} 0 ${10 + offset}`}
      fill="none" stroke="#CFEFFF" strokeWidth={1} vectorEffect="non-scaling-stroke" strokeLinecap="round"
      pathLength={1} strokeDasharray={1} strokeDashoffset={1 - reveal}/>) }
  </svg>;
};
