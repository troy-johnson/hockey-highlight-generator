# PROTOTYPE: graphics package and 3D stingers (throwaway)

This folder is throwaway code. It answered design questions for the Recap graphics package. Do not build on it. Build the real package from the decisions below.

## Questions it settled

1. **What does the graphics package look like?** (Beads: "Decide the graphics package look", hhg-3r5.10.) Variant D, the "Ice Pak Slab", was chosen. Variants A–C were rejected and are kept here only for reference.
2. **How do the stingers look and move?** (Beads: "Prototype the stinger motion in 3D", hhg-3r5.16.) The user approved both 3D stingers:
   - **Open (about 3 s):** a slab hockey-stops over a 3D regulation rink with the ICEPAK/HOCKEY lockup. The slab is one painted surface (body and caps). It has a soft snow spray, frost with skate scrapes, and a matchup plate.
   - **Period transition (about 1 s, over the footage):** snow on the lens. A skater's spray hits the camera, frost hides the cut, the crest appears on the glass, and the frost melts open onto the new period.

## Quality rules learned

- Do not use faceted or flat-shaded particles. Draw particles as soft quads that are motion-blurred along their screen-space velocity (`src/StreakFX.jsx`).
- Mist sprites must not depth-test against the ice or the slab. Depth-testing gives straight clipped edges.
- Do not use 1-px line geometry. Render at `dpr={2}` and let the capture downsample.
- Attached parts (caps, trims) share the parent's geometry, texture, and frost.
- Snow over bright ice footage needs cool grey-blue shading to read.

## Run

```sh
npm install
npm run studio                              # Remotion Studio
node render-all.mjs video Stinger3D         # renders out/Stinger3D-Open.mp4 and out/Stinger3D-Period.mp4
node frames.mjs Stinger3D-Open 0.6,1.2      # stills at the given seconds
```

Put the assets listed in `public/README.md` into `public/` first. The player names in the sample data are placeholders.

## Files

- `src/Stinger3D.jsx`: the open scene, the period transition ("snow on the lens"), and the earlier slab version of the period transition (`Stinger3D-PeriodSkate`).
- `src/StreakFX.jsx`: motion-blurred particles, the painted slab, and the lens frost.
- `src/IceFX.jsx`: the slab frost shader, mist, and snow.
- `src/Rink.jsx`: the procedural regulation rink.
- `src/VariantD.jsx`: the chosen package (scorebug, goal, penalty, period, and final cards). `src/StingerD.jsx` is the earlier CSS stinger.
