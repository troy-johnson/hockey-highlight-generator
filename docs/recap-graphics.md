# Recap graphics

The `graphics` stage runs after `mix`. Python writes props from the Game Sheet and the assembly timeline.
Remotion renders Variant D **Ice Pak Slab** graphics as ProRes 4444 clips with alpha.
ffmpeg downsamples the graphics and overlays them on the mixed Recap. It copies the mixed audio without encoding it again.

## Setup and commands

The graphics package requires Node.js, npm, ffmpeg, and ffprobe. Its pinned dependencies include Remotion 4.0.529 and React 19.1.0.
Install the dependencies once:

```sh
npm ci --prefix v3/graphics
```

Remotion downloads Chrome Headless Shell on its first render. Later renders use the local browser and bundled Inter fonts.
The Inter font license is in `v3/graphics/public/Inter-OFL.txt`.
The stingers use pinned `@remotion/three` 4.0.529, React Three Fiber 9.8.1, and three.js 0.186.1.
The renderer selects Chromium's ANGLE backend for WebGL.

Run the pipeline with `hockeyrecap run GAME_FOLDER`, or rerun graphics with:

```sh
hockeyrecap rerun GAME_FOLDER --from graphics
```

For a direct run, pass effective options as JSON. The pipeline supplies these options automatically.

```sh
.venv/bin/python v3/scripts/recap_graphics.py GAME_FOLDER --options '{}'
```

Add `--props-only` to write props and check the mixed video's dimensions, frame rate, and frame count.
For a manual preview, render an inclusive frame range:

```sh
node v3/graphics/render.mjs GAME_FOLDER/recap_graphics_props.json OUTPUT_DIRECTORY 150:179
```

A preview manifest contains only that range. Do not use it as the full Recap overlay.

## Inputs and outputs

The stage reads `game_sheet.json`, `selection.json`, `recap_assembly.json`, and `recap_audio.json`.
It reads the mixed video named by `recap_audio.json.output`.
The assembly uses schema 3 at 30 fps. Selection uses schema 2.
The stage rejects mismatched goal sets, audio duration, audio source, and video frame counts.

The stage writes:

- `recap_graphics_props.json`: the Python-to-Remotion interface.
- `recap_graphics.json`: the output name, source video, duration, and flags.
- `<assembly output stem>_graphics.mp4`: the Recap with graphics and mixed audio.
- `.recap_cache/graphics/`: copied logos, alpha clips, and `clips.ffconcat`.

The runner fingerprints the input JSON, mixed video identity, Team Config, logos, and renderer sources.
It reruns graphics when these inputs change.
The runner does not fingerprint installed tool versions. Use `rerun --from graphics` after a toolchain change.
The stage checks the completed video's frame count before it publishes the video, props, and report.
Failed renders preserve the previous published props and report. The cache retains diagnostic props and alpha clips.

## Theme

The stage matches each Game Sheet team name to `Team Config.name`, without case differences.
The effective options supply `teams`, `focus_team`, `perspective`, and `_layers.config_dir`.
Team Config supports:

```json
{
  "name": "Ice Pak",
  "short_name": "IPK",
  "colors": {"primary": "#1E3160", "secondary": "#5AA4D0"},
  "logo": "assets/icepak-square.png"
}
```

Colors can also be an array: `[primary, secondary]`. Each color must use `#RRGGBB`.
A relative logo path starts at the config directory, normally `~/hockey/`.
Supported logos are PNG, JPEG, SVG, and WebP. Python copies each logo to the renderer's public directory.
The stage uses an acronym when no logo exists. It flags a configured logo that is missing.
Invalid colors use defaults and produce flags.
An unmatched Focus Team produces a flag and uses neutral cards.

The home and away tiles use their primary colors. Their edge stripes use their secondary colors.
Semantic tokens stay consistent: white score and GOAL panels, navy state panels, black details, and blue labels.
In focus perspective, an opponent GOAL uses a compact card. Neutral perspective uses equal cards.

## Props contract: schemaVersion 1

All frame positions refer to the **output Recap**, not the camera timeline.
`startFrame` is inclusive. `durationFrames` is a positive integer. The end frame is exclusive.
The renderer must not infer scores, players, periods, or timing.

```json
{
  "schemaVersion": 1,
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "durationFrames": 1200,
  "teams": {
    "home": {"name": "Ice Pak", "acronym": "IPK", "primary": "#1E3160", "secondary": "#5AA4D0", "logo": "home.png"},
    "away": {"name": "Visitors", "acronym": "VSR", "primary": "#7B2D36", "secondary": "#FFFFFF", "logo": null}
  },
  "focusSide": "home",
  "perspective": "focus",
  "tokens": {"score": "#FFFFFF", "state": "#0D1B2A", "info": "#0A0A0A", "hero": "#FFFFFF", "label": "#5AA4D0"},
  "events": [{"kind": "scorebug", "startFrame": 0, "durationFrames": 150, "data": {"score": [0, 0], "period": 1}}],
  "flags": []
}
```

`logo` is a public asset basename or `null`. `focusSide` is `home`, `away`, or `null`.
`score` always uses `[home, away]`. A period greater than 3 displays as overtime.
Events appear in layer order: scorebug first, event cards next, FINAL last.

| Event kind | Data fields |
| --- | --- |
| `scorebug` | `score`, `period` |
| `goal` | `id`, `team`, `num`, `scorer`, `assists` (display strings), `type`, `period` |
| `penalty` | `id`, `team`, `num`, `name`, `minutes`, `infraction` |
| `period` | `score`, `period` |
| `final` | `score`, `table.home`, `table.away` |
| `open_stinger` | `team` (`home`, `away`, or `null`) |
| `period_wipe` | `team`, `score`, `period` |

Each FINAL table contains `skaters` and `goalies` arrays.
A skater row has `num`, `name`, `g`, `a`, `pts`, and `pim`.
`pim` is a number or `null`. Unreadable penalty minutes produce a flag and an em dash in FINAL.
Numeric minutes, combined penalties such as `2+10`, and durations such as `2:30` contribute to PIM.
A goalie row has `num`, `name`, and `saves`.
Skaters with points or PIM appear in descending PTS, goals, and PIM order.
FINAL includes both teams and every eligible row.

## 3D stingers

The default `--start stinger` reserves 75 frames before chronological play.
The slab slides onto a 200-by-85-foot rink, turns through a hockey stop, and releases soft snow.
The face uses the Focus Team's name, colors, and configured logo. Ice Pak displays the `ICEPAK / HOCKEY` lockup.
A missing logo uses the wordmark. An unmatched Focus Team uses a neutral hockey face.
The face, stripes, and frost share the slab outline and parent transform.
Screen-space snow and mist cannot intersect the rink or slab geometry.

`--start cold_open` adds up to six seconds from the highest-interest surviving play before the stinger.
The main Recap retains chronological order and includes that play again.
The teaser has source audio and PA muting, but no scorebug, goal horn, or goal card.
`--start play` starts chronological play immediately and omits the opening stinger.
All modes retain the four-minute assembly cap.

Assembly schema 3 adds `start` and `opening`.
`opening` contains ordered `cold_open` and `stinger` entries with the same frame and source fields as clips.
A stinger has no source parts. The assembly renders black frames; graphics replaces them with the 3D scene.
The mix places `game_start` and its neutral sting at the stinger's output start.
Graphics rejects a reserved opening that does not match the audio cue.
For an older assembly, explicit `start: stinger` overlays its first 75 frames without changing audio or duration.
An older assembly needs `rerun --from assembly` for a cold open or a reserved opening.

Each `sfx_period_wipe` cue starts a 30-frame lens-snow wipe with the Focus Team crest or wordmark.
The two-second period card follows the wipe. Both events use the upcoming period's score.
Short end-of-video events stop at the last frame.
Layer order is scorebug, cards, stingers, then FINAL.
The renderer produces alpha at twice the output resolution. ffmpeg downsamples with Lanczos and copies the mixed audio.

The existing render CLI also accepts `OpenStinger` and `PeriodWipe` as its fourth argument:

```sh
node v3/graphics/render.mjs open-props.json OUTPUT_DIRECTORY 0:74 OpenStinger
node v3/graphics/render.mjs wipe-props.json OUTPUT_DIRECTORY 0:29 PeriodWipe
node v3/graphics/render.mjs recap-props.json OUTPUT_DIRECTORY --validate
```

Standalone props use the same schema, teams, tokens, and `focusSide`, with an empty `events` array.
Set `durationFrames` to 75 for an opening or 30 for a wipe.
`--validate` checks frame bounds without creating output files or launching Chromium.

## Data and timing rules

Goal IDs use `home:1` or `away:1`, with a one-based Game Sheet row index.
Penalty IDs use `penalty:home:1` or `penalty:away:1`.
Player names come from the team roster. Missing names display as `#<jersey>`.
FINAL derives G, A, PTS, and PIM from all Game Sheet rows, including goals omitted from the Recap.
Goalie saves use this optional Game Sheet section:

```json
{"goalkeeping": {"home": [{"player": "30", "name": "C. Goalie", "saves": 24}], "away": []}}
```

Zero saves is a valid value. Missing saves produce a flag and an unavailable label, not an invented number.
The current Scoresheet reader does not supply this section. A reviewed Game Sheet can supply it.

The scorebug changes at the live goal's output moment. Replays keep the post-goal score.
All Selection goals count, including omitted goals. Their period and elapsed time define the scoring order.
When a sheet clock is missing, known camera moments define the order and produce a review flag.
The scorebug does not remove goals that it has already counted.
For other plays, unmatched goals use the Selection window's expected camera time.
The Selection field is `window.expected`. Missing timing produces a scorebug review flag; FINAL still counts that goal.

GOAL and penalty cards start at their existing audio cues.
GOAL and penalty cards last up to four seconds. They stop at the source clip boundary or the next top card.
Period cards follow the one-second wipe and last up to two seconds.
FINAL starts five seconds before the Recap end, or earlier if its audio cue starts earlier.
The existing FINAL audio cue plays during the card. Every card stops at the Recap end.
The stage flags a late card that overlaps FINAL. Review that closing timeline before publishing the Recap.
This stage does not change audio cue timing or extend the Recap.

Remotion renders at twice the output dimensions. ffmpeg downsamples with Lanczos filtering.
ProRes 4444 retains alpha; the final MP4 uses H.264 video and copies the mixed audio.
The final video uses BT.709 and limited range.
