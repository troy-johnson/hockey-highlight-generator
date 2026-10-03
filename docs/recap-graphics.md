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

Each FINAL table contains `skaters` and `goalies` arrays.
A skater row has `num`, `name`, `g`, `a`, `pts`, and `pim`.
`pim` is a number or `null`. Unreadable penalty minutes produce a flag and an em dash in FINAL.
Numeric minutes, combined penalties such as `2+10`, and durations such as `2:30` contribute to PIM.
A goalie row has `num`, `name`, and `saves`.
Skaters with points or PIM appear in descending PTS, goals, and PIM order.
FINAL includes both teams and every eligible row.

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

GOAL, penalty, and period cards start at their existing audio cues.
GOAL and penalty cards last up to four seconds. They stop at the source clip boundary or the next top card.
Period cards last up to 2.5 seconds.
FINAL starts five seconds before the Recap end, or earlier if its audio cue starts earlier.
The existing FINAL audio cue plays during the card. Every card stops at the Recap end.
The stage flags a late card that overlaps FINAL. Review that closing timeline before publishing the Recap.
This stage does not change audio cue timing or extend the Recap.

Remotion renders at twice the output dimensions. ffmpeg downsamples with Lanczos filtering.
ProRes 4444 retains alpha; the final MP4 uses H.264 video and copies the mixed audio.
The final video uses BT.709 and limited range.
