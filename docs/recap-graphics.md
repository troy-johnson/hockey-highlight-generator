# Recap graphics

The `graphics` stage runs after `mix`. Python writes props from the frozen game data and the assembly timeline.
Remotion renders the approved **Ice Pak regional** graphics as ProRes 4444 clips with alpha.
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
The graphics stage uses a local workspace under the system temporary directory, keyed by the Game Folder path.
Add `--workspace LOCAL_DIRECTORY` to select a persistent local directory.
Props, copied logos, alpha clips, the concat manifest, and the temporary video stay in that workspace.
The stage validates the local video before copying final artifacts to the Game Folder.
It copies all artifacts before replacing existing outputs. A failed copy preserves the existing video and reports.
Do not select a network share as the workspace.
For a manual preview, render an inclusive frame range:

```sh
node v3/graphics/render.mjs GAME_FOLDER/recap_graphics_props.json OUTPUT_DIRECTORY 150:179
```

A preview manifest contains only that range. Do not use it as the full Recap overlay.

## Inputs and outputs

The stage reads `game_data.json`, `coverage.json`, `audio_signals.json`, `selection.json`, `recap_assembly.json`, and `recap_audio.json`.
Only the frozen snapshot supplies names, numbers, score, date, and statistics. A graphics-only rerun never refreshes a Roster.
Coverage supplies periods. The stage checks presentation cues against the matched whistle and rejects an early cue.
The mix confirms onset in the kept-camera audio and records `presentation_whistles` with the matched time, onset, and method.
Graphics checks that proof against the unchanged source timeline. It does not use an earlier merged-camera timestamp for presentation.
The proof includes the exact source mapping. An older report without proof needs `rerun --from mix`.
It reads the mixed video named by `recap_audio.json.output`.
The assembly uses schema 3 at 30 fps. Selection uses schema 2.
The stage rejects mismatched goal sets, audio duration, audio source, and video frame counts.

The stage writes:

- `recap_graphics_props.json`: the Python-to-Remotion interface.
- `recap_graphics.json`: the output name, source video, duration, and flags.
- `<assembly output stem>_graphics.mp4`: the Recap with graphics and mixed audio.
- Local workspace: copied logos, alpha clips, diagnostic props, and `clips.ffconcat`.

The runner fingerprints the input JSON, mixed video identity, frozen Team Config, logos, and renderer sources.
It reruns graphics when these inputs change.
The runner does not fingerprint installed tool versions. Use `rerun --from graphics` after a toolchain change.
The stage checks the completed video's frame count before it publishes the video, props, and report.
Failed renders preserve the previous published props and report. The cache retains diagnostic props and alpha clips.

## Theme

The stage matches each Game Sheet team name to `Team Config.name`, without case differences.
The effective options supply `teams`, `focus_team`, `perspective`, `kit`, and `_layers.config_dir`.
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

### Kits

A Team Config can list the team's kits (jerseys). The `kit` option selects one for the Focus Team only:

```json
{
  "default_kit": "red",
  "kits": {
    "red": {"fill": "#A60735", "line": "#E8354F", "opening": "assets/black_ice_opening_red.mp4"},
    "navy": {"fill": "#102E52", "line": "#5AA4D0"},
    "white": {"fill": "#FFFFFF", "line": "#EEF3F6", "text": "#1E3160", "crest": "assets/icepak.png"}
  }
}
```

- `fill` replaces `primary` for large areas. `line` is for thin lines and light. `text` is the type colour on the fill.
- Without `text`, the stage uses white when white has 4.5:1 contrast on the fill, and graphite otherwise.
- `crest` is optional and replaces the Team Config crest for that kit.
- `opening` is optional: a pre-rendered opening video for that kit (see "Pre-rendered opening"). A team-level `opening` applies to every kit without one.
- `default_kit` becomes the inferred `kit` in `recap_options.json`. Use `--kit navy` or set `"inferred": false` to change it.
- `fill` and `line` are required. An unknown kit, a missing `fill` or `line`, or an invalid colour produces a flag and keeps the team colours.
- The kit follows the selected Focus Team: a pinned `focus_team` in `recap_options.json`, or `--focus-team` for one run.

Every team gets a `line` colour. It must have 3:1 contrast on the graphite plate `#0E1319`.
A line that fails uses frost grey `#9FB2C1`. A failing kit line also produces a flag.
When the opponent's line is too similar to the Focus Team line (RGB distance under 120), the opponent uses its secondary colour, or frost grey when that also fails.
The accent token is the Focus Team line. Without a Focus Team it stays ice cyan `#7FD3F0`.

The home and away tiles use their primary colors. Thin team lines (scorebug, rails, FINAL, matchup rail) and the frost-pane glow use the `line` colour.
Only the period-plate score slab keeps a secondary-colour edge stripe.
Semantic tokens stay consistent: white score and GOAL panels, navy state panels, black details, and blue labels.
Focus perspective uses the Focus Team for transitions and the opening. Neutral perspective uses neutral transitions and equal team treatment.
Both Perspectives retain both teams in the scorebug and FINAL. Black Ice is the package name, not a channel brand.

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
    "home": {"name": "Ice Pak", "acronym": "IPK", "primary": "#A60735", "secondary": "#000000",
             "line": "#E8354F", "kit": "red", "text": "#FFFFFF", "logo": "home.png"},
    "away": {"name": "Visitors", "acronym": "VSR", "primary": "#7B2D36", "secondary": "#FFFFFF",
             "line": "#9FB2C1", "logo": null}
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
| `event_bar` | `team`, `label` |
| `goal` | `id`, `team`, `num`, `scorer`, `assists` (display strings), `type`, `period` |
| `penalty` | `id`, `team`, `num`, `name`, `minutes`, `infraction` |
| `period` | `score`, `period` |
| `final` | `score`, `table.home`, `table.away` |
| `open_stinger` | `team` (`home`, `away`, or `null`) |
| `matchup_rail` | `team` (`home`, `away`, or `null`), `date` |
| `period_wipe` | `team`, `score`, `period` |

Each FINAL table contains `skaters` and `goalies` arrays.
A skater row has `num`, `name`, `g`, `a`, `pts`, and `pim`.
`pim` is a number or `null`. Unreadable penalty minutes produce a flag and an em dash in FINAL.
Numeric minutes, combined penalties such as `2+10`, and durations such as `2:30` contribute to PIM.
A goalie row has `num`, `name`, and `saves`.
Skaters keep the frozen snapshot order. Graphics does not sort tied rows again.
FINAL includes both teams and every frozen row. More than ten rows use multiple pages, with at least 150 frames per page.
Recorded zero PIM stays visible. A short closing section fails rather than silently dropping a row.

## 3D stingers

The default `--start stinger` reserves 135 frames before chronological play.
The full matchup holds for about 1.6 seconds before the exit starts.
The slab slides onto a 200-by-85-foot rink, turns through a hockey stop, and releases soft snow.
The face uses the Focus Team's colors and configured `lockup`. Ice Pak displays the `ICEPAK / HOCKEY` lockup.
A missing lockup uses the wordmark, then the logo, then the team name. An unmatched Focus Team uses a neutral hockey face.
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
In placeholder mode, `game_start` follows the reserved opening duration.
Its original sound sketch has a blade scrape, impact, shimmer, and exit scrape.
The impact occurs at frame 26 of the 135-frame choreography.
The exit scrape occurs at frames 108–134. The sound stays quiet during the matchup hold.
An approved `game_start` file keeps its declared duration. A play start keeps the legacy sound.
The generated sketch remains placeholder audio until the user approves a Cue Library asset.
Graphics rejects a reserved opening that does not match the audio cue.
### Pre-rendered opening

When the Focus Team kit (or Team Config) has an `opening` video, the assembly uses it in place of the black frames.
The video must be 1920×1080 at 30 fps. The Black Ice opening is 148 frames, rendered by `v3/blender/black_ice_macro.py`.
A relative path is relative to the config folder. An unreadable video produces a flag and uses the 3D stinger.
The stinger entry takes the video length less 9 frames and stores the video in `opening_video`.
The last 9 frames dissolve over the first 9 frames of the next piece. There is no white flash.
Graphics then draws only a `matchup_rail` event, not the 3D stinger.
The rail rises from frame 95 of the opening, holds for 75 frames after the dissolve, and exits in 10 frames.
The rail ends at least 4 frames before the first information rail. A shortened rail produces a flag.
The runner fingerprints the video, so a new render of it reruns the assembly.

The approved v12 opening includes mastered sound. Set the kit's `opening` to that asset, not an older silent render.
The asset stays outside git. The pipeline does not change the real database or Team Config to select it.
The normal mix uses `audio.mode: game_sound` when an assembly contains a pre-rendered opening.
It keeps source seeks and speed and applies 15 ms cut fades.
By default a Replay has live sound: the live clip's source continues at the clip's speed, so the sound under a Replay follows on from the play.
FINAL continues from the end of that sound. Sound that runs past the end of a chapter file continues in the Recording's next chapter.
If no next chapter exists, silence fills the rest and the mix flags it.
Whistle times map through the sound the mix used. Graphics read `live_replay` from `recap_audio.json`, so a proof stays bound to its mix.
Set `audio.live_replay: false` to use the slowed Replay sound, lowered by 8 dB.
It normalizes rink sound to -20 LUFS before adding the opening sound, then limits the output peak.
PA music muting still applies. This mode adds no synthetic Bed or placeholder sounds.
When a Cue Library manifest exists, this mode adds the game cues from it (hhg-3r5.110). See `docs/cue-library.md`.
Set `audio.mode: broadcast` to retain the Cue Library mix. Without a pre-rendered opening, broadcast remains the default.

For an older assembly, explicit `start: stinger` compresses the opening motion into its first 75 frames.
This fallback does not change audio or duration.
An older assembly needs `rerun --from assembly` for a cold open or a reserved opening.

By default a period change is a plain cut. The scorebug period cell rolls to the new period and flashes cyan for 30 frames, like a score change. The mix writes `sfx_period_wipe` cues only when the option `period_transitions` is `true` (intended for full-game coverage).
Each `sfx_period_wipe` cue starts a 30-frame frost-pane wipe. The pane uses a WebGL frost shader and shows the Focus Team `crest` at about 60% of the screen height. A missing `crest` uses the `logo`. Use a crest with a transparent background.
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
Set `durationFrames` to 135 for an opening or 30 for a wipe.
`--validate` checks frame bounds without creating output files or launching Chromium.

## Data and timing rules

Goal IDs use `home:1` or `away:1`, with a one-based Game Sheet row index.
Penalty IDs use `penalty:home:1` or `penalty:away:1`.
Player names and game numbers come from `game_data.json`. Missing names display as `#<jersey>`.
FINAL reads frozen G, A, PTS, and PIM, including events omitted from the Recap.
Goalie saves use this optional Game Sheet section:

```json
{"goalkeeping": {"home": [{"player": "30", "name": "C. Goalie", "saves": 24}], "away": []}}
```

Zero saves is a valid value. Missing saves produce a flag and an unavailable label, not an invented number.
The current Scoresheet reader does not supply this section. A reviewed Game Sheet can supply it.

The scorebug changes on the first output frame at or after the matched referee whistle.
The event bar starts on that frame. The event card starts its entry six frames later.
Adjacent clips retain the score roll's origin. Replay footage cannot trigger another score change.
If the live cut excludes the whistle, map through the first retained Replay piece that contains it, using that piece's seek and speed.
Python supplies each card's score and the scorebug animation origins. The renderer does not derive game state.
Without a matched whistle, the scorebug holds the old score through that goal's Replay and records a flag.
The next chronological play can include that recorded goal. FINAL always uses the frozen score.
All Selection goals count, including omitted goals. Their period and elapsed time define the scoring order.
When a sheet clock is missing, known camera moments define the order and produce a review flag.
The scorebug does not remove goals that it has already counted.
For other plays, unmatched goals use the Selection window's expected camera time.
The Selection field is `window.expected`. Missing timing produces a scorebug review flag; FINAL still counts that goal.

GOAL and penalty cues must bind to a measured referee whistle. Graphics rejects a stale cue before that whistle.
GOAL and penalty cards last up to four seconds. They stop at the source clip boundary or the next top card.
Period cards follow the one-second wipe and last up to two seconds.
FINAL starts five seconds before the Recap end, or earlier if its audio cue starts earlier.
The existing FINAL audio cue plays during the card. Every card stops at the Recap end.
The stage flags a late card that overlaps FINAL. Review that closing timeline before publishing the Recap.
This stage does not change audio cue timing or extend the Recap.

The Short and single-goal paths confirm their kept audio before rendering graphics. Their text overlay waits for confirmation and fits complete names.
This port does not replace that older Short overlay with the regional Remotion layout.

Remotion renders at twice the output dimensions. ffmpeg downsamples with Lanczos filtering.
ProRes 4444 retains alpha; the final MP4 uses H.264 video and copies the mixed audio.
The final video uses BT.709 and limited range.
