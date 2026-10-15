# Black Ice: broadcast graphics package

## Approved production design (hhg-3r5.106)

The approved v21 adapters define the Ice Pak regional design. The v22 `final.tables` defines the FINAL values and row order.
This section replaces conflicting review-pass rules below. This work implements the approved design; it does not start another design round.
Black Ice remains the package name. Do not add a Black Ice channel mark or watermark.

- Use a stacked scorebug with two team rows and one period cell. Place the event bar below both rows.
- Show the full-width, light REPLAY tab below the scorebug when no event bar occupies that position.
- Use matching event cards with a crest, event label, player, detail, and separate value cell.
- Use the approved medium team glow and cubic ease-in-out motion for entry, hold changes, and exit.
- Use dark skated-pond transitions. Keep the smaller crest inside the pane. Do not add coloured edge glow.
- Centre the opening matchup rail. Put the date above the two team cards and their VS cell.
- Stack both teams in the FINAL result plate. Lift that plate before revealing the two statistics tables.
- Keep G, A, PTS, and PIM on the shared grid, including recorded zero PIM. Keep unknown goalie figures unavailable.

### Frozen data and timing

Names, numbers, scores, dates, and statistics come only from `game_data.json` (ADR-0004).
Graphics must not read live Rosters or recompute frozen statistics. Keep the snapshot row order, including equal-stat rows.
The approved v22 validation requires Kenyon Ross at 15 PIM: a major (5) plus a game misconduct (10).
This is a validation value, not a production constant.

Periods come from `coverage.json`, not a cached Selection period list. Keep approved Selection choices, cuts, and Replay frames unchanged.
Missing coverage stops graphics. A new snapshot or coverage file invalidates the graphics fingerprint.

The user decided on 2026-10-15 that **score changes, event bars, and event cards must all wait for the referee whistle**.
Bind the whistle to the kept event. Map its source time through the existing cut and speed.
Use the first output frame at or after the whistle, never the nearest frame before it.
The flow moment has 12 Hz precision; a bound whistle may precede that estimate by one analysis sample.
Never use that tolerance to move the announcement before the actual whistle.

The event bar enters at the whistle frame. The matching card starts its entry six frames later.
Keep score and period animation origins across adjacent clips, including a live-to-Replay handoff.
When whistle timing is missing, omit the announcement and hold the old score through that goal's Replay.
The next chronological play can include the recorded goal. FINAL always uses the frozen score.
Reject stale mix cues that precede the bound whistle. Do not repair them with a fixed game-specific delay.

### Opening and game sound

Use the configured approved v12 opening video with its mastered sound. Keep its 148 frames and nine-frame dissolve.
The asset remains outside the repository. Configuration supplies its path; package source must not contain a machine-specific path.
With a pre-rendered opening, the normal mix defaults to `audio.mode: game_sound`.
Use the same source seeks and speed as the approved footage. Lower Replay sound by 8 dB and fade cut edges over 15 ms.
Normalize rink sound to -20 LUFS before adding the mastered opening sound. Apply the output peak limiter.
Do not add a synthetic Bed, horn, or graphic cue in game-sound mode. PA music muting still applies.
Keep `audio.mode: broadcast` available for the existing Cue Library mix and both Perspectives.
Detection audio weighting remains zero. Graphics sound does not change Detection.

### Text and complete statistics

Fit the complete text inside each cell. Never truncate a surname or use an ellipsis.
FINAL keeps all frozen rows. Use pages of at most ten skaters per team when necessary.
Each page needs at least 150 frames. Reject a shorter closing section instead of dropping rows.
Validate long names, both Perspectives, normal playback, fractional whistle frames, sound, and every approved FINAL value.

## Identity

Black Ice is the Ice Pak package only. A generic package for other teams is a separate, later design.
Ice Pak is a beer-league team. Most players are 45+, the core has played together for 15+ years, and players' kids now join the team.

"Black ice" has two meanings, and the package uses both:

- Pond ice that freezes slowly on cold, still nights. It is clear, deep, and strong. It takes years, like the team.
- The road hazard that you do not see until you are on it. Opponents underestimate a 45+ team.

These four features tie every component together:

1. Material: polished black ice. It is dark and glossy, with depth, trapped bubbles, fine cracks, and old skate scratches.
2. Motion: the blade cut. A skate carves a bright scratch through the ice and shows what is below it.
3. Graphic device: the line. A kit-coloured line under the ice, like the centre line, is the rule line on cards and the edge of wipes.
4. Sound: one ice family. Blade carve, the deep boom of lake ice, and a puck on the post.

Only Ice Pak's kit colour and crest change between games. The relief-wall opening (v4, hhg-3r5.86) is rejected as overdone.

### Tone

Serious and competitive, with a beer-league wink.

- Never joke on data. Scores, names, penalties, and calls stay exact.
- Humour appears only in optional copy slots from the Team Config. Each slot can be empty.
- Example slots: opening tag ("EST. YYYY · Still going"), period break ("Ice resurfacing. Hydrate."),
  FINAL win ("Cooler's open."), FINAL loss ("Same time next week.").
- A "Family assist" line needs verified roster family links. Do not infer family links from names.

### Kits

Ice Pak wears three kits. The red kit is the default and the most common.
A per-game option selects the kit. Each kit sets a fill colour, a line colour, and a crest.

| Kit | Fill | Text on fill | Line on black ice | Trim | Crest on dark surfaces |
| --- | --- | --- | --- | --- | --- |
| Red (default) | `#A60735` | `#FFFFFF` (7.8:1) | `#E8354F` (4.5:1) | black | `icepak_dark_logo.png` (placeholder) |
| Navy | `#102E52` | `#FFFFFF` | `#5AA4D0` (6.8:1) | light blue `#5AA4D0`, white | `icepak_dark_logo.png` |
| White | `#FFFFFF` | `#1E3160` | `#EEF3F6` (16.7:1) | navy `#1E3160`, light blue `#5AA4D0` | `icepak_dark_logo.png` |

Black ice: `#07090C` for the surface, with the graphite plate `#0E1319` as the reading surface.
Contrast ratios are WCAG ratios against `#0E1319`, or against the fill for text.

Logo files are in `Ext. Images/Ice Pak/Logos/` on the NAS (`/mnt/@usb/sdb1/...`; on the Mac `/Volumes/Partition_1/...`):

- Dark blue logo, `icepak_dark_logo.png`: RGBA with a solid white body. Use it on black ice and dark plates.
  For now, every kit and every surface uses this logo only (user decision).
- White logo, `Final_IcePak_Logo.png`: RGB on a white background, with no alpha. Its body is the background white.
  A cut-out makes the body transparent, so it reads as hollow on dark surfaces. Do not use it for now.
  The vector `Final_IcePak_Logo.ai` has the same hollow body.

Sources and status:

- Red: sampled from `icepak_alt_logo_1.png` (red and near-black wolves logo). Placeholder until the red-and-black crest arrives.
- Navy fill: sampled from the navy jersey sheet `icepak_dark_jersey.ai` (`#102E52`). The crest navy is `#1E3160`.
- White kit trim: from the white jersey mockup in `Final_IcePak_Logo.ai` (navy lower band and light-blue stripe).
- Replace the red values and the red kit crest when the red-and-black crest arrives.
Previews that use the blue crest for the red kit must say so.

The fill is for large areas, such as the score flash and the team cell. The line is for thin lines and light.
A line must have at least 3:1 contrast on `#0E1319`. The deep red fails (2.4:1), so the red kit uses a brighter red for lines.
Opponent cells use the opponent's colours. When an opponent colour fails 3:1 as a line, use a frost-grey line in that cell.
When the opponent line is too similar to the kit line, the opponent uses its secondary colour, or frost grey.
Thin team lines (scorebug, rails, FINAL, matchup rail) use the line colour. Large fills, such as the score flash, use the fill.

Ice-cyan accents in accepted components (replay tab, rule lines) change to the kit line colour in a reviewed pass.
The period flash is generic ice-white, not team-specific (user review). A correction uses an ice-white bar.
The GOAL rail logo tile has no bottom border (user review).
FINAL player rows (skaters and goalies) have no team-colour underline on the number cell (user review). Until that pass, the accepted cyan rules below apply.

## Design decision

Black Ice uses the accepted dark-studio opening as its parent scene.
The package combines carved ice, rectangular graphite plates, and clear upright type.
Team crests and narrow color accents identify teams. Large painted panels no longer carry every message.
The logo, score, and event remain more important than the effects.

This is an original design direction. Broadcaster references describe the quality target, not an endorsement.
Opus 5.5 reviewed the current source through OpenCode on Vercel. Its review did not include video playback.
The user accepted the v4 opening as an improvement, not as the final broadcast package.

## Review workflow

Each pass ends with a new named preview. Stop for user review before the next pass.
Keep previous previews and the published Recap. Do not replace the full Recap during design passes.
Each component pass includes a 16:9 Recap layout and a 9:16 reels layout.
The user accepted Pass 1 and requested the scoring team's primary color for the goal flash.
The user accepted Pass 1 v4. Pass 2 previews use a confirmed goal and separate fictional examples.
The user accepted Pass 2 v5. Pass 3 previews keep the complete confirmed replay and isolate the period example.
The user accepted Pass 3 replay entry and exit timing, but rejected the empty-looking transition material.
Its second revision replaces the SVG pane with a shader frost pane: growing crystalline frost, a bright leading front, a team-colour edge line, and a crest at about 60% of the screen height. The optional team `crest` asset supplies a transparent crest; the `logo` is the fallback.
Keep the accepted replay timing during material revisions. A fictional period example cannot use an actual team crest.
Fictional penalty and call cards use sample teams, a sample score, and a visible design label on a neutral background.
Actual officiating records are not connected yet. The renderer rejects calls outside labeled design previews.

| Pass | Review scope |
| --- | --- |
| 1 | Scorebug, readable typography, plate shape, team hierarchy, score change, replay state |
| 2 | GOAL, penalty, and officiating explanation cards; one shared information rail |
| 3 | Transition, replay entry and exit, period change; one shared frost-pane motion |
| 4 | FINAL summary, complete verified statistics, separate closing section |
| 5 | Opening materials, physical frost, reveal, lighting, reflection, and designed sound |
| 6 | Full sequence integration and vertical Remotion layouts |

The order can change after user review. A review can require another pass on the same component.

## Shared visual system

- Use a graphite plate, `#0E1319`, with an opaque reading surface.
- Use ice-white text, `#EEF3F6`, and frost-grey labels, `#9FB2C1`.
- Use rectangular plate edges, following user review. Keep text and numbers upright.
- The scorebug is rectangular, following user review. Its score cells use a slightly lighter graphite fill.
- Use the bundled Inter font. Use tabular figures and fixed score cells.
- Use weights 600–800. Reserve the largest type for the score and the event result.
- Use a bottom-only primary-color border on each team cell. Use a bottom-only ice-cyan border on the replay tab, with no period-cell border.
- Label regulation periods `1st`, `2nd`, and `3rd`. Use `OT` for overtime and an explicit `SO` for shootout.
- Flash the scoring team's primary color when its score increases. A correction uses ice cyan.
- Fill the scoring team's crest and score segment for 18 frames, then fade over 12 frames.
- Keep the score cell slightly lighter during the flash, using a ten-percent white tint of the primary color.
- Choose light or dark type from the displayed fill luminance to keep the flash readable.
- Use restrained frost edges and shallow shading. Remove repeated white outlines and decorative underlines from new plates.
- Keep text within the design margin: 96 pixels horizontally and 54 pixels vertically at 1920×1080.
- Use normal case for names. Do not force every name into an italic headline.

These margins are package design targets, not a claim about a broadcaster's delivery standard.
Date labels use a readable date. The underlying record keeps its ISO date.

## Component rules

All timing below uses 30 fps. Frame ranges use an exclusive end.

| Component | Trigger and information | Motion and duration |
| --- | --- | --- |
| Opening stinger | Before the first play, or after an optional cold open. Recap only: a Short starts on its first play. Team crest, opponent, date. | 148 frames, pre-rendered in Blender (`v3/blender/black_ice_macro.py`): the crest wakes under black ice, heats, breaks through, and holds as the hero shot. The last 9 frames (139–147) dissolve into the first play. No white flash. The matchup rail enters on frames 95–118, holds for 75 frames after the dissolve, and exits in 10 frames. Without a pre-rendered opening, the 135-frame Remotion stinger is the fallback. |
| Transition | A confirmed period boundary or an explicit editorial boundary. | 30-frame dark frost pane. Cut footage only during full cover. No white flash. |
| Scorebug | Play and replay footage. Home and away crests, abbreviations, scores, period. | Persistent. Enter in 12 frames. Change a score with a 10-frame number roll and a short filament response. |
| GOAL | A confirmed goal bound to the source play. Use its matched post-goal whistle for presentation timing. | Lower information rail. Enter in 14 frames; readable hold of at least 75 frames; exit in 10. Maximum 120 frames. |
| Penalty | A verified penalty with a kept source play. Show player, team, infraction, and assessed duration. | Shared rail. Enter in 12 frames; hold 75–105; exit in 10. Label duration as `2 MIN`, not a countdown. |
| Officiating explanation | A confirmed, source-bound decision entered in game data. | Shared rail. Result first, reason second, penalty disposition third. Enter in 14 frames; hold 90–120; exit in 10. |
| Replay | A declared replay entry. No score changes caused by replay footage. | 16-frame pane centered on each cut. Full cover on frames 7–9. Persistent REPLAY tab stays above the pane. |
| Period change | A verified upcoming period. | Recap and Short default: plain cut. The scorebug period cell rolls to the new period in 10 frames and flashes generic ice-white for 30 frames, like a score change. Option `period_transitions: true` (for future full-game coverage) restores the period pane and the 82-frame period rail: 12-frame entry, 60-frame hold, 10-frame exit. |
| FINAL | After all accepted footage and complete replays. Show both teams and the verified final score. | Separate closing section of at least 150 frames. Result first, then statistics. Never cover the last replay. |

The scorebug target is about 540×56 pixels at `(96, 54)`.
The event rail sits below the action, with a target width of 880–1100 pixels.
Its exact position must pass real-footage review for the camera view.
The Pass 2 vertical rail uses x=96, width=840, and bottom=430 at 1080×1920.
These margins require platform-interface review before release.

## Information and event arbitration

Use four layers: footage, scorebug, one information rail, and full-frame presentation.
Full-frame presentation temporarily hides the scorebug and rail.
GOAL, penalty, and explanation cards do not replace the scorebug after the rail pass is complete.

Rail priority is officiating result, GOAL, penalty, then replay explanation.
A higher-priority event exits the old card before entering the new card.
Use an eight-frame exit and a four-frame gap. Never draw two rail cards together.
Queue an event only when it retains enough readable time in its own source segment.
Period panes cover frames 12–17, with the source cut at frame 15. The graphic moves to the existing cut.
Replay panes keep all source frames. A period pane replaces a replay pane when they share a cut.
Both panes cover the scorebug and the event rails while they are on screen.
Period rails use the accepted rail margins in each layout. Omit a rail that conflicts with an event card.
The current mix still places period sound at its existing cue. Flag this difference until audio integration passes review.
Otherwise, omit the card and record the reason.
A GOAL card can continue into its own replay. It cannot continue into an unrelated play.
Reserve FINAL before scheduling the rail. FINAL cannot shorten a replay.

## Goal and decision states

Keep source game time separate from edited output frames.
The renderer receives the presentation schedule. It does not infer the game state.
A whistle confirms presentation timing only; it does not establish that a goal occurred.

1. Bind a goal to a verified sheet record or a source-bound human review.
2. Use a matched post-goal whistle to start the GOAL card and sound.
3. When the whistle is unavailable, omit the announcement and flag it.
4. Hold the old score through that goal's Replay. Keep the frozen goal in FINAL and later chronological play.

Use a score ledger for counted, overturned, and corrected goals.
An explicit correction can reduce a score. Replay footage cannot change the ledger.
An omitted source goal still contributes to the next chronological score state.

Officiating decisions require a source-bound entry with a result, reason, status, and penalty disposition.
Supported presentation includes `NO GOAL`, `GOAL STANDS`, `GOAL OVERTURNED`, and `GOALTENDER INTERFERENCE`.
Show `NO PENALTY ON THE PLAY` only when that disposition is confirmed.
Do not infer these calls from whistles, player gathering, or the absence of a sheet goal.
An unresolved review can show `UNDER REVIEW` only when the review itself is confirmed.

## Names, statistics, and clocks

Keep the home and away data order stable. Labels must make that order clear.
Do not add an apparent running clock when only sheet times are available.
Verified sheet time can appear as a static event label. Output elapsed time is not game time.
Show power-play status, shots, saves, and penalty timers only when the inputs support them.

Flag OCR artifacts and self-assists for review. Do not silently rewrite names with a broad suffix pattern.
Amateur jersey numbers and legitimate name suffixes require league context, not an invented universal restriction.
Use a number or team-only fallback when the player name remains unverified.
Fit names with measured width, line wrapping, or an approved initial form. Do not silently cut a surname.

FINAL shows both teams. Omit empty stat headers and unknown goalie values.

Pass 4 (accepted, v4):

- The result plate comes first and lifts to the top when statistics follow.
- Each team table lists at most 10 skaters with a goal, an assist, or PIM. Sort by points, then goals.
- A team with no recorded skater stats shows a dark faded "NO STATS RECORDED" panel.
- One GOALTENDING table lists both teams: SA, GA, SV, SV%. Values come from the Game Sheet (with overrides) only. Saves come from SA minus GA. SV% shows only when SA is known.
- Shots on goal and goalie identity come from the sheet or `game_sheet_overrides.json`. Deriving shots from video is follow-up work.
Display whole-number PIM without a decimal; retain valid fractional durations when applicable.
Use a longer closing section or a second page when necessary to show every eligible verified row.
Do not claim a five-second card can show an unlimited roster at a readable size.

## 3D and sound direction

Preserve the accepted dark studio, ice plane, hockey stop, frost cover, and logo breakthrough.
The ice plane is black ice. The reveal uses the blade cut, and the kit line colour lights the cut and the crest rim.
Refine these through separately reviewed previews rather than adding effects without a purpose.
Use a raking key light, controlled rim light, and one motivated specular sweep.
Keep the crest colors readable. Frost must have visible growth, material depth, and a clear break.
Reflection must follow the ice roughness and viewing angle. It must not resemble a flat duplicate.
Match the transition pane to the opening material. Keep the live scorebug visually quiet.

Sound follows the blade, stop, frost break, and exit. The readable hold stays quiet.
Use one related sound family for score changes, rail cards, replay, period, and FINAL.
The current generated sound is a sketch. Approved music and cues remain an acceptance requirement.
Preview audio must identify placeholders. Do not describe them as approved assets.

## Vertical layouts

Use the same Remotion components and event data with a dedicated 1080×1920 layout.
The vertical scorebug uses a centered 696×73-pixel plate at y=220, with a separate replay tab.
Do not stretch a 16:9 composition or draw a second OpenCV graphics package.
Keep the hook on action. Adapt the scorebug, event rail, replay tab, and FINAL for mobile reading.
Validate safe margins against the intended platform interface before release.
Do not hide the scorer or puck behind captions.

## Acceptance

- Review each new preview on actual game footage at normal playback speed.
- Read scores, periods, names, and decisions without pausing.
- Keep the scorebug visible during replay and event cards, except full-frame presentation.
- Verify every score change against the ledger. Verify every call against its source record.
- Keep the logo and matchup readable throughout the opening hold.
- Use no rail collisions, clipped names, empty stat headers, or FINAL overlap with footage.
- Verify contrast, safe margins, long names, both team colors, overtime, and corrected decisions.
- Check frame count, frame rate, full decode, and local-workspace publication before sharing each preview.
- Obtain user acceptance before calling the package polished or broadcast-ready.

Source review: `RecapGraphics.jsx`, `Stingers.jsx`, `StingerEffects.jsx`, `SkateCutAccent.jsx`,
`recap_graphics.py`, `audio_mix.py`, and `docs/recap-graphics.md`.
The cached full Recap uses older opening timing. It does not represent the accepted v4 opening preview.
