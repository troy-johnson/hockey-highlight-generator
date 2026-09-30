# Spec 002: Automated Recap and Short from a Game Folder

Date: 2026-09-29. Status: Draft for handoff.
Source: Wayfinder map "Automated NHL-style Recap and Short from a Game Folder" (Beads `hhg-3r5`). Every decision below links to the closed ticket that holds its detail and evidence. The glossary is [CONTEXT.md](../../CONTEXT.md). Decisions that are hard to reverse are in [docs/adr/](../adr/).

## 1. Purpose

One command turns a Game Folder into a finished **Recap** and a **Short**, with one human review before publication. The Recap is 16:9, about 3 min, and never more than 4 min, in the style of an NHL-channel game recap. The Short is 9:16 and has the top 3–5 plays. Human time must be as close to zero as possible. Play quality must not matter: every game gets the same treatment.

**Out of scope for this spec:**
- automatic upload to YouTube or social platforms
- commentary, including AI voice
- the product and commercial question (`hhg-x56`)

## 2. Inputs and outputs

**Inputs**
- **Game Folder:** a flat folder named `opponent_MMDDYYYY` holding both cameras' card contents and a photo of the Scoresheet. The pipeline sorts files into cameras and Recordings by itself. (`hhg-3r5.20`)
- **Team Configs and League configs:** stored outside the repo (for example `~/hockey/`), because they hold personal data. (`hhg-3r5.8`)
- **Cue Library:** hand-approved, Content-ID-safe music and SFX, with a manifest. (`hhg-3r5.5`, `hhg-3r5.15`)
- **Per-game options file:** created by the first run. (`hhg-3r5.20`)

**Outputs** (written into the Game Folder on approval; nothing is uploaded) (`hhg-3r5.9`)
- `YYYY-MM-DD_Team-vs-Opponent_Recap.mp4`
- the compilation Short
- up to 3 single-goal clips
- a text file with a paste-ready YouTube title and description

## 3. Capture assumptions

- **Cameras:** two GoPro HERO12 Black behind the nets, 4K120 16:9 Wide on Contacto external power. The fallbacks are 4K60, then 2.7K60, and the pipeline accepts any mode. (`hhg-3r5.12`, `hhg-3r5.13`)
- **Aim:** keep the current low aim, which keeps the near net sharp. The far end is out of frame. (`hhg-3r5.25`)
- **Side camera:** a side (bench glass) camera is possible, and its spot changes from game to game. (`hhg-3r5.26`)
- **Scheduled recording:** recording is scheduled at game start, so the warm-up may or may not be in the footage. (`hhg-3r5.21`)
- **Bench mic:** optional. A DJI Mic Mini records only through its receiver into a phone (USB-C, 48 kHz / 24-bit). (`hhg-3r5.17`)

## 4. Command and configuration (`hhg-3r5.20`)

**Command:** `hockeyrecap run | status | review | rerun --from <stage>`.
- The pipeline runs unattended, and its stages are resumable.
- `hockeydetect` (`run_detect.sh`) keeps working until `hockeyrecap` replaces it.
- A watcher on the NAS folder comes later.

**Options** come in four layers, each overriding the one before:
1. League: clock mode, time direction, number of periods, period length, break, penalties, tie rules.
2. Team Config: name, short name, colors, logo or wordmark, Roster, and whether the team is one of the user's own.
3. The per-game options file. Inferred values are marked as inferred:
   - League, opponent, and date (from the folder name)
   - Perspective and Focus Team
   - live-play speed (default **110%**, amended in `hhg-3r5.11`)
   - cold open or start on play
4. CLI flags, for one run only.

The review page writes corrections back to the per-game file.

**Progress and status**
- The terminal shows a stage list with checkmarks, a progress bar with elapsed time and an estimate of time left, and the flags found so far (Python `rich`, MIT license).
- The run writes a status file in the Game Folder, which `status` reads.
- The full log goes to a file.
- When the run is done, a macOS notification opens the review page.

**Failure policy:** the run stops only when there is no footage or no usable camera. Everything else becomes a flag in review.

## 5. Stages

### 5.1 Discovery and camera identity (`hhg-3r5.20`, bug `hhg-38a.13`)

- **Camera identity:** from the `CASN` serial or `CAME` id atom near the end of each MP4. Verified on the Ghost Pirates game.
- **Recordings:** chapters of one Recording share the 4-digit file number. A camera that stopped and restarted has more than one Recording, each placed on the game timeline by its start time and audio.
- **Black recordings:** near-black Recordings (lens covered) are flagged and skipped.
- **Camera position:** detected for each camera and each game, from HockeyAI on a median background: end camera (one large goal frame) or side camera. (`hhg-3r5.26`)

### 5.2 Sync (bugs `hhg-38a.11`, `hhg-38a.12`)

- **Timecode is only a starting guess.** GoPro timecode is each camera's own clock. On Ghost Pirates, the cameras were 20 s apart while their timecodes were 1 frame apart.
- **Offset and drift from audio:** measure them from the two cameras' rink audio (onset-envelope NCC), with a confidence grade. Drift is 50–80 ppm between cameras.
- **Flag:** a disagreement over about 0.5 s is flagged.
- **Seeking:** apply offsets with `ffmpeg -ss` before the concat input. **Never use concat `inpoint`.** On GoPro HEVC it applies only about 1/3 of the offset.
- **Bench mic (optional):** synced to both cameras by onset search, 8-s GCC-PHAT windows, and a Theil–Sen drift fit. Grade: synced / check / failed. If failed, the Recap uses GoPro audio only. (`hhg-3r5.17`)

### 5.3 Coverage and period structure (`hhg-3r5.21`, `hhg-3r5.25`)

- **Coverage:** for each camera, its Recordings minus black or blocked stretches, mapped to the net it covers in each period by the end-switch rule. The review page shows a coverage strip.
- **Period breaks:** a quiet stretch at both nets of about 60 s or more, confirmed by the goalies swapping nets (jersey color in the crease of the goal frame), guided by League timing.
- **Game start and end:** the start is the first center-ice faceoff after an optional warm-up. The end is the last whistle, or the end of the Recording.
- **Uncertain breaks:** flagged. The user picks another candidate from a list.

### 5.4 ROIs and detection signals (`hhg-3r5.19`, `hhg-3r5.20`, `hhg-3hk.1`)

- **ROIs:** automatic on every game. Build a median background per camera, detect the HockeyAI goal frame (found in 100% of frames on both test cameras), and derive the slot by geometry. There is no manual ROI picker and no per-rink setup. The review page shows the overlay only when confidence is low.
- **Optical flow:** Farneback on a padded crop around the ROIs, with VideoToolbox decode + `scale_vt`, and both cameras in parallel. On Ghost Pirates this took 25.5 min against 111 min before, with the same events (correlation 0.9997).
- **HockeyAI (YOLOv8m):**
  - Run it only inside Selection windows and on median backgrounds, not across the whole game.
  - Use the goal frame and players only. Do not use the stock puck class: it had 18% recall near goals.
  - The Ultralytics runtime is AGPL-3.0, while the weights are MIT. Plan an ONNX export.
- **Audio signals:** whistles and stoppages. PA music is found with YAMNet and muted in rink audio (about 25 s per game). (`hhg-3r5.18`)

### 5.5 Scoresheet and Game Sheet (`hhg-3r5.4`, `hhg-3r5.8`, `hhg-3r5.3`)

- **Transcription:** Claude vision with a JSON schema, then hockey consistency checks that flag rows. The Game Sheet holds goals, penalties, player lists, the header, goalie saves, and PIM.
- **Order-based goal matching** with soft clock windows. Running time converts after the period starts are detected. With stop time, the clock only bounds the order.
- **Scorer check:** jersey-number recognition verifies only the claimed scorer. It gives five review cases. Assists and penalties come from the Game Sheet only.

### 5.6 Selection ([ADR-0003](../adr/0003-game-sheet-first-selection.md), `hhg-3r5.14`)

**Goals**
- The Game Sheet decides which goals exist, and the video locates each goal's moment.
- The search window comes from the period, clock time, team, and the end-switch rule, at one net.
- **Goal signature:** a flow peak at that net, a whistle, a long stoppage, then a faceoff at center. "Puck inside the goal frame" is an optional cue, because the puck often bounces out.
- v1 uses a weighted rule. A learned model replaces it once enough labeled games exist.
- **No convincing moment:** the goal is flagged "no clip found".

**Non-goal plays**
- **Automatic:** plays at the net (chances, saves, near misses, defensive stops at the net).
- **Penalties:** the infraction clip is automatic on a confident Game Sheet time match with a whistle just after it.
- **Hits and fights:** go to the Next Best list only, until a model trained on review labels reaches 80% precision.

**Interest score:** one score drives cutting, the fill, the best-goal full treatment, the cold open, and the Short. For goals, game context comes first (game-winner, tying goal, OT or shootout, last 2 min, Focus Team).

**Targets**
- Every sheet goal has a clip or a flag.
- At least 90% of goal clips place the goal moment correctly without correction.
- At least 80% of added non-goal plays are worth keeping.

**Shootouts:** single-skater attempts after the last period, all included in order. The winner comes from the sheet.

### 5.7 Camera angles and framing (`hhg-3r5.22`, `hhg-3r5.25`, `hhg-3r5.26`)

**Live angle**
- A goal uses the camera behind the scoring net, confirmed by flow.
- On a rush, start on the far camera and cut to the scoring camera at the handover, with each shot at least 2 s.
- A non-goal play uses the camera at the net where it happened.

**Framing:** every live clip gets an automatic framing of about 1.15–1.3× from 4K, with the horizon levelled from the boards line and the fisheye edges cropped out. Goals also get a slow push-in.

**Replays:** both shots come from the scoring camera: a wide shot, then a tight punch-in on the goal frame box. Slow motion uses native 120 fps if captured, otherwise RIFE (rife-ncnn-vulkan) on the replay block only. (`hhg-3r5.6`)

**Uncovered net**
- A goal at a net with no camera gets a 3–4 s graphics-only 3D goal card, after any visible build-up. There is no replay.
- These goals are left out of the Short, the best-goal treatment, and the cold open.
- A side camera replaces the card when it sees that net.

### 5.8 Recap edit grammar (`hhg-3r5.11`, `hhg-3r5.1`)

**Open**
- The default is a 2–3 s Focus Team 3D stinger.
- The options are a cold open on the best play, or start on play.

**Period change**
- A ~1 s snow-on-the-lens wipe (see §5.9).
- Then a ~2 s "2ND PERIOD" + score card over the faceoff.

**Per-goal structure:** build-up → goal → GOAL card over the celebration → dissolve → replay block → dissolve. Budget tiers by goal count:

| Goals | Build-up | Celebration | Replay | Total per goal |
|---|---|---|---|---|
| 1–6 | 6 s | 3 s | 2 shots, ~8 s | ~20 s |
| 7–12 | 5 s | 2 s | 1 shot, ~5 s | ~14 s |
| 13+ | 4 s | none (card over the replay) | ~4 s | ~10 s |

The 2–3 best goals always get the full treatment.

**Goal cutting**
- Never cut: the first goal, a tying or lead-change goal, the game-winner, OT and shootout goals, or the 2–3 best goals.
- Otherwise a goal may be cut only when its score is low and the game has 10 or more goals.
- A cut goal is still shown by a score flap during the next play.

**Non-goal plays:** 5–10 s at play speed. Only the best one gets a short replay. Penalty plays carry the penalty card.

**Length:** about 3 min, and never more than 4 min. If goals alone exceed the cap, shorten the replay blocks first.

**Speed:** live play at 110% by default. This is the user's house style; broadcasts use 1.0×.

**Close:** a FINAL card of about 5 s, with both teams' G / A / PTS / PIM and goalie saves.

**Perspective**
- **Neutral**, or **Focus Team**. Focus is the default when exactly one team is the user's.
- Opponent goals get a compact card.

### 5.9 Graphics (`hhg-3r5.2`, `hhg-3r5.10`, `hhg-3r5.16`)

**Engine:** Remotion (React) renders ProRes 4444 alpha clips from props written by Python. ffmpeg assembles the cuts, replays, overlays, and audio. Resolve is only an optional finisher.

**Package:** Variant D, "Ice Pak Slab".
- semantic color tokens
- team primary and secondary colors, a square logo or wordmark, and an acronym
- a mirrored scorebug
- GOAL, penalty, period, and FINAL cards

**Stingers (3D, three.js via @remotion/three)**
- **Open:** a slab hockey-stops over a 3D regulation rink, with the ICEPAK/HOCKEY lockup and frost.
- **Period:** snow on the lens over the footage, with the crest.

**Quality rules**
- Soft, motion-blurred particles only.
- Mist is never clipped by depth.
- Render at 2× and downsample.
- Attached parts share the parent's geometry and frost.

**Prototype:** branch `prototype/graphics-package`, draft PR #15.

### 5.10 Audio (`hhg-3r5.15`, `hhg-3r5.5`, `hhg-3r5.18`)

**Recap**
- A low Bed for each period, cut on bars and changed under the period wipe, 18–20 dB below full level.
- Cues: a Game Start hit; a Win, Loss, or neutral close; and 2–4 s stings.
  - regular penalty sting
  - Comic Call (a review toggle)
  - power play
  - fight
  - one neutral sting in Neutral Perspective
- A horn on Focus Team goals only, and on all goals in Neutral.
- SFX for each graphic.

**Short:** one beat-synced Bed, with rink sound about 12 dB under it.

**Cue Library**
- Tracks come from the YouTube Audio Library and from Pixabay tracks without the Content ID flag. SFX come from Freesound CC0.
- Beat grids come from `beat_this`.
- The pipeline rotates tracks between games.

**Mix**
- Scripted ducking from the event timeline.
- Two-pass loudnorm to −14 LUFS, −1 dBTP.
- PA music in rink audio is muted in the spans YAMNet finds.

**Safety net:** a private upload before publishing, because buried music (10 dB or more under the rink sound) is not detected.

### 5.11 Short (`hhg-3r5.23`, `hhg-3r5.7`)

**Layout for each play** comes from how wide the motion spreads:
- full-screen 9:16 tracking, if the motion fits for at least 85% of the play
- otherwise a 4:5 window
- otherwise a 1:1 window, inside the 9:16 frame

The virtual camera is our own, with smoothed offline tracking.

**Bands:** a scorebug at the top, and a caption at the bottom for goals and penalties only.

**Structure:** a 1–2 s hook, then 3–5 plays in game order, then a FINAL card, in 30–60 s. A goal gets a tight goal-box replay in 9:16.

**Extra output:** up to 3 single-goal clips. The user picks which to publish.

### 5.12 Review (`hhg-3r5.9`)

**Surface:** a local web page. Two stages, always both:
1. **Data review** (about 1–2 min): every goal as a thumbnail row with flags first, plus the inferred options, the coverage strip, the period starts, the ROI overlay when unsure, and the sync flags.
2. **Watch and approve** the rendered Recap and Short.

**Allowed changes**
- names
- accept or reject a goal
- remove or promote a play (from Next Best)
- change the camera angle
- set the Play Type (one tap)
- the Comic Call toggle
- pick another period break

There is no trimming. Corrections are saved to the per-game file and the Roster, and they are logged as labeled data.

## 6. Performance budget (`hhg-3r5.19`, `hhg-3r5.24`)

**Budget:** about **1 hour** of unattended machine time per game, on an M3 Pro.

**Measured on Ghost Pirates**

| Stage | Minutes |
|---|---|
| Detection | 25.5 |
| HockeyAI at 1 fps (whole game) | ~13 |
| Remotion graphics | ~9.5 |
| Assembly | ~2.5 |
| Short | ~1 |
| RIFE replays | ~1.4 per goal |

**Total** for a 10-goal game at 60 fps: about 65 min. Two changes each bring it under the budget:
- HockeyAI scoped to Selection windows (about −10 min)
- 4K120 capture, so no RIFE (about −14 min)

## 7. Fix first (known defects, V3 epic `hhg-38a`)

1. **P1 — concat `inpoint` applies only about 1/3 of the sync offset** (`hhg-38a.11`). Fix: `-ss` before the concat input. Verified.
2. **P1 — sync trusts GoPro timecode without checking it** (`hhg-38a.12`). Fix: offset and drift from audio.
3. **P2 — separate Recordings are joined as one timeline, and black Recordings are analyzed** (`hhg-38a.13`).

## 8. Test data and acceptance

**Answer keys:** each is built from a published Recap's EDL plus the raw footage. The EDL source time is mapped to raw time by frame matching: SIFT/RANSAC undoes the punch-ins, and background-subtracted thumbnails do the matching. The user's edits run at 1.0995×.
- **Ghost Pirates (2025-11-09):** 20 clips and 15 goals. It is the first answer key.
- **Next:** build 10–15 more (`hhg-3hk.11`).

**Baseline on Ghost Pirates, with correct sync:**
- 12 of 15 goals are inside a detected event.
- Goals rank from 8th to 117th of 124 events by flow score.
- The goal-timing target above (90%) is measured against these keys.

## 9. Backlog that follows this spec

| Item | Ticket |
|---|---|
| Puck-in-goal-frame cue trained on behind-the-net footage | `hhg-3hk.9` |
| Post "ping" detection in rink audio | `hhg-3hk.10` |
| Side-mount test | `hhg-3hk.12` |
| Existing Phase 2 items: HockeyAI integration, automatic ROI (`hhg-3hk.3`), ML re-ranker | `hhg-3hk` epic |
| Verify the Scoresheet transcription on 10–20 real sheets | — (per `hhg-3r5.4`) |
| One real test game with the bench mic | — (per `hhg-3r5.17`) |
| Trial one game at 4K120 on external power | — (per `hhg-3r5.13`) |

**Research branches** (local, not pushed):
- `research/nhl-recap-format`
- `research/render-engines`
- `research/jersey-number-recognition`
- `research/scoresheet-transcription`
- `research/music-and-sfx`
- `research/slow-motion-replays`
- `research/vertical-reframe`
- `research/hero12-capture-and-power`
- `research/pa-music-detection`
- `research/bench-mic-sync`
