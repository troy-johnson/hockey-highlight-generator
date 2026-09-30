# Hockey Highlight Generator — Domain Glossary

The pipeline that analyzes GoPro hockey footage, detects high-action moments, and assembles a multicam reel in DaVinci Resolve for human verification and finishing.

## Pipeline Stages

**Discovery**: Scanning a game folder for GoPro chapter files in `cam1/` and `cam2/` and ordering them alphabetically by filename.
_Avoid_: Chapter scanning, File listing

**Sync**: Aligning the two camera streams to a common game-clock origin using GoPro timecode metadata, with `creation_time` fallback. Produces `sync_info.json` with the offset in seconds.
_Avoid_: Time alignment, Camera matching

**Detection**: Running optical-flow (Farneback) over net and slot ROIs on both synced streams, computing a rolling P95 score, emitting events into `events.csv` with start/end/score/color/angle. Audio is disabled.
_Avoid_: Event extraction, Highlight finding

**Selection**: Choosing the plays for a Recap and a Short. Goals come from the Game Sheet, and the video locates each goal's moment. Non-goal plays are added only when they are likely worth keeping, ranked by one interest score.
_Avoid_: Filtering, Ranking, Culling

## Reel Concepts

**Candidate Pool**: All events detected by the pipeline for a game, typically 10–20 min of footage. Exists before selection runs.
_Avoid_: Raw events, All clips

**Verify Timeline**: The ~5–6 min multicam timeline assembled in Resolve for human verification. Lives on Track 1 with angles pre-switched. The user deletes junk and corrects angles here at ~2x speed (~3 min).
_Avoid_: Review reel, Draft timeline

**Next Best**: The short list of plays that Selection ranked just below the cut. The review page's "promote" draws from it. It replaces the Reserves timeline.
_Avoid_: Reserves, Bench

**Play Type**: What kind of play a clip shows: goal, chance, save, defense, hit, penalty, or fight. Set by Selection, corrected in review with one tap, and used as a training label.
_Avoid_: Category, Event class

**Keeper**: An event that survives verification — not deleted, possibly angle-corrected, and annotated. Only Keepers receive dress-up (slo-mo, graphics, callouts).
_Avoid_: Selected clip, Good clip

**Recap**: The main published video for one game — a condensed highlight in the style of an NHL-channel game recap, with broadcast graphics. Its length follows how much interesting action the game had: about 3 min for a normal game, never more than 4 min.
_Avoid_: Final Reel, Output video, Highlight video

**Short**: One vertical (9:16) video per game containing the top 3–5 most interesting plays — usually goals, but penalties, fights, or big saves qualify. Published to Reels / Shorts / TikTok.
_Avoid_: Reel, Clip, Social video

**Cue**: A short piece of music or a sound effect tied to one event type — Game Start, a penalty sting, a fight sting, the goal horn, the Win / Loss / neutral close, or a graphic's sound. Which Cue plays depends on the event and on whose event it is.
_Avoid_: Song, Music drop

**Bed**: Low music under the play — one track per period in a Recap, one track for a whole Short. It ducks under Cues and goals.
_Avoid_: Background music, Soundtrack

**Cue Library**: The hand-approved set of Content-ID-safe tracks and sound effects the pipeline may use, each with its source and license proof. The pipeline rotates through it; changing a track means editing the library.
_Avoid_: Music folder, Playlist

**Comic Call**: A Focus Team penalty the user marks in review as a bad or funny call. It gets the comic penalty sting instead of the regular one.
_Avoid_: Bad call, Joke penalty

**Team Config**: A team's identity — name, short name, colors, optional logo, and Roster. Full for the user's own teams; often just a name for opponents. Without a logo, the team name stands in as a styled wordmark.
_Avoid_: Branding, Theme

**Perspective**: Whose story a Recap tells — Neutral, or focused on one of the two teams. Chosen per game.
_Avoid_: Mode, Side, Bias

**Focus Team**: The team a focused Recap is about — usually one of the user's own teams (Ice Pak or another team they play on). Its brand leads the open, stingers, and watermark; its players fill the FINAL card and the Short. The default when exactly one team in the game is one of the user's teams.
_Avoid_: Home team, Our team, Primary team

**Scoresheet**: The official paper game sheet — photographed or scanned and dropped into the Game Folder. Its jersey numbers are not always correct.
_Avoid_: Game sheet (for the paper), Box score

**Game Sheet**: The structured record transcribed from the Scoresheet — each goal (period, clock time, scorer, assists) and penalty (period, clock time, player, infraction). Treated as a claim to verify against the video, not as ground truth.
_Avoid_: Annotation, Scoresheet data

**Roster**: A team's jersey-number → player-name mapping. Pre-seeded for Ice Pak and known opponents; overridable per game because players forget jerseys and subs appear.
_Avoid_: Lineup, Player list

**League**: The competition a game belongs to. Fixes the game rules every game in it shares: clock mode (running or stop time), whether Scoresheet times count elapsed or remaining, number of periods (usually 3), period length, the break between periods (usually about 1 min, no resurface), penalty lengths, and how a tie ends (overtime, shootout, or tie) — which may differ between regular season and playoffs. Teams switch ends every period in all Leagues.
_Avoid_: Division, Season

**Sub**: A rostered, part-time player on a team. Listed on the Scoresheet like any other player when present.
_Avoid_: Spare, Fill-in

**Unlisted Player**: Someone who plays in a game but is not on that game's Scoresheet player list; their goals or assists may be credited to a listed player. Graphics credit the person who actually played, as confirmed in review.
_Avoid_: Sub, Ringer, Guest

**Score Timeline**: The derived sequence of score changes computed from the Game Sheet (goal events increment the correct team's score in chronological order). Drives the score-bug graphic, not entered manually per change.
_Avoid_: Score log, Score history

**Replay**: A second instance of a goal clip played at 50% speed immediately after the full-speed version, keyed off the REPLAY marker in `markers.csv`.
_Avoid_: Slow-mo, Slo-mo replay

## Resolve Integration Patterns

**Live Control**: Running a Python script inside Resolve (Workspace → Scripts) that manipulates the open timeline directly via Resolve's scripting API. Used for marker import and reel assembly. Only available in Resolve Studio.
_Avoid_: Resolve script, In-app automation

**Interchange**: Exporting markers or edit data from Resolve into a portable format (FCPXML, EDL, CSV) that can be consumed by external tools, or importing external data into Resolve. The free-Resolve-compatible path.
_Avoid_: Export/import, File exchange

**In-Resolve Script**: A script that runs inside Resolve's scripting environment (not an MCP call). The current pattern for `compile_reel.py` and marker-expand tools. Works in Free for read-only operations; write requires Studio.
_Avoid_: Resolve Python script

## Camera and Angle

**Multicam Clip**: A Resolve clip that contains both camera angles synced by timecode, allowing instant angle switching during playback or editing. Created automatically by `compile_reel.py` from all imported chapter files.
_Avoid_: Multicamera, Synced clip

**Primary Camera**: The angle (cam1 or cam2) assigned to an event based on which side of the rink the action occurred — currently by optical-flow magnitude, eventually by puck-proximity via HockeyAI. Stored in `events.csv` as `primary_cam` (1 or 2).
_Avoid_: Best angle, Active camera

## Configuration and Output

**ROI** (Region of Interest): The net and slot rectangles defined per-camera in `rois.json`, used to constrain optical-flow computation. Picked interactively on first run; varies by rink camera position.
_Avoid_: Region, Detection zone

**Game Folder**: One game's folder, named `opponent_MMDDYYYY` (the name supplies the default opponent and date). The user copies both cameras' card contents and the Scoresheet photo into it, flat; the pipeline sorts files into cameras and Recordings itself and writes its outputs and the per-game options file next to them.
_Avoid_: Project folder, cam1/cam2 layout

**Recording**: One continuous capture by one camera, from start to stop. The camera splits it into Chapters (files); a camera that is stopped and restarted makes a new Recording, with a time gap between them.
_Avoid_: Clip, Session

**Concat Manifest**: The `cam1_concat.txt` / `cam2_concat.txt` files in ffmpeg concat format that list chapter files in order, allowing detection to run on the full game without pre-stitching.
_Avoid_: Chapter list, Merge list
