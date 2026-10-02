# Plain Recap

`hockeyrecap run <game_folder>` now includes the `assembly` stage after Selection.
`hockeyrecap rerun --from assembly <game_folder>` renders the Recap again.
Earlier stages still follow the normal cache checks.

The stage writes these files in the Game Folder:

- `YYYY-MM-DD_Team-vs-Opponent_Recap.mp4`: 1920×1080, 30 fps, H.264.
- `recap_assembly.json`: source cuts, output frame counts, speed, and review flags.

The plain Recap is silent. Graphics, replays, framing, music, and non-goal plays belong to later stages.
It can be much shorter than three minutes. Its maximum duration is four minutes.

Each selected goal uses its scoring camera. The default live speed is 1.1×.
Build-up and celebration follow the tiers in spec 002 §5.8, counted by selected goals.
Selection finds an action moment, not an exact puck crossing.
The cut retains three seconds of goal action, then the tier's celebration allowance.
Recording coverage can trim either end; the clip is flagged when that happens.
The four-minute limit shortens build-up first, never the three seconds of goal action.
If the cap still cannot be met, the stage fails instead of dropping goals.
A clip whose chapter decodes shorter than its manifest duration is flagged and omitted.

Unmatched goals remain flagged and have no clip. The stage does not invent replacement footage.
An older Recap with a different name is kept and flagged, not deleted.
Review the video before publication, particularly its goal cuts and unmatched goals.
The render replaces an existing video only after ffprobe confirms its duration.
