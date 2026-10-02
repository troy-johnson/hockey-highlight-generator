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
Build-up and celebration follow the goal-count tiers in spec 002 §5.8.
Selection finds an action moment, not an exact puck crossing.
The cut retains three seconds of goal action before its celebration allowance.
Selection bounds can limit both allowances.
The four-minute limit reduces context when necessary, while keeping every selected goal.

Unmatched goals remain flagged and have no clip. The stage does not invent replacement footage.
Review the video before publication, particularly its goal cuts and unmatched goals.
The render replaces an existing video only after ffprobe confirms its duration.
