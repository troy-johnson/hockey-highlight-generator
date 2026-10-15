# Short and single-goal clips

The `short` stage follows the Recap graphics stage. It reads Selection, the Game Sheet, chapter manifests, ROIs, and PA music spans.

```bash
hockeyrecap rerun /path/to/game --from short
```

The stage writes:

- `short_assembly.json`: source cuts, frame counts, motion fit, tracking, graphics data, flags, and output names.
- `short_audio.json`: Bed identity, PA mute count, duck events, and measured encoded loudness for each output.
- `*_Recap_short.mp4`: a 1080×1920, 30 fps Short with a two-second hook, three to five plays, and a five-second FINAL.
- `*_Recap_goal_1.mp4` through `*_Recap_goal_3.mp4`: separate selected goals with their tight replays and FINAL.

The Short ranks covered plays by interest, then places them in game order. A configured Focus Team excludes opposing goals and penalties. Net plays use the period's net mapping to infer the attacking team, with a flag. Plays without a team mapping are flagged and excluded from a focused Short. The Recap perspective option does not override the Focus Team's Short selection.

The offline camera measures optical-flow spread at five samples per second. It uses a 9:16 window when at least 85% of samples fit. Otherwise, it tries 4:5, then 1:1. These windows remain inside the vertical output. Samples without measured action count as uncertain full-width samples. Tracking uses preceding and following samples. The cache is in `.recap_cache/short_motion/`.

Goal replays use the camera's goal ROI. Native half-speed replay requires at least 60 fps, as in the Recap. Lower rates use the existing optional RIFE stack. Missing RIFE produces duplicate frames and a flag. Missing ROIs produce a flagged full-height replay.

The Cue Library supplies one approved Bed. The stage trims the intro before its first measured beat. It loops whole bars and rounds cumulative cut times to output frames. The separate Short rotation file is beside the manifest. Rotation is saved after all outputs succeed.

Rink audio runs approximately 12 dB below the Bed. PA music mute windows apply before mixing. The Bed ducks at goal moments. The stage uses two-pass loudnorm, then measures the encoded AAC track. The targets are −14 LUFS and at most −1 dBTP. Any remaining deviation is flagged.

Missing approved music produces a flagged synthetic placeholder. This output does not verify approved-music acceptance. Missing goalie saves are omitted and flagged. Uncovered goals remain in the score and FINAL data, but do not become video cuts.
# Saved play reviews

Place `play_review.json` in the Game Folder to retain non-goal reviews across reruns.
Use `schema_version: 1` and a `plays` object keyed by Selection play ID.
Each review requires `status`: `keep`, `reject`, or `uncertain`.
The required `source` object copies `candidate_id`, `detection_s`, `primary_cam`, `start_s`, and `end_s` from the reviewed Selection play.
Changed sources stop rendering until the review is updated.
The optional `reason` field records the review decision.
Camera changes require new source boundaries and are not supported in this review file.
Recap and Short exclude rejected and uncertain plays, including their replays.
Net plays require an explicit `keep` review in both outputs.
The file does not change the Game Sheet or overwrite detector output.
The runner includes this file in assembly and Short cache fingerprints.
