# Detection regression set

## Command

Run this command from the repository root with the project environment active:

```bash
python v3/scripts/detection_regression.py \
  --games-root '/Volumes/Partition_1/Ext. Projects/Ice Pak'
```

Alternatively, set `HHG_GAME_ROOT`. The runner processes every game listed in
`tests/answer_keys/suite.json`, in list order. Ghost Pirates is first.
The runner does not evaluate games without answer keys. Missing games produce blocked results.
The runner continues with the remaining games and returns exit code 2 if any game is blocked.

The default run uses valid cached signals only. It never extracts footage on a cache miss.
**Obtain approval before you add `--allow-processing`.** This flag permits footage extraction for missing Recordings.
The runner reuses valid cached Recordings and saves new caches outside the Game Folder.

The runner requires existing `rois.json` and both synchronized concat manifests.
It does not run discovery, sync, ROI selection, scoresheet processing, graphics, or assembly.
It calls the existing V2 detector with the detection defaults from `recap_options.py`.
An optional `detection` block in the suite file overrides these defaults for experiments.
Repeated `--set detection.KEY=VALUE` arguments override settings for one run without changing files.

## Matching rules

Each reference clip has a verified `reference_s`, camera, unique ID, and explicit goal label.
Times use the synchronized detection timeline, not the EDL source timeline or recap timeline.
The runner rejects camera mappings that differ from the answer key.

- An event contains a reference when `start_s <= reference_s < end_s`. There is no tolerance.
- Goal recall counts each labeled goal whose reference point lies inside any event, regardless of primary camera.
- A correct clip match requires containment and the same primary camera.
- Clip matches use deterministic maximum one-to-one matching. One event cannot match two reference clips.
- An unmatched event that contains a reference is incorrect: `wrong_camera` or `duplicate`.
- With a complete key, an unmatched event without a reference is incorrect: `no_reference`.
- With an incomplete key, an event without a reference is unjudged. It is not a false positive.
- If the key has no labeled goals, recall is `null`, shown as `N/A`.

Goal recall measures candidate coverage. It does not measure goal classification or whole-clip coverage.
A wrong-camera event can recover a goal for recall but cannot correctly match its clip.
The result lists missed goals, unmatched reference clips, and incorrect matches with their reasons.

## Answer keys and saved results

Answer keys live in `tests/answer_keys/`. Generated results default to the ignored
`.regression-results/` directory. `--suite` and `--results` select other locations.
The runner rejects result locations that overlap answer-key directories or Game Folders.
It never overwrites a Game Folder output or an answer key.

Each run creates a new directory containing `result.json` and per-game events,
markers, and detection logs. Earlier results remain unchanged.
The runner compares each game against its most recent successful saved result.
A blocked game does not replace its last successful result.

Comparisons include metric deltas, recovered and newly missed goals, gained and lost
clip matches, and runtime changes. They reject changed keys, rules, raw inputs, or camera timelines.
Detection settings and detector code may change: those are the changes this command checks.
Results record settings and detector and extraction hashes.
Runtime includes cache loading, detection, and evaluation. It excludes the original cache extraction cost.
Runtime deltas are unavailable when one run uses footage processing and the other uses cached signals.

Cache validity uses the existing extractor's keys: raw file identity, seeks, ROIs,
sample rate, width, audio mode, decode mode, and extraction code hash.
Damaged caches block default runs instead of silently starting footage extraction.
Approved processing bypasses rejected caches. The runner validates fresh arrays before evaluation or cache replacement.
Invalid fresh arrays block the game and leave its last successful result unchanged.

## Experimental peak candidates (`hhg-3r5.91`)

Use this command to check separate peak windows without changing production defaults:

```bash
python v3/scripts/detection_regression.py \
  --games-root '/Volumes/Partition_1/Ext. Projects/Ice Pak' \
  --set detection.preserve_peaks=true
```

This opt-in path keeps each detected peak separate, instead of merging overlapping windows across cameras.
It compares camera activity within one second of each peak.
Equal activity retains the camera that supplied the peak.
Duration limits center the shortened window on its peak.
The path skips start-time cooldown, which can remove another nearby moment.
`min_sep_s` still controls peak separation within each camera.
`merge_gap_s` and `cooldown_s` do not apply in this mode.
The marker columns remain unchanged.

The CLI can also test lower thresholds and shorter peak spacing:

```bash
python v3/scripts/detection_regression.py \
  --games-root '/Volumes/Partition_1/Ext. Projects/Ice Pak' \
  --set detection.preserve_peaks=true \
  --set detection.thresh_pct=90 \
  --set detection.min_sep_s=4
```

### Cached Ghost Pirates comparisons

| Settings | Goals found | Correct matches | Incorrect matches | Events | Cached runtime |
|---|---:|---:|---:|---:|---:|
| Unchanged default | 11/15 | 9 | 2 wrong camera | 122 | 0.296 s |
| Separate peaks, P95, 12 s | 11/15 | 11 | 0 | 161 | 0.069 s |
| Separate peaks, P90, 12 s | 14/15 | 14 | 2 duplicates | 198 | 0.017 s |
| Separate peaks, P90, 4 s | 15/15 | 15 | 8 duplicates, 2 wrong camera | 361 | 0.026 s |

Separate peaks recover the original wrong-camera matches for `away:1` and `away:10`.
P90 also recovers `away:2`, `away:7`, and `away:11`.
Four-second spacing recovers the remaining `away:6` reference.
The increased event count and incorrect matches prevent approval as a new default.
The experiments use warm signal caches. These runtimes do not include footage extraction.
One incomplete key cannot establish non-goal precision or performance across other games.

Next checks need more verified keys and better handling of duplicate moments.
Whistle-based candidates could recover weak goals without lowering the flow threshold across the whole game.
No production default, Game Folder output, answer-key label, graphics file, or scoresheet file changed.

## Cached whistle candidates (`hhg-3r5.92`)

Use the existing Selection candidate rules as a separate, goal-oriented experiment:

```bash
python v3/scripts/detection_regression.py \
  --games-root '/Volumes/Partition_1/Ext. Projects/Ice Pak' \
  --candidate-source whistles
```

The command reuses `selection.build_candidates` and its repeated-whistle grouping.
It finds flow evidence before each whistle and ranks cameras with the existing Selection weights.
Different stoppages remain separate, even when they occur close together.
The command reads existing flow caches, audio caches, coverage, and sync metadata.
It does not read the Game Sheet or extract new audio features.
Missing caches, invalid audio activity, and inconsistent timelines block the game.
Valid inputs with no whistle candidates report zero recall and list every verified goal as missed.
The adapter checks sync metadata against the verified manifest mapping, not only against other generated metadata.
Audio RMS weighting stays zero unless the caller explicitly overrides it.

The measured Ghost Pirates run found all 15 goals and 15 correct clip matches
in 44 candidates, with no incorrect matches and 29 unjudged candidates.
Cached runtime was 0.952 s, including metadata probes and cache reads.
Against the previous 361-event experiment, it removed 317 candidates and ten incorrect matches without losing labeled goals.

Each run keeps the original flow `events.csv` and `markers.csv` unchanged.
The adapter writes `whistle_events.csv` and `whistle_candidates.json` beside them.
The result records the evaluated file, candidate source, and hashes of the candidate code and metadata.
Comparisons identify the previous candidate source.
The whistle file is an evaluation artifact, not a replacement Resolve reel.

These results measure candidate coverage, not goal classification or final Selection accuracy.
The 29 unjudged candidates are not labeled goals or false positives.
Five published clips still lack available mappings, so the flow pool must remain available for non-goal selection.
Production defaults remain unchanged. More verified keys are needed before promotion.

## Published-clip intake

The intake command creates review drafts, not answer keys.
It parses EDL video cuts and retains audio cues and speed effects as evidence.
It does not infer goal labels from an EDL source name.
Compound `Game v1` cuts can contain logos, credits, or multiple camera views.

After approval, prepare discovery, confirmed sync, and ROIs in a separate analysis folder.
Link raw chapters there. Do not overwrite the original Game Folder outputs.
Use the downloaded Recap and independently verified published references for matching:

```bash
python v3/scripts/published_clip_intake.py \
  --edl '/Users/troyjohnson/Desktop/yetis_11022025.edl' \
  --recap '.regression-results/answer-key-intake/67MCRG0OGzk.mp4' \
  --game '.regression-results/answer-key-intake/games/yetis_11022025' \
  --output '.regression-results/answer-key-review/yetis_goals_02' \
  --frame-cache '.regression-results/answer-key-intake/frame_cache' \
  --references '.regression-results/answer-key-intake/yetis_published_goal_refs.json' \
  --allow-processing --refine
```

Use a new output directory for a new run.
Add `--resume` only to continue the same run with unchanged inputs, settings, and code.
The runner saves each coarse proposal before local refinement and retains completed clip checkpoints.
It rejects changed input identities and publishes the final draft atomically.
Its runtime covers the current invocation, not earlier extraction or interrupted work.

Static SIFT/RANSAC alignment estimates the crop. Moving-player comparisons check the camera time.
Local refinement decodes the relevant chapter, including chapter boundaries.
A strong score is still a proposal. Inspect paired published and raw frames before recording a verified reference.
Repeated player positions can produce false matches; use independent anchors to check the edit timing relationship.
Do not promote predicted times or generated detection choices to answer-key labels.

Published reference files bind their labels to the Recap's SHA-256.
Each reference requires a unique ID, a published time, an explicit goal label, and verification evidence.
The intake output protects source folders and the answer-key directory.
The command never writes keys or suite entries.

## Available coverage

The suite contains five mapped answer-key games with 39 verified goals and five verified non-goals across BLB and Yetis:

| Game | Verified goals | Limits |
|---|---|---|
| Ghost Pirates | 15 | Five published clips remain unmapped |
| Yetis | 9 | Two non-goals are verified; other non-goals and Play Types remain unlabeled |
| Flyers | 2 of 3 published goals | The third scoring-play reference is unverified |
| BLB | 3 | Three non-goals are verified; other non-goals and Play Types remain unlabeled |
| Blue Devils | 10 of 11 published goals | `icepak:7` remains unverified |

The Ghost Pirates key copies only the 15 verified goal references from its existing
`selection_validation.json`. It excludes generated selections, verdicts, and metrics.
The key records the source hash and source definition.

The Ghost Pirates Recap has 20 clips. Five clips lack available raw-time mappings,
so `clips_complete` is false. No labels were created for those five clips.
The reference points describe scoring-play frames, not exact puck crossings.

The Yetis key contains nine goals: eight Ice Pak goals and one Yetis goal.
Published GOAL banners establish the labels. Paired raw frames establish the camera and scoring-play reference times.
Seven references use `cam2`; two use `cam1` after the teams change ends.
The key records the EDL, Recap, and paired-image hashes.
It uses the audio-confirmed 9.813-second `cam2` seek.
Two continuing-play references now have verified non-goal labels. Other non-goals and Play Types remain unlabeled, so `clips_complete` is false.

Yetis preparation lives in the separate analysis folder, not the original NAS Game Folder.
The runner does not copy that metadata to the NAS.
The Flyers key has two verified goals out of three published goals.
The third remains excluded: its close camera is obstructed, and its published reference shows distant action.
That goal requires further timing verification. Flyers recall uses the two verified references, not all three goals.
The suite maps the dated Flyers key to the NAS folder named `flyers_121142025`.

The BLB key contains three visually matched goals, all from `cam1`.
Its EDL names the game as 2026-01-04; the suite maps it to the folder named `buzzlightbeer_01042025`.
Its confirmed audio sync uses a 1.331-second `cam2` seek.
Lower-contrast alignment narrowed the searches but did not supply labels. Paired-frame review verified each final reference.

The Blue Devils key contains ten visually matched goals, split equally between the two cameras.
It uses the audio-confirmed 13.151-second `cam2` seek.
The analysis copy excludes a verified covered-lens pre-game Recording; it leaves the original NAS metadata unchanged.
The seventh Ice Pak goal has a published score change but no verified scoring-play reference. The key excludes it.

Use this prepared root for the five-game validation:

```bash
python v3/scripts/detection_regression.py \
  --games-root '.regression-results/answer-key-intake/games' \
  --results '.regression-results/validation'
```

The analysis root links Ghost Pirates to its existing Game Folder for read-only cache reuse.
If a signal cache is absent, this command reports a blocked game.
Add `--allow-processing` only after approval. Later runs reuse the extracted signal caches.
Five-game measurements are available below. All five games now have validated, reusable signal caches.
The five-game baseline reused existing Ghost Pirates, Yetis, and Flyers signal caches.

## Measured Ghost Pirates result

The first cache-only run on 2026-10-10 produced:

| Metric | Result |
|---|---|
| Goal recall | 11/15 (73.3%) |
| Correct clip matches | 9/15 |
| Incorrect matches | 2, both wrong camera |
| Unjudged events | 111 |
| Total events | 122 |
| Runtime | 0.850 s, cached signals |

Missed goals: `away:2` (986.588 s), `away:6` (2412.752 s),
`away:7` (2874.813 s), and `away:11` (3347.409 s).
Wrong-camera matches: `away:1` and `away:10`.
The existing Game Folder events give the same metrics under these rules.
The second run took 0.230 s. All metric deltas were zero; runtime decreased by 0.620 s.
These runtimes reflect cache loading, not full footage processing.

The spec's older 12/15 baseline has no compatible saved result or explicit reference-point rules.
Do not interpret that difference as a measured regression.

## Measured Yetis result

The approved baseline extraction took 4379.984 seconds, or 73 minutes, for both camera signal caches.
It found seven of nine verified goals, with seven correct-camera matches and no incorrect matches.
It missed `icepak:4` at 1666.109 seconds and `yetis:1` at 2732.272 seconds.
The command reported HEVC decoder warnings; cache validation passed, but full footage integrity was not checked.

Cached experiments reused those signals without further footage extraction:

| Settings | Goals found | Correct matches | Incorrect matches | Candidates | Unjudged |
|---|---|---|---|---|---|
| Baseline P95/12s | 7/9 | 7 | 0 | 156 | 149 |
| Preserved P95/12s | 8/9 | 8 | 0 | 228 | 220 |
| Preserved P90/12s | 8/9 | 8 | 0 | 280 | 272 |
| Preserved P90/4s | 9/9 | 9 | 4 duplicates | 507 | 494 |
| Preserved P95/4s | 9/9 | 9 | 2 duplicates | 347 | 336 |
| Whistles | 8/9 | 8 | 0 | 63 | 55 |

Peak preservation recovered `icepak:4` at P95 without adding incorrect matches.
Reducing the separation to four seconds recovered `yetis:1`, but added duplicates.
P95/4s also missed two Ghost Pirates goals that P90/4s found.
These results do not support one new production default.
Unjudged candidates are not false positives.
The cached whistle run took 0.732 seconds, excluded extraction, and missed `yetis:1`.
It removed 444 candidates compared with full-recall P90/4s, but lost one verified goal.
Ghost Pirates remained at 15/15 with no incorrect matches and 44 whistle candidates.

The closest detected whistle after the missed Yetis reference is 23.198 seconds later.
Its normal candidate window cannot contain that reference.
Local cached audio features fail the current whistle rules; individual threshold changes did not recover a nearby whistle.
Changing several rules together produced weak, unstable detections. These are not verified whistle labels.
A synthetic adapter test confirms that a delayed seed does not imply goal coverage.
Whistle-only coverage is not sufficient on this game. Keep the separate flow pool available.

Saved reports live under `.regression-results/validation/`:

- Baseline: `20261011T040019.887364Z-f80ecdad/result.json`.
- Preserved P95/12s: `20261011T051343.326117Z-aeaf2fd7/result.json`.
- Preserved P90/12s: `20261011T051354.436140Z-ae26b4bf/result.json`.
- Preserved P90/4s: `20261011T051403.237680Z-32f98be1/result.json`.
- Preserved P95/4s: `20261011T051530.012820Z-c92562de/result.json`.
- Whistles: `20261011T054005.090473Z-15ca4822/result.json`.

Cached Yetis runs took 0.050–0.067 seconds. These runtimes exclude signal extraction and answer-key preparation.
The earlier experiments reported Flyers as blocked while its second signal cache was being extracted.

## Measured Flyers result

The approved baseline extraction took 4022.197 seconds, or 67 minutes, for both cameras.
It found both verified goals, with two correct-camera matches, no incorrect matches, and 131 candidates.
The other 129 candidates are unjudged, not false positives.
The third published goal remains excluded from recall because its scoring-play time is not independently verified.

The saved processing report is `.regression-results/validation/20261011T043450.156804Z-cbe88549/result.json`.
A cache-only rerun took 0.041 seconds and produced identical metrics.
The combined baseline report is `.regression-results/validation/20261011T054242.635926Z-e7335aca/result.json`.
That report covers all three games: 20/26 verified goals and 18 correct-camera matches, with two wrong-camera matches.
The cached Flyers whistle run found both verified goals in 33 candidates.
It produced one duplicate match for `flyers:1`, no wrong-camera matches, and 30 unjudged candidates.
Its runtime was 0.374 seconds, excluding extraction.

## Three-game comparison

All three flow signal caches are now available. These complete-suite comparisons require no footage extraction:

| Settings | Goals found | Correct matches | Incorrect matches | Candidates | Unjudged |
|---|---|---|---|---|---|
| Baseline P95/12s | 20/26 | 18 | 2 wrong camera | 409 | 389 |
| Preserved P95/12s | 21/26 | 21 | 0 | 551 | 530 |
| Preserved P90/4s | 26/26 | 26 | 15 duplicates, 2 wrong camera | 1231 | 1188 |
| Whistles | 25/26 | 25 | 1 duplicate | 140 | 114 |

The saved baseline is `20261011T054242.635926Z-e7335aca/result.json` under `.regression-results/validation/`.
Preserved P95/12s is `20261011T054504.261918Z-0e2ccd51/result.json`.
Preserved P90/4s is `20261011T054512.476450Z-6709ce50/result.json`.
Whistles is `20261011T060236.613816Z-32ee4ff0/result.json`.
Full recall requires 822 more candidates than the baseline and adds 15 incorrect matches.
Whistles remove 1091 candidates and 16 incorrect matches compared with full-recall flow, but lose the verified `yetis:1` reference.
The one remaining whistle duplicate belongs to `flyers:1`. There are no whistle wrong-camera matches in these three keys.
These measurements cover only verified references, not all published goals or non-goal plays.
Production defaults remain unchanged.

## Five-game comparison

The approved extraction finished with 39 verified goal references across five games.
These tables predate the three BLB non-goal labels added below. Their saved results remain unchanged.
BLB processing took 3515.921 seconds. Blue Devils processing took 4313.444 seconds.
HEVC decoder warnings occurred; signal validation passed, but full footage integrity was not checked.
A cache-only baseline reproduced every processing result with zero metric changes.
Its BLB and Blue Devils runtime comparisons are unavailable because their earlier runs included processing.

| Settings | Goals found | Correct matches | Incorrect matches | Candidates | Unjudged | Cached runtime |
|---|---:|---:|---|---:|---:|---:|
| Baseline P95/12s | 31/39 | 28 | 3 wrong camera | 721 | 690 | 0.676 s |
| Preserved P95/12s | 32/39 | 32 | 0 | 984 | 952 | 0.087 s |
| Preserved P90/4s | 39/39 | 39 | 26 duplicates, 4 wrong camera | 2164 | 2095 | 0.164 s |
| Whistles | 37/39 | 37 | 2 duplicates | 259 | 220 | 4.341 s |

Runtimes sum the per-game measurements. They exclude original extraction, preparation, and test runs.
The first cached baseline includes cold imports. Do not infer a speed improvement from these single-run timings.

| Game | Baseline goals | Preserved P95/12s goals | Preserved P90/4s goals | Whistle goals | Whistle correct / incorrect | Whistle candidates |
|---|---:|---:|---:|---:|---:|---:|
| Ghost Pirates | 11/15 | 11/15 | 15/15 | 15/15 | 15 / 0 | 44 |
| Yetis | 7/9 | 8/9 | 9/9 | 8/9 | 8 / 0 | 63 |
| Flyers | 2/2 | 2/2 | 2/2 | 2/2 | 2 / 1 | 33 |
| BLB | 3/3 | 3/3 | 3/3 | 2/3 | 2 / 0 | 63 |
| Blue Devils | 8/10 | 8/10 | 10/10 | 10/10 | 10 / 1 | 56 |

Peak preservation at P95 removes all three baseline wrong-camera matches and recovers Yetis `icepak:4`.
Blue Devils baseline misses `icepak:8` at 3516.849 seconds and `bluedevils:3` at 4020.917 seconds.
Its wrong-camera match belongs to `icepak:6`; peak preservation corrects that match.
P90/4s recovers both misses but adds eight incorrect Blue Devils matches.
Full-recall P90/4s adds 1443 candidates and 27 incorrect matches against the five-game baseline.

Whistles recover both Blue Devils misses and correctly match all ten references in 56 candidates.
Its duplicate belongs to `bluedevils:3`; the other whistle duplicate belongs to Flyers `flyers:1`.
Whistles remove 1905 candidates and 28 incorrect matches against full-recall flow, but lose two verified goals.
The missed references are Yetis `yetis:1` and BLB `blb:1`.
BLB's missed reference is at 1427.167 seconds. The next detected whistle is 27.173 seconds later.
No cached whistle candidate contains that reference. This does not prove that an audible whistle occurred there.
The ordinary BLB flow baseline contains and correctly matches the reference, so keep the separate flow pool.

BLB and Blue Devils audio and coverage preparation reused caches in their separate analysis folders.
Signal links point to validated regression caches; original NAS files and outputs remain unchanged.
PA music detection and the goalie swap check were disabled for this preparation.
Coverage assumed three periods because League timing was unavailable. BLB reported no detected period breaks.
These flags limit period interpretation; the candidate adapter uses verified camera coverage and timeline origins.

Saved reports under `.regression-results/validation/`:

- Processing baseline: `20261011T071716.621981Z-1be74d22/result.json`.
- Cached baseline: `20261011T092823.765880Z-5cd88a84/result.json`.
- Preserved P95/12s: `20261011T092824.447281Z-b1d2fa27/result.json`.
- Preserved P90/4s: `20261011T092824.538311Z-ab0a65fe/result.json`.
- Whistles: `20261011T092834.716695Z-8671eb21/result.json`.

These comparisons cover verified scoring-play references, not all published goals or non-goal plays.
Unjudged candidates are not false positives. Production defaults remain unchanged.

## Combined candidates (`hhg-3r5.97`)

Use the combined pool without changing production settings:

```bash
python v3/scripts/detection_regression.py \
  --games-root '.regression-results/answer-key-intake/games' \
  --results '.regression-results/validation' \
  --candidate-source combined \
  --set detection.preserve_peaks=true \
  --set detection.min_sep_s=4
```

The command saves `combined_events.csv` and `combined_candidates.json` beside the original flow and whistle artifacts.
The diagnostics record each contributing source event, settings, original pool counts, and the removed count.
The result records code and original pool hashes. Candidate construction never reads reference times or labels.
Missing whistle inputs block combined evaluation; they do not silently select a flow-only result.
Valid empty pools report zero recall rather than a blocked game.

The conservative duplicate rule requires the same camera and window centers within one second.
The windows must overlap by at least half the shorter window.
The retained interval contains both original intervals; it never removes same-camera point coverage.
Extensions cannot exceed `detection.max_win_s`. Longer inherited windows remain unchanged.
The original center remains fixed, so merging cannot form an unlimited chain of nearby events.
Distinct whistle clips remain separate, even when their windows overlap.
Whistle clips with exactly equal camera, start, and end now share one cut; diagnostics retain both source IDs.
Candidates from different cameras remain separate. This rule does not guess a new camera.
Retained scores keep their original source scale; they are not combined classification probabilities.

### Five-game cached results

These tables use the earlier goal-only keys. The expanded-key results appear in the next section.

| Settings | Goals found | Correct matches | Incorrect matches | Candidates | Removed from raw union | Unjudged |
|---|---:|---:|---|---:|---:|---:|
| Combined preserved P95/12s | 38/39 | 38 | 19 duplicates | 1167 | 76 | 1110 |
| Combined preserved P95/4s | 39/39 | 39 | 26 duplicates, 1 wrong camera | 1628 | 125 | 1562 |
| Combined preserved P90/4s | 39/39 | 39 | 38 duplicates, 4 wrong camera | 2238 | 185 | 2157 |

The raw union means both original pools before duplicate removal.
P95/12s still misses `yetis:1`. P95/4s recovers it and keeps every other verified goal.
P95/4s removes 125 near-identical windows from its 1753-event raw union without losing goal coverage.
It uses 536 fewer candidates than the earlier full-recall P90/4s flow result.
That difference combines a stricter threshold, whistle coverage, and duplicate removal; it is not duplicate removal alone.
Its one wrong-camera match belongs to Blue Devils `icepak:3`.
There is also a correct match for that reference; recall alone would hide the camera error.
This experiment does not meet the target of full recall with no wrong-camera matches.

| Game | P95/4s goals | Correct matches | Incorrect matches | Candidates | Cached runtime |
|---|---:|---:|---|---:|---:|
| Ghost Pirates | 15/15 | 15 | 8 duplicates | 276 | 0.201 s |
| Yetis | 9/9 | 9 | 6 duplicates | 377 | 0.296 s |
| Flyers | 2/2 | 2 | 3 duplicates | 264 | 0.243 s |
| BLB | 3/3 | 3 | 2 duplicates | 337 | 0.288 s |
| Blue Devils | 10/10 | 10 | 7 duplicates, 1 wrong camera | 374 | 0.358 s |

The final P95/4s run took 1.385 seconds, summed across games and excluding preparation.
Verification disabled footage processing and checked each original interval against the combined pool.
Every source interval retained same-camera coverage. Every verified key and existing Game Folder sidecar hash remained unchanged.
The combined recall equaled raw-union recall on every game.
Nearby windows may still represent distinct moments; broader suppression needs additional evidence, not reference-guided deletion.
Non-goal and Play Type labels were unavailable for these runs, so they do not establish precision or final Selection accuracy.
Production defaults remain unchanged.

Saved reports under `.regression-results/validation/`:

- Final P95/12s: `20261011T164039.276561Z-e1677e3e/result.json`.
- Final P95/4s: `20261011T164041.982231Z-2796f263/result.json`.
- Exploratory P90/4s: `20261011T163824.969749Z-d2bdbaa1/result.json`.

## Verified non-goal coverage (`hhg-3r5.98`)

BLB now has three independently reviewed non-goal references, in addition to its unchanged three goal references.
That first expansion kept 39 goals and gave the suite 42 labeled clips in total.
Each new label has a published sequence, a visually checked raw-frame pair, and extended raw context reviewed at four frames per second.
The review manifest binds all 20 evidence images and raw source identities to the labels.
The downloaded Recap hash matches the existing verified source hash.
Fitted timing supplied search proposals only. Visual checks established the camera/time mappings and non-scoring outcomes.

| Reference | Camera | Detection time | Reviewed outcome |
|---|---|---:|---|
| `non_goal:003` | cam1 | 306.000 s | Attack stops; a faceoff restarts play beside the same net |
| `non_goal:006` | cam2 | 770.919 s | Puck goes outside the net behind the end line; play continues |
| `non_goal:009` | cam1 | 1268.833 s | Net scramble sends the puck behind the end line; play continues |

The `cam2` reference uses raw chapter time 772.250 seconds minus the verified 1.331-second seek.
The labels describe these three published plays only. They do not label every instant in each reviewed span.
No save, miss, penalty, or other Play Type subtype was inferred.
The key remains incomplete. Unreviewed candidates remain unjudged, not false positives.

The report now separates goal and non-goal coverage and correct-camera matches.
`correct_clip_matches` remains the total across both label types.
New fields include `goal_clip_matches`, `non_goal_clip_matches`, `non_goals_total`, `non_goals_found`, and `missed_non_goals`.
Non-goal recall is unavailable when a key has no verified non-goals.
An older saved report without these fields remains readable; its new metric deltas are unavailable.
BLB's first expanded-key run rejects comparisons with its older key. Later runs compare against the expanded key.

### First expanded-key cached results (BLB only)

| Pool | Goal coverage | Correct goal matches | BLB non-goal coverage | Correct non-goal matches | Incorrect matches | Candidates | Unjudged |
|---|---:|---:|---:|---:|---|---:|---:|
| Baseline flow | 31/39 | 28 | 3/3 | 3 | 3 wrong camera | 721 | 687 |
| Whistles | 37/39 | 37 | 1/3 | 1 | 2 duplicates | 259 | 219 |
| Combined preserved P95/4s | 39/39 | 39 | 3/3 | 3 | 29 duplicates, 2 wrong camera | 1628 | 1555 |

Whistles miss the continuing-play references `non_goal:006` and `non_goal:009`.
This further supports retaining flow candidates for non-goal plays.
The combined pool correctly matches all 42 labeled clips but still has 31 incorrect extra matches.
The added labels expose three additional duplicates and a wrong-camera match for `non_goal:003`.
Those four candidates were previously unjudged; the candidate output did not change.
The other wrong-camera match remains Blue Devils `icepak:3`.

These are candidate-coverage results, not goal-classification results.
A correctly matched non-goal is a reviewed net play, not a false goal prediction.
Whole-game precision and final Selection accuracy remain unavailable.
All evaluation runs reused caches with footage processing disabled; only the approved review sampled new raw frames.
The five-game summed runtime was 0.408 seconds for flow, 3.523 seconds for whistles, and 1.582 seconds for combined.
Production defaults, original NAS outputs, graphics, and scoresheets remain unchanged.

Evidence manifest: `.regression-results/answer-key-review/blb_non_goal_review_01/verified_non_goal_review.json`.
Saved reports under `.regression-results/validation/`:

- Baseline flow: `20261011T165602.195383Z-a4f1d488/result.json`.
- Whistles: `20261011T165602.608435Z-ac9a79d3/result.json`.
- Combined P95/4s: `20261011T165606.143581Z-81b81bbd/result.json`.

### Second-game non-goal checks (Yetis)

Yetis now adds two independently reviewed continuing-play references:

| Reference | Camera | Detection time | Raw chapter time |
|---|---|---:|---:|
| `non_goal:006` | cam1 | 791.583333 s | 791.583333 s in `GX010025.MP4` |
| `non_goal:010` | cam2 | 1011.937 s | 1021.750 s in `GX010006.MP4` |

Published and paired raw frames show the same player positions and action.
The extended raw sequences show the puck outside the net and continuing play behind the end line and along the boards.
Both references use four-frame-per-second raw context review. Their evidence manifest binds 14 images and source identities.
The `cam2` mapping subtracts the verified 9.813-second seek.
No Play Type subtype was inferred.
EDL clip `edl:008` remains unlabeled: the goalie obscures the puck during its stoppage, leaving the outcome unverified.
The original nine Yetis goal references remain unchanged. Both expanded keys remain incomplete.

The suite now has 39 goals and five non-goals, or 44 labeled clips across five games.
All new evaluations reuse caches; Yetis comparisons across its key change are rejected.

| Pool | Goals covered | Correct goal matches | Non-goals covered | Correct non-goal matches | Incorrect matches | Candidates | Unjudged |
|---|---:|---:|---:|---:|---|---:|---:|
| Baseline flow | 31/39 | 28 | 5/5 | 5 | 3 wrong camera | 721 | 685 |
| Whistles | 37/39 | 37 | 2/5 | 2 | 2 duplicates | 259 | 218 |
| Combined preserved P95/4s | 39/39 | 39 | 5/5 | 5 | 30 duplicates, 2 wrong camera | 1628 | 1552 |

Whistles cover Yetis `non_goal:006` but miss `non_goal:010`, as well as the two BLB continuing-play references.
Combined candidates correctly match all 44 references but still produce 32 incorrect extra matches.
The new Yetis labels expose one more duplicate for `non_goal:010`; that candidate was previously unjudged.
This is a label expansion, not a changed detector output.
The two wrong-camera extras remain Blue Devils `icepak:3` and BLB `non_goal:003`.
These partial labels do not establish whole-game precision or classification accuracy.
The summed runtimes were 0.413 seconds for flow, 3.281 seconds for whistles, and 1.428 seconds for combined.
Original NAS outputs and production defaults remain unchanged.

Evidence: `.regression-results/answer-key-review/yetis_non_goal_review_01/verified_non_goal_review.json`.
Saved reports under `.regression-results/validation/`:

- Flow: `20261011T172459.548863Z-108ff3b2/result.json`.
- Whistles: `20261011T172459.968557Z-b1c14130/result.json`.
- Combined: `20261011T172503.257258Z-a00ccabc/result.json`.

### Raw review and exact duplicate removal

The five-game feedback check reproduced 30 duplicates and two wrong-camera extras before the change.
Raw review artifacts cover all 32 extras in 26 reference groups, with 28 camera overviews sampled at one frame per second.
All overview hashes and raw file identities were checked.
This coarse review does not establish exact play boundaries or every distinct action.
Answer keys remain unchanged. Reference times guided review only, not candidate construction.

Blue Devils `bluedevils:3` had two whistle clips with exactly equal camera and interval.
The combiner now keeps one source cut and both source IDs.
Nearby but unequal whistle intervals still remain separate.
The cache-only check keeps 39/39 goals, 5/5 non-goals, and all 44 correct matches.
It now reports 29 duplicates and two wrong-camera extras in 1,627 candidates, with 1,552 unjudged events.
It removes 126 windows from the 1,753-window raw union; the prior rule removed 125.
Every source interval retains same-camera coverage. All ten original pool CSV files match the prior run byte-for-byte.
The summed cached runtime was 3.774 seconds. This excludes raw review preparation and footage extraction.

The two camera errors need different treatment:

- Blue Devils `icepak:3`: cam1 shows earlier action around 1508–1512 seconds, then cam2 shows action around 1518–1521 seconds.
  The cam1 window overlaps the cam2 reference but contains earlier useful play.
  Deleting or reassigning that whole window could lose useful coverage.
- BLB `non_goal:003`: the cam1 overview shows the reviewed net play.
  The cam2 overview shows goalie movement while the skaters remain at the far end.
  These frames do not establish a safe general camera filter from the cached flow alone.

A diagnostic sweep tested wider same-camera grouping without changing the saved one-second setting:

| Grouping radius | Candidates | Duplicates | Wrong camera | Correct matches |
|---|---:|---:|---:|---:|
| 1 second | 1627 | 29 | 2 | 44/44 |
| 2 seconds | 1598 | 25 | 2 | 44/44 |
| 3 seconds | 1563 | 19 | 2 | 44/44 |
| 4 seconds | 1514 | 12 | 2 | 44/44 |
| 6 seconds | 1322 | 6 | 2 | 44/44 |

Every variant retained same-camera interval coverage, but this does not prove that distinct plays remain separate.
A synthetic pair three seconds apart stays separate at one second and merges at three or six seconds.
The wider rules therefore fail the existing distinct-moment requirement and were not promoted.
Neither wider grouping nor exact-cut removal fixes the two camera errors.
The next camera experiment needs action timing and camera handover evidence, not reference-based deletion.
Production defaults, original Game Folder outputs, graphics, and scoresheets remain unchanged.

Evidence and checks:

- Raw review: `.regression-results/answer-key-review/combined_error_review_01/review_findings.json`.
- Wider grouping: `.regression-results/answer-key-review/combined_grouping_sweep_01.json`.
- Coverage check: `.regression-results/answer-key-review/combined_exact_cut_validation_01.json`.
- Saved result: `.regression-results/validation/20261011T184450.834844Z-389a521a/result.json`.
- The diagnostic feedback script still fails because 31 extra matches remain; ordinary synthetic tests pass.

### Action-time and camera-score diagnostics

The next cache-only experiment recorded the detector's actual flow peaks and Selection's whistle action times.
Instrumentation checked each recorded peak against its exported flow interval and score.
It also checked each whistle action time against its exported interval, score, and camera.
Grouping still used the same one-second limit; it did not use reference times.

| Grouping time | Candidates | Duplicates | Wrong camera | Correct matches |
|---|---:|---:|---:|---:|
| Original clip centers | 1627 | 29 | 2 | 44/44 |
| Flow peaks only | 1682 | 43 | 2 | 44/44 |
| Whistle action times only | 1630 | 29 | 2 | 44/44 |
| Both extracted action times | 1632 | 32 | 2 | 44/44 |

Every variant retained same-camera source coverage, but action-time substitution did not improve the duplicate result.
No action-time substitution was promoted.

Camera diagnostics used the same cached arrays, original intervals, and unchanged whistle candidates.
A control recomputed the original local maximum scores and reproduced every original flow camera without reassignment.
Alternative scores changed only the flow camera for this diagnostic, not source files or ranking scores.
They are not complete detector implementations or calibrated classification scores.

| Camera score | Candidates | Duplicates | Wrong camera | Reassigned flow cuts | Original cuts without same-camera coverage |
|---|---:|---:|---:|---:|---:|
| Original local maximum | 1627 | 29 | 2 | 0 | 0 |
| Local mean | 1631 | 29 | 2 | 21 | 21 |
| Local net maximum | 1626 | 29 | 2 | 26 | 26 |
| Local slot maximum | 1626 | 28 | 3 | 38 | 38 |
| Whole-window maximum | 1637 | 31 | 0 | 113 | 113 |
| Whole-window mean | 1636 | 30 | 1 | 121 | 121 |

All variants correctly matched the 39 goals and five non-goals, but those incomplete references do not verify the reassigned cuts.
Whole-window maximum scoring removes both measured camera errors at the cost of 113 original camera intervals.
The new synthetic handover test shows the risk: a stronger later attack can change the camera for an earlier attack.
The preserved local rule keeps both synthetic camera matches; assigning both cuts to the later camera loses the earlier match.
No alternative camera score was promoted.

At BLB's erroneous flow peak, cam2 has stronger cached motion in both the net and slot regions than cam1.
The raw overview shows goalie movement there, so net-versus-slot score changes alone do not establish the correct camera.
Blue Devils' earlier cam1 action and later cam2 action still require time-specific camera evidence.
Further camera work needs independently checked action boundaries or spatial evidence, not a single camera for the whole overlap.

Saved evidence:

- Source run and extracted peak sidecars: `.regression-results/validation/20261011T201941.620080Z-9cdbb238/`.
- Action-time variants: `.regression-results/answer-key-review/combined_action_time_experiment_01.json`.
- Camera-score variants with the control: `.regression-results/answer-key-review/combined_camera_score_experiment_02.json`.
- Immutability checks: `.regression-results/answer-key-review/combined_action_camera_validation_01.json`.
- Synthetic handover guard: `test_whole_window_camera_score_can_hide_an_earlier_attack` in `tests/test_detection_regression.py`.

All five key digests and 20 source and accepted CSV files match the previous saved run.
The accepted candidate output remains at 29 duplicates and two wrong-camera extras.
Answer keys, production defaults, original Game Folder outputs, graphics, and scoresheets remain unchanged.

### Dense camera-boundary review

Four-frame-per-second raw review now covers both remaining camera errors at 1280-pixel analysis width.
The review inspected 374 frames, 32 dense sheets, and two paired sample images across four source streams.
All 34 image hashes, four cache hashes, and raw file identities were checked.
This is separate scene evidence, not a new answer key or candidate filtering rule.

| Game | Independently observed scene | Bounded timing |
|---|---|---|
| Blue Devils | Near-net action at cam1, departure from that net, approach to cam2, then near-net action at cam2 | Cam1 departure bracket: 1513.25–1515.75 s; cam2 approach at 1517.75 s; cam2 net action at 1521.0 s |
| BLB | Cam1 play followed by players settling; cam2 goalie movement with no nearby attacking skaters in the sampled views | Cam1 settling bracket: 306.083–308.083 s; cam2 goalie-motion anchors: 312.333, 315.083, and 318.083 s |

Blue Devils' first reviewed cam1 frames already contain action, so its start remains left-censored.
The broader camera-transition bracket is 1513.25–1521.0 seconds; no exact camera-switch time was assigned.
The earlier cam1 attack remains useful footage and must not disappear through whole-window camera reassignment.
This review does not label that earlier attack as a goal, non-goal, or Play Type.

BLB's 79 sampled cam2 frames show the goalie as the only nearby player; the other skaters remain at the far end.
This does not prove that no brief action occurred between samples.
It confirms that strong aggregate motion at the reviewed extra peak does not establish a near-net attack.

Times describe nominal sample bins using the existing sync offsets.
The quarter-second allowance describes sampling only, not total sync error.
The brackets are bounded scene observations, not exact full-play starts, ends, or puck-crossing times.
Candidate construction must not consume these reference-selected brackets.
All 39 goal and five non-goal labels remain unchanged; no detector setting or source interval changed.

Evidence: `.regression-results/answer-key-review/camera_boundary_review_01/verified_scene_review.json`.
Its SHA-256 is `5be342d8d148aec3d6c9153b9ab71f35e6ff5150616077163e0a003828fea339`.
Cached motion samples remain separate in `.regression-results/answer-key-review/camera_boundary_motion_samples_01.json`.
Follow-up `hhg-3r5.99` tracks opt-in spatial skater evidence, with explicit uncertainty and camera-handover tests.
The follow-up must use source frames for evidence and these reviewed scenes only for evaluation.
The accepted result remains 44 correct matches, 29 duplicates, and two wrong-camera extras.

### Opt-in spatial observation probe (`hhg-3r5.99`)

`spatial_camera_probe.py` now reports model boxes against each camera's existing net and slot ROIs.
It does not feed Detection or Selection, delete candidates, change cameras, or change the frozen CSV schemas.
The geometry module imports no ML runtime; the CLI loads the optional runtime only when needed.
The probe requires existing local HockeyAI weights and never downloads them.
Missing weights or invalid inputs return an explicit failure, not an empty-net label.

Example using the saved BLB evaluation cache:

```bash
python v3/scripts/spatial_camera_probe.py \
  --frame-cache .regression-results/answer-key-review/camera_boundary_review_01/blb_01042026/non_goal_003/cam2_dense.npz \
  --rois .regression-results/answer-key-intake/games/buzzlightbeer_01042025/rois.json \
  --camera camera_2 \
  --weights "$HOME/.cache/hockey-highlight-generator/HockeyAI_model_weight.pt" \
  --output .regression-results/spatial-evidence/evaluation_03/blb_01042026/cam2.json \
  --stride 4 --scope evaluation_only
```

The cache contains BGR frames, nominal detection times, and the raw file path, size, and modification time.
The probe binds observations to frame bytes, raw identity, model weights, runtime versions, ROI bytes, code, stride, and declared scope.
It reuses a valid prediction output without model calls.
Changed inputs or invalid cached counts require a new output path; the probe never overwrites the old result.
Outputs must remain separate from raw sources and Game Folder inputs.
`can_filter` and `can_reassign_camera` always remain false.

Model player, goalie, and referee observations remain separate.
The geometric position is the box-bottom center, not a verified foot or ice-contact point.
Players outside the ROI are recorded as `outside_roi_players`, not distant players.
A positive ROI-player count does not establish an attack.
A zero count does not establish absence, especially with occlusion or lateral action outside the ROI.

The experiment reused local weights with SHA-256 `58ac4450b4bd2545aaff43812bd05171af3383f30195aab5e010571039cf6d90`.
It used CPU inference at 1280 pixels, prediction confidence 0.15, and observation confidence 0.25.
These settings were not fitted to answer-key labels.

Two sample sets remain separate:

- Preliminary evaluation: 94 one-second samples from the two reviewed camera-error windows.
  BLB cam1 has positive ROI-player observations in all 20 samples; cam2 has none in those 20 samples.
  Blue Devils exposes a model error: the goalie at cam2 is labeled as a player with confidence 0.753822 at 1507.75 seconds.
- Independent source selection: three candidate-center times per game, selected at chronological rank fractions 0.25, 0.5, and 0.75.
  The sampler never reads an answer key or scene bracket. It prepared 15 paired times and 30 frames across five games.
  Ten pairs have a positive player observation at one camera; five pairs have positive observations at both cameras.
  These are model observations, not verified camera choices or held-out filtering accuracy.

All seven overlay images were visually reviewed, covering the two preliminary scenes and all 15 independent source pairs.
The independent Ghost Pirates sample also labels a nearby goalie as a player.
The Flyers sample at 950.75 seconds shows a skater beside cam2's net but outside the ROI region used for counting.
These failures prevent automatic filtering or camera reassignment from this probe.
Further work must address role confusion, occlusion, and ROI coverage before proposing a candidate rule.
No outcome labels or Play Types were inferred from these still frames.

Raw model boxes were reused when the output terminology changed; all count decisions remained unchanged.
All seven overlay hashes, 34 prediction-output hashes, and raw source identities were checked.
The five-game cache-only comparison still correctly matches all 39 goals and five non-goals in 1,627 candidates.
It retains 29 duplicates, two wrong-camera extras, and 1,552 unjudged candidates.
Camera changes, filtered candidates, and lost source coverage are zero because no spatial rule was applied.
All 20 source and accepted CSV files match the previous run byte-for-byte.
This unchanged comparison does not validate spatial filtering.

Current artifacts under `.regression-results/spatial-evidence/`:

- `evaluation_03/`: current preliminary observations with explicit box-bottom and outside-ROI terminology.
- `source_samples_01/inventory.json`: label-free sampling protocol and source-bound frame caches.
- `source_probe_03/inventory.json`: current five-game source observations.
- `source_probe_03/validation.json`: unchanged candidate comparison and scope limits.
- `review_findings_01.json`: visually reviewed model and geometry failures, with output and overlay hashes.
- `box_cache_reuse_02.json`: old-to-new prediction provenance and unchanged count checks.

Saved candidate comparison: `.regression-results/validation/20261011T210233.455269Z-b0fc9c82/result.json`.
`hhg-3r5.99` remains in progress; filtering and camera decisions remain unvalidated.
Production defaults, answer keys, original outputs, graphics, and scoresheets remain unchanged.

### Cached role-conflict review

`spatial_role_review.py` now flags conflicting player and goalie labels in linked model boxes.
A component is a group of linked boxes, not a count of people.
The links do not verify that two boxes show the same person.
The review never runs inference, decodes footage, changes role labels, or changes candidates.

```bash
python v3/scripts/spatial_role_review.py \
  --observations .regression-results/spatial-evidence/evaluation_03/bluedevils_10192025/cam2.json \
  --frame-cache .regression-results/answer-key-review/camera_boundary_review_01/bluedevils_10192025/icepak_3/cam2_dense.npz \
  --rois .regression-results/answer-key-intake/games/bluedevils_10192025/rois.json \
  --output .regression-results/spatial-evidence/role_cli_example/cam2.json
```

The CLI validates the existing prediction output against its frame cache, raw identity, ROI, code, and model settings.
It records the prediction hash and the role-review code hash.
It refuses output in Game Folders or on input paths, and never overwrites an existing result.

The association rules are diagnostic, not fitted camera rules:
Overlap means intersection area divided by union area.

- Boxes need player or goalie confidence of at least 0.15.
- Same-frame conflicting roles link at overlap of 0.5, or containment of 0.8 within the smaller box.
- Adjacent frames link at overlap of 0.3, or containment of 0.8, with at most 1.5 seconds between samples.
- Temporal matches use deterministic one-to-one box links; missing samples break the links.
- Crossing or overlapping players can still produce a false association.
- `role_verified`, `physical_identity_verified`, `can_filter`, and `can_reassign_camera` remain false.

The four preliminary streams contain 16 mixed-role components involving 97 model boxes:

| Scene | Camera | Mixed-role components | Boxes in those components |
|---|---|---:|---:|
| Blue Devils | cam1 | 3 | 24 |
| Blue Devils | cam2 | 5 | 22 |
| BLB | cam1 | 7 | 49 |
| BLB | cam2 | 1 | 2 |

These counts are not verified role errors or goalkeeper tracks.
All 30 independent source-frame reports explicitly have insufficient temporal samples.
Overlapping same-frame labels can still be flagged, but one frame does not establish a track.

The known Blue Devils cam2 goalie mislabel remains unresolved.
Its linked component contains only the player label across eight samples, from 1507.75 to 1514.75 seconds.
It has no mixed-role flag, although the reviewed image shows a goalie.
Stable wrong labels therefore defeat a rule that treats consistency as verified role evidence.
A new synthetic guard covers this failure: eight consistent player labels never authorize filtering or camera reassignment.

Current role outputs: `.regression-results/spatial-evidence/role_review_02/`.
Its `inventory.json` binds all 34 reports; `validation.json` records the persistent mislabel and scope limits.
Five answer-key digests and all 20 accepted and source CSV hashes remain unchanged.
The accepted candidate result still has 44 correct matches, 29 duplicates, and two wrong-camera extras.
No accuracy improvement or filtering safety is claimed from this diagnostic.

`hhg-3r5.100` now tracks independent goalie and skater role references.
`hhg-3r5.99` depends on those references before role-based camera decisions can be validated.
The follow-up verifies data only; it does not authorize training, weight changes, or default promotion.

### Independent visual role references (`hhg-3r5.100`)

`tests/role_references/` now contains a separate role-reference suite, a frozen sampling protocol, and reviewed labels.
It does not change `tests/answer_keys/` or provide goal, non-goal, or Play Type labels.
The visual review used raw images and adjacent raw context, without model predictions or prediction overlays.
Here, independent means the visual labels did not come from the model or outcome keys.
It does not mean a separate human reviewer confirmed the labels.

The protocol was frozen before new decoding and review.
Its SHA-256 is `02227019ff6208e57235fd2bc8db644b36143be0fed175faadd9b3926dee10e8`.
Two candidate-center times per game use fixed chronological rank fractions 0.375 and 0.625.
Each time has both camera views and nine nominal samples at two fps, from two seconds before to two seconds after.
All 20 camera contexts were prepared, yielding 180 source-bound frames.
No missing view was replaced with another sample.

The protocol keeps two partitions:

- Development: Ghost Pirates and Flyers, with eight camera contexts.
- Reserved evaluation: Yetis, BLB, and Blue Devils, with 12 camera contexts.

All five games already have outcome evaluation, so these are not held-out games.
No new sample overlaps the named earlier source-single or dense diagnostic cache intervals.
That check does not prove nobody previously viewed another image from those times.
No role method or model was fitted during this intake.

The review covers 20 raw center images, 20 raw context sheets, and five manually selected subject-crop sheets.
It records visible equipment or skating evidence rather than using net position alone.
The labels contain 20 goalie, 21 skater, and four referee references; two obscured subject references remain uncertain.
These are frame-subject references, not counts of unique people or verified tracks.
The development partition contains both uncertain references; no uncertain label was forced into a confirmed role.
Selected subjects do not enumerate every person or establish absent players.

Each subject has a manual image point, a review crop, visible evidence, and explicit limits.
The point and crop do not provide a complete body box or physical identity across frames.
The references bind the exact center-frame bytes, context-image bytes, cache bytes, and frozen frame inventory.
The local inventory also records each raw file's path, size, modification time, and camera manifest hash.
The sample times remain nominal bins, not verified video PTS or total sync accuracy.

`spatial_role_references.py` validates these bindings without decoding video or loading the ML runtime:

```bash
python v3/scripts/spatial_role_references.py \
  --references tests/role_references/references.json \
  --protocol tests/role_references/protocol.json \
  --inventory .regression-results/spatial-evidence/role_references_01/frame_inventory.json
```

The validator refuses changed source identities, images, caches, partitions, times, or frame indexes.
It retains every planned camera sample, including unavailable samples.
It checks that the center image equals the bound cache frame.
It refuses model labels or net position alone as verified evidence.
It requires equipment evidence for goalies and stripe evidence for referees.
The validator checks provenance and schema; it does not establish that a visual label is correct.
`can_filter` and `can_reassign_camera` remain false.

Current reference SHA-256: `4e852affc7afc4b234b8077f4e550560740a3d2dca6cff8db0fce44f88722028`.
Local evidence and validation remain under `.regression-results/spatial-evidence/role_references_01/`:

- `planned_samples.json`: frozen times, source ranks, CSV hashes, and known-review overlap checks.
- `frame_inventory.json` and `frames/`: 20 raw context caches, center images, context sheets, and source identities.
- `manual_subjects.json`: the raw visual annotation record.
- `*_subject_crops.png`: five reviewed crop sheets, without model predictions.
- `bundle_validation_01.json`: current reference hash, partition counts, evidence hashes, and preservation checks.

The tracked labels require those local artifacts for full provenance validation.
Synthetic tests do not require the NAS footage or frame caches.
No inference ran on the new role-reference samples, so no new model accuracy is claimed.
The known persistent-goalie mislabel remains discovery evidence in `role_review_02/validation.json`, not reserved-evaluation evidence.
Further validation must compare unchanged model outputs against the frozen references without fitting to the reserved partition.
Five outcome-key digests and 20 accepted/source CSV hashes remain unchanged.
The accepted result retains all 44 correct matches, 29 duplicates, and two wrong-camera extras.
No candidates, defaults, original outputs, graphics, or scoresheets changed.
`hhg-3r5.100` remains in progress for review; no issue was closed.

### Fixed-model comparison at frozen role points

`spatial_role_benchmark.py` now compares cached model boxes against the unchanged role-reference points.
It does not change weights, confidence settings, points, labels, ROIs, candidates, or production defaults.
The prediction driver reads only the frozen source-frame inventory, not the role labels.
It selects the center index already fixed by the source protocol.
All 20 center frames came from existing caches; no footage was decoded again.
The weights, runtime versions, and inference settings match the earlier source probe.
No model or dependency was downloaded.

The association protocol was frozen before this comparison.
It keeps confidence 0.25 as the primary view and 0.15 as the existing secondary view.
No threshold was fitted from the results.

Association uses half-open containment of a manual subject point within a predicted person-role box.
It does not use the reference role to choose a box.
Multiple boxes remain ambiguous, even if one box has the expected role.
A box that contains more than one reference point remains ambiguous for all affected points.
This rule includes uncertain reference points, so they cannot silently permit a shared match.
The two uncertain references remain unjudged.
Predictions without a labeled point remain unjudged, not false positives.
The references do not enumerate all people in a frame.

Primary confidence view, across the 47 reference subjects:

| Partition | Role agreement | Role disagreement | Ambiguous point | Unassociated point | Unjudged reference |
|---|---:|---:|---:|---:|---:|
| Development | 8 | 2 | 3 | 4 | 2 |
| Reserved evaluation | 19 | 3 | 2 | 4 | 0 |
| Total | 27 | 5 | 5 | 8 | 2 |

All five primary disagreements associate a verified goalie point with a player-class box.
The fixed-model overlays were visually reviewed for all 47 subjects across all five games.
The five goalie cases are:

| Game | Camera | Nominal time | Player confidence | Partition |
|---|---|---:|---:|---|
| Flyers | cam2 | 1427.417 s | 0.835290 | Development |
| Flyers | cam1 | 2184.333 s | 0.699168 | Development |
| Yetis | cam2 | 2809.000 s | 0.685551 | Reserved evaluation |
| Blue Devils | cam1 | 1566.583 s | 0.799762 | Reserved evaluation |
| Blue Devils | cam2 | 2813.500 s | 0.777991 | Reserved evaluation |

These cases show the role problem beyond the earlier discovery scenes.
At the 20 goalie reference points, the primary view has six agreements, five disagreements, three ambiguous associations, and six unassociated points.
An unassociated point does not prove that the model failed to detect the person elsewhere.
Box association does not verify actor identity or a track.
Role agreement is conditional on unambiguous point geometry, not whole-game role accuracy or detection precision.
The reserved partition is not a held-out game set.

The secondary view has 29 agreements, seven disagreements, five ambiguous points, four unassociated points, and two unjudged references.
Lower confidence therefore does not resolve the role problem.
No threshold view supports automatic filtering or camera reassignment from this evidence.

The saved prediction runtimes sum to 8.838029 seconds for 20 cached center frames.
That sum includes the first model load, but excludes cache preparation, validation, review, and outer process overhead.
The cached comparison itself never loads the model or decodes video.
It validates frozen reference and prediction identities, source context hashes, center pixels, ROIs, settings, and all planned samples.
It refuses stale inputs and missing or duplicated sample records.
Missing predictions remain unavailable, not empty-frame or verified-role evidence.

Run the read-only cached comparison:

```bash
python v3/scripts/spatial_role_benchmark.py \
  --references tests/role_references/references.json \
  --protocol tests/role_references/protocol.json \
  --frame-inventory .regression-results/spatial-evidence/role_references_01/frame_inventory.json \
  --predictions .regression-results/spatial-evidence/role_benchmark_01/predictions/inventory.json \
  --association-protocol .regression-results/spatial-evidence/role_benchmark_01/association_protocol.json
```

Artifacts under `.regression-results/spatial-evidence/role_benchmark_01/`:

- `association_protocol.json`: frozen settings, association rules, reference hashes, and limits.
- `predictions/`: 20 source-bound center caches, fixed model outputs, and the prediction inventory.
- `result.json`: immutable per-reference comparisons, per-game counts, and separate partition counts for both confidence views.
- `overlay_inventory.json` and `overlays/`: five reviewed overlay sheets and their hashes.
- `review_findings.json`: five visually reviewed goalie cases, role counts, runtime scope, and preservation checks.

Five outcome-key digests and 20 accepted/source CSV hashes remain unchanged.
The accepted candidates still correctly match all 39 goals and five non-goals.
They retain 29 duplicates and two wrong-camera extras.
Filtering, camera changes, and lost source coverage remain zero because no spatial rule was applied.
`hhg-3r5.99` remains in progress: role-based filtering and camera decisions remain unvalidated.

## Checks

```bash
python -m pytest -q tests/test_spatial_role_benchmark.py tests/test_spatial_role_references.py tests/test_spatial_role_review.py \
  tests/test_spatial_camera_evidence.py tests/test_spatial_camera_probe.py \
  tests/test_detection_regression.py tests/test_combined_candidates.py tests/test_signal_cache.py \
  tests/test_signals_concat.py tests/test_signals_perf.py tests/test_run_detect_folder_mode.py \
  tests/test_selection.py tests/test_selection_plays.py tests/test_answer_key_edl.py \
  tests/test_answer_key_frames.py tests/test_answer_key_matching.py tests/test_published_clip_intake.py
python -m pytest -q
git diff --check
```

The tests use synthetic labels, events, footage identities, and signal caches.
They check matching, metrics, comparisons, immutable history, cache placement,
processing approval, and protection of keys and Game Folder outputs.

All 343 targeted checks pass, including fixed-role comparisons, reference provenance, cache binding, combined candidates, and interrupted intake recovery.
The full suite passes 839 tests. Two unrelated graphics tests fail because the new worktree lacks Node dependencies.
Graphics files and dependencies remain unchanged.
