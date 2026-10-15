# Interesting non-goal plays from behind-net GoPro footage

Research date: 2026-10-03. Scope: fixed, fisheye, behind-net amateur hockey footage.
This note separates source findings, user observations, and proposed detection rules.
No model inference, video processing, or new performance measurement was run.

## Main findings

- Treat a dangerous attack and its outcome as one sequence. A whistle is optional.
- Add rush chances, rapid lateral shot changes, cross-crease passes, and passes from behind the net to the candidate indicators. [6–8]
- Keep distant shots when their outcome creates interest: a save, rebound, tip, post impact, or defensive stop. [5,7,8]
- Use player convergence and sustained net-area activity to propose scrambles. They do not prove puck possession or a shot. [9,10]
- HockeyAI detects objects. It does not directly detect shots, saves, rebounds, goals scored, scrambles, possession changes, or yelling. [1–3]
- Preserve goal-first selection. Rank non-goal plays within the remaining budget, with a separate check for clip quality. [12]

Shot quality and highlight interest are different targets.
NHL expected-goals research supports candidate features, not amateur scoring probabilities or viewer preferences. [7,8]
The rules below require local labels before automatic inclusion.

## What HockeyAI actually provides

### Model, source, and labels

The official model card describes a YOLOv8 medium model trained on HockeyAI. [1]
The official demo loads its weights, calls `model.predict(image)`, and plots the first result. [3]
That demo contains no temporal event classifier, possession logic, tracking stage, or audio analysis. [3]

The dataset metadata lists these seven classes. [2]
The card uses descriptive names; the published precision-recall plot uses some different spellings. [1,4]

| Dataset label | Meaning | Possible downstream use; not a model event output |
|---|---|---|
| `centerIce` | Center circle | Rink reference; potential restart context when visible |
| `faceoff` | Faceoff dots | Rink reference; not detection of a faceoff occurring |
| `goal` | Goal frame | Net-area reference; not a goal scored |
| `goaltender` | Goalie | Location and box changes over time; not save recognition |
| `player` | Hockey player | Counts, convergence, and tracks after extra processing |
| `puck` | Puck | Visible candidate locations; not continuous possession or trajectory |
| `referee` | Official | Location; not a penalty, whistle, or referee gesture |

HockeyAI has no separate attacker, defender, team, stick, glove, pad, pose, screen, or goalpost-impact class. [1,2]
Player boxes do not supply identities, team membership, body orientation, or stick contact.
Those outputs need additional models or rules.
Check the loaded weight file's class mapping before assigning numeric class IDs.
This research did not download or inspect the weight file.

### Training domain and evidence limits

The dataset card describes approximately 2,100 manually annotated, 1920×1080 frames from SHL broadcasts. [2]
The paper's author abstract states 2,101 frames from multiple SHL seasons and teams. [14]
This is professional broadcast material, not a documented behind-net amateur GoPro benchmark. [2,14]

There are packaging differences:

- The dataset card's generated metadata lists 1,890 examples in a `train` split, with only an image feature. [2]
- Its `dataset-info.json` lists `train`, `valid`, and `test`, plus the seven classes. [2]
- The repositories also supply a dataset ZIP. Its contents and split membership were not inspected. [15]

Do not infer the test size, game-separated split policy, or annotation completeness from these metadata alone.

The model card lists `mAP@0.5: XX.X%`, 100% precision, 95% recall, and 93% F1 for all classes. [1]
These claims are incomplete and inconsistent if they describe the same evaluation point.
Precision 1.00 and recall 0.95 would give F1 about 0.974, not 0.93.

The official precision-recall figure supplies more specific evidence. [4]
Its legend reports puck AP 0.702, goal AP 0.991, and all-class mAP@0.5 0.950.
It labels the goalie class `goalie` and the center class `centriod`.
These are figure labels, not a verified weight-file mapping.
AP is not frame recall, event accuracy, or save detection performance.
The aggregate training plots also do not establish 100% precision for each class. [4]

The project ADR reports puck detections in 18% of frames near goals in one local game. [12]
This is a recorded project observation, not a new measurement or a verified general recall estimate.
The ADR does not give the visible-puck denominator, thresholds, or full annotation protocol.

**Conclusion:** use HockeyAI first for net and player context.
Treat puck evidence as optional until local evaluation supports stronger use.
Missing puck detections must mean unknown, not no shot or no interesting play.

## Event definitions that prevent false labels

The NHL defines a shot on goal by the puck's path toward the net. [5]
The path must enter the net, or would enter without the goalie stopping it.
This includes tips and deflections. A goalie moving or dropping alone does not meet that definition. [5]

NHL EDGE defines a save as a shot on goal stopped by the goalie. [5]
It excludes blocked shots, missed shots, and post or crossbar hits from save-percentage shot counts. [5]
Therefore, keep near misses and defensive blocks as possible highlights without labeling them saves.
The NHL save definition does not require a whistle. [5]

Natural Stat Trick includes goals, saved shots, misses, and blocks in shot attempts. [6]
Its rebound definition is another attempt within three seconds of a blocked, missed, or saved attempt, without an intervening stoppage. [6]
Its rush definition is an attempt within four seconds of an event in the neutral or defensive zone, without an intervening stoppage. [6]
These are that provider's analytical rules, not universal video thresholds.
They support preserving continued play and inspecting short event sequences.

Natural Stat Trick scores chances using location, rush/rebound context, and a deduction for blocks. [6]
NHL EDGE uses a different geometric high-danger region, within 29 feet of the goal center with specified lateral bounds. [5]
MoneyPuck defines high danger by modeled goal probability, at least 20%. [7]
Do not interchange these definitions or attach NHL danger labels to uncalibrated image rectangles.

## Candidate indicators

The user observed distance shots, carries into the slot/crease, saves, rebounds, clears, possession changes, scrambles, post impacts, and yelling.
These observations define the desired content. They are not measured detector capabilities.
The following proposed cues translate those observations and source findings into testable rules.

| Play or indicator | Source support | Proposed visible cue | Main ambiguity |
|---|---|---|---|
| Carry into slot/crease | Shot distance and angle are established shot-quality features. A carry alone is not a shot. [7,8] | A tracked player enters the near-net region, then a shot-like action or goalie response follows. | The player may lack the puck; depth is distorted. |
| Distant shot with a consequential outcome | MoneyPuck models rebound creation and notes that some distant shots create more expected rebound value than direct shot value. [7] | A release-like action outside the slot precedes net activity, a goalie response, or a second attempt. | Flow can miss the distant shooter and emphasize unrelated nearby motion. |
| Save with or without whistle | Saves depend on stopping shots on goal, not a stoppage. [5] | An attack precedes a goalie response, then a freeze, rebound, or clear follows. | A goalie adjustment is not proof of a save. |
| Rebound or repeated attempts | Rebounds and shot flurries have explicit analytical treatment. [6,7] | Several attack-response peaks occur in one continuous net-area sequence. | Peaks may be skating, celebration, or post-whistle movement. |
| Net-area scramble | Temporal player context helps puck localization; crowding is only an indirect cue. [9,10] | Players converge, remain close, and repeatedly move around the crease. | A faceoff setup or post-whistle gathering can look similar. |
| Rush chance or breakaway-like attack | Natural Stat Trick gives rush attempts additional chance value. [6] | A fast approach from farther up ice ends in a near-net attempt or goalie response. | The camera may omit the entry; player count does not establish possession. |
| Cross-crease pass, lateral rebound, or one-timer-like sequence | MoneyPuck illustrates rebound angle change. Evolving-Hockey discusses cross-crease passes missing from its event data. [7,8] | Rapid movement across the net precedes an attempt and goalie lateral movement. | Box movement alone cannot distinguish a pass from a skating route. |
| Pass from behind the net or wraparound-like attack | Evolving-Hockey includes wraparound shot type and discusses a pass from behind the net. [8] | Movement behind the goal continues around a post or into the slot before an attempt. | The goal frame can hide the puck and stick contact. |
| Tip, deflection, or screened attempt | The NHL includes tips/deflections in shots on goal; Evolving-Hockey includes these shot types. [5,8] | An attempt occurs with a player in the apparent shot lane, followed by goalie response. | Camera overlap does not prove a goalie screen or a puck deflection. |
| Defensive block, clear, or possession change | Blocks are shot attempts; prior giveaways and takeaways are shot-model features. [6,8] | Pressure at the net ends with movement away, or a new attack follows a turnover. | Teams, puck contact, and possession require extra evidence. |
| Post/crossbar impact | NHL EDGE treats a post-only attempt as a missed shot, distinct from a save. [5] | An attack aligns with a short impact sound and a visible near miss. | A goalie or stick can hit the post; audio alone cannot establish puck impact. |
| Yelling or player reaction | User observation. No inspected source validates this as an amateur highlight detector. | A vocal burst supports an already plausible visual event. | Calls for a pass, PA speech, and unrelated reactions are common alternatives. |

The strongest additions are rush context, lateral changes, cross-crease passes, and behind-net feeds.
Tips, screens, blocks, and possession changes are useful review labels but need stronger evidence for automatic naming.
A dramatic goalie recovery is another proposed interest cue, not a HockeyAI output or a validated shot-quality feature here.

## Relevant action and puck research

### Temporal player action: HARPET

Cai et al. combine pose, stick keypoints, and optical flow across three images. [11]
Their classes are skating forward, skating backward, passing, and shooting.
They are not save, rebound, scramble, goal, or turnover classes.
HARPET contains 424 three-image sequences from internet clips and instructional DVDs. [11]
The paper uses a random 70/15/15 split, not a documented held-out GoPro game test.

Table 4 reports 83.60% average accuracy across three selected checkpoints; the abstract rounds the result to about 85%. [11]
That result must not become a projected amateur-footage accuracy.
The paper reports stick occlusion and motion problems, plus confusion between passing and shooting. [11]
It supports testing temporal shot-like actions, rather than treating whole-region flow as a shot detector.

### Joint puck location and event recognition

Vats et al. use temporal video features and player-location heatmaps on NHL broadcast clips. [9]
Their four event classes are Faceoff, Advance (dump in/out), Play (pass/stickhandle), and Shot.
The dataset has 8,987 two-second clips with approximate puck locations for each clip, not precise frame-level trajectories. [9]

The reported localization AUC is 73.1%, integrating accuracy over distance tolerances from 5 to 50 feet. [9]
It is not 73.1% puck-box recall or goal-line crossing accuracy.
Their multitask Shot F1 is 58.4%; macro event F1 is 66.8%. [9, Table 7]
Continuous-video evaluation is qualitative because the authors lack frame-level ground truth. [9, §4.1.3]

This supports using temporal context and player interaction when the puck disappears.
It does not support claiming reliable save outcomes or continuous possession from object boxes.

### Contextual puck localization: PLUCC

Salass et al. combine full-image features with segmented-player context to produce puck-location heatmaps. [10]
Their proprietary VIP-PuckDataset has 150,000 images; matches do not cross dataset splits. [10, §4.1]
Player orientation and position provide useful context, but the context-only experiment performs poorly. [10, §4.6.1]
Convergence alone is therefore not an adequate substitute for puck observation.

The paper reports 83.5 distance-threshold mAP in Table 1 and 1.05 m average rink-space error in Table 2. [10]
Those metrics differ from HockeyAI box AP and the Vats tolerance AUC.
They cannot be compared as if they measure the same task.
The combined pipeline runs at about 6 FPS on the reported GPU setup and requires player detection and segmentation. [10]
Weights availability and performance on this project's cameras were not verified.

**Practical conclusion:** these papers support feature choices and future specialist models.
They do not provide an established turnkey non-goal detector for this camera view.

## View and audio constraints

### Fixed fisheye behind-net view

The following are engineering deductions for this view, not measured HockeyAI results:

- Fixed cameras permit stable per-game regions and background references, provided the mount does not move.
- Fisheye distortion and perspective make raw pixel distance a poor substitute for rink distance.
- Correct lens distortion before using a planar rink mapping for distance or angle features.
- Validate that mapping with known ice landmarks and player foot positions.
- Goal-frame boxes are useful references, but do not define a precise goalmouth plane or puck depth.
- Airborne pucks do not lie on the ice plane. A single image cannot reliably prove a goal-line crossing.
- Players, the goalie, mesh, boards, blur, and small image size can remove the puck evidence.
- A crop can preserve useful detail, but cannot restore an occluded or unresolved puck.
- Camera agreement can strengthen timing evidence only where the views show the same play.
- Two opposite-net cameras are not automatically a calibrated stereo system.

Small puck size, blur, occlusion, and viewpoint changes are documented problems in the inspected puck research. [9,10]
Do not assume low-rate sampling catches the instant of release, puck impact, or save contact.
Use sparse sampling for broad candidates; evaluate denser short windows separately.
No universal sampling rate or window duration is established by these sources for GoPro footage.

### Audio without a crowd

The user reports player yelling and post sounds, with PA interference and no useful crowd signal.
Do not use crowd cheering or commentator excitement as required indicators.
Do not restore global audio-loudness weighting on this evidence.

The existing whistle study documents one game's false positives from goalie-post contact, a squeak, and a bench door. [13]
It also records missed whistles and unchecked inferred stoppage times. [13]
Its estimated accuracy is a local study result, not a multi-rink validation.

Proposed audio use:

- Use a whistle to support an endpoint or freeze, never as a condition for a candidate.
- Keep rebound, clear, turnover, and continued-play endings eligible without whistles.
- Mark PA music and speech overlap as uncertain audio evidence.
- Do not reject a visually strong play because PA audio masks the cue.
- Require visible attack context before promoting impact or yelling candidates.
- Keep source labels such as `impact-like` until review establishes post contact.
- Test voice, impact, and whistle cues separately from general audio energy.

No inspected primary source establishes a reliable puck-post sound classifier or player-yelling interest classifier for this setting.

## Practical staged recommendation

These stages are proposed evaluation work, not implementation changes or proven rules.

### Stage 1: define interesting sequences

Use existing clips and caches first.
Label the attack start, peak action, outcome, and end of the useful sequence.
Record whether the outcome is save/freeze, rebound, clear, turnover, continued play, miss, block, goal, or unknown.
Allow multiple outcomes within one scramble.
Separate the play label from the user's keep/reject decision.

Include ordinary skating, faceoff setup, post-whistle gatherings, warmups, and PA interference as negative examples.
Include rejected and quiet windows, not only published highlights.
An answer key built only from selected clips cannot measure missed-play recall.

### Stage 2: propose candidates without whistle or puck requirements

Combine existing net/slot motion with temporal shape, approach direction, and repeated activity peaks.
Use validated HockeyAI player and goalie boxes to add convergence and near-net occupancy.
Treat movement outside the slot as possible buildup, not an automatic rejection.
Use optional visible-puck evidence to strengthen an attack hypothesis.

Keep the whole attack-response-outcome sequence.
End on a stable departure, clear, possession transition, freeze, or user-defined maximum length.
A whistle may help locate an endpoint but does not prove the sequence has ended visually.
Merge duplicate candidates from one shot flurry rather than selecting every motion peak.

### Stage 3: refine only short candidate windows

Compare native-detail crops, denser samples, and simple object tracks on labeled short windows.
Require consistent observations before computing puck direction or assigning possession.
Keep gaps explicit; do not bridge long occlusions as confirmed puck movement.
Evaluate a temporal action model only if simpler cues leave important errors.
Do not start whole-game specialist inference before the small evaluation supports it.

### Stage 4: rank for goal-first publication

Preserve the existing goal obligations and game context before allocating non-goal space. [12]
Rank non-goal candidates by likely interest, event confidence, and visible clip quality.
Keep these quantities separate even if the published interface later combines them.
A chaotic but unreadable clip can have strong motion and poor highlight value.
Do not interpret a rank score as an NHL expected-goals probability.

Favor complete, understandable sequences: a dangerous attack, its response, and its outcome.
Require the existing 80% worth-keeping target for automatic non-goal inclusion on held-out local games. [12]
Leave uncertain candidates available for review rather than inventing precise event names.

## Tests and evidence needed

No tests below were run in this research.

| Question | Required evidence |
|---|---|
| Can HockeyAI provide useful context here? | Local per-class precision/recall for net, goalie, player, and visible puck; thresholds, image size, and sample counts. |
| What happens during occlusion? | Labels for visible, occluded, out-of-frame, and human-uncertain puck states; track-gap errors. |
| Does it find interesting non-goal plays? | Event recall against reviewed continuous periods; selected-clip precision and confidence intervals. |
| Does it preserve plays without whistles? | Separate results for whistle/no-whistle and freeze/rebound/clear/turnover/continued-play endings. |
| Which features help? | Compare motion-only, object context, optional puck, and optional audio at the same clip budget. |
| Does geometry help? | Landmark mapping errors; slot-entry and lateral-movement errors before and after distortion correction. |
| Are distant shots retained? | Recall for point shots that lead to tips, rebounds, saves, blocks, and misses. |
| Are scrambles confused with stoppages? | Hard negatives containing faceoffs, post-whistle traffic, goalie adjustments, and celebrations. |
| Does audio add value? | Reviewed post/voice/whistle examples, PA-overlap negatives, and an audio-disabled comparison. |
| Does the result generalize? | Split by game and rink, with camera angle, light, jersey, and mount variation represented. |
| Are clips watchable and distinct? | User keep rate, boundary corrections, incomplete outcomes, and duplicates per play. |
| Does goal-first ranking remain intact? | Regression against existing goal answer keys and the same total Recap budget. |
| Is added processing affordable? | Measured latency, memory, decoded frames, and device details for short windows. |

Agree on minimum sample counts and acceptable uncertainty before using automatic inclusion results.
Do not tune thresholds on the same games used to report the final keep rate.

## Unresolved questions

1. Can the authors provide the HockeyAI paper's full tables, exact split membership, training settings, and weight-to-metric correspondence?
2. Which local cues reliably distinguish a shot, a pass, and ordinary stick movement from this view?
3. Can team membership and possession be inferred with enough confidence to name defensive stops and turnovers?
4. Does the user prefer every strong chance, or only plays with a clear save, near miss, or defensive outcome?
5. How much buildup and continued play makes a scramble clip useful rather than too long?
6. Do player yelling and puck-post impacts improve selection after controlling for the visual cues and PA interference?
7. Is higher-resolution short-window puck analysis useful enough to justify its cost?

## Primary sources and inspection limits

All accessible sources below were inspected on 2026-10-03.
The research used primary sources for substantive findings.
Bibliographic search services helped locate the HockeyAI paper; their summaries were not used as performance evidence.
An authorized research agent inspected the sources and wrote this note.

1. **HockeyAI official model card.** Model type, seven objects, and incomplete performance claims.
   <https://huggingface.co/SimulaMet-HOST/HockeyAI/blob/main/README.md>
2. **HockeyAI official dataset card and metadata.** Broadcast domain, labels, resolution, and packaging differences.
   <https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI/blob/main/README.md>
   <https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI/blob/main/dataset-info.json>
   <https://huggingface.co/datasets/SimulaMet-HOST/HockeyAI/blob/main/dataset-metadata.json>
3. **HockeyAI official demo source.** Single-image inference and plotted predictions.
   <https://huggingface.co/spaces/SimulaMet-HOST/HockeyAI/blob/main/app.py>
4. **HockeyAI official evaluation figures.** Inspectable PR legend and aggregate training curves; no local GoPro benchmark.
   <https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/Hockey_AI_PR_curve.png>
   <https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/HockeyAI_training_results.png>
5. **NHL official definitions.** Shots on goal, saves, post hits, and geometric shot regions.
   <https://www.nhl.com/info/hockey-glossary>
   <https://www.nhl.com/nhl-edge/glossary>
6. **Natural Stat Trick provider glossary.** Shot attempts, scoring chances, rebounds, and rush definitions.
   <https://www.naturalstattrick.com/glossary.php?teams>
7. **MoneyPuck provider documentation.** Shot features, rebound creation, flurries, and probability-based danger bands.
   <https://moneypuck.com/about.htm>
   <https://moneypuck.com/glossary.htm>
8. **Younggren and Younggren, Evolving-Hockey.** First-party shot-model features, code, and limitations for cross-crease and behind-net passes.
   <https://evolving-hockey.com/blog/a-new-expected-goals-model-for-predicting-goals-in-the-nhl/>
9. **Vats et al., CVPR Workshops 2021.** Full paper inspected; event classes, approximate puck targets, evaluation definitions, and results.
   <https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Vats_Puck_Localization_and_Multi-Task_Event_Recognition_in_Broadcast_Hockey_Videos_CVPRW_2021_paper.html>
   <https://arxiv.org/html/2105.10563v1>
10. **Salass et al., 2025, PLUCC.** Full paper inspected; contextual localization, proprietary data, metrics, and processing cost.
    <https://arxiv.org/html/2506.04365v1>
11. **Cai et al., HARPET, 2018 preprint.** Full paper inspected; temporal actions, stick/pose cues, split, and results.
    <https://arxiv.org/html/1812.09533v1>
12. **Project decision and recorded local evidence.** Not an external benchmark; measurements were not rerun.
    [Game-Sheet-First Selection](../adr/0003-game-sheet-first-selection.md)
13. **Project whistle study.** One-game observations and explicit limits; audio was not reprocessed.
    [Whistle and stoppage detection](whistle-detection.md)
14. **Sarkhoosh et al., HockeyAI, ACM MMSys 2025, pp. 228–234.** Publisher page, author abstract, and references inspected.
    <https://doi.org/10.1145/3712676.3718335>
    The full-text and PDF endpoints returned HTTP 403. Full paper tables and split methods remain unverified.
15. **Official repository listings.** Used to check available artifacts, not to inspect ZIP or weight contents.
    <https://huggingface.co/api/models/SimulaMet-HOST/HockeyAI>
    <https://huggingface.co/api/datasets/SimulaMet-HOST/HockeyAI>
    Model revision: `438c4d6e430bcf0bb5eb048727fadac0cf6b1058`.
    Dataset revision: `2f7b6319c07a6a8f08d2b29fa72e0ea65f9a9929`.

The inspected sources establish no performance result for automatic interesting non-goal detection on fixed behind-net amateur GoPro footage.
