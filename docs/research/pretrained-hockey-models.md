# Pretrained models for behind-net hockey highlights

Research date: 2026-10-03. Target: fixed, fisheye, behind-net GoPro footage of amateur hockey.

## Recommendation

**Use public pretrained features, then train a small model on the user's selections.**
Start with a temporal candidate ranker using X3D-S and existing motion/object features.
Compare it with a small VideoMAEv2 encoder before choosing a backbone.
Fine-tune the backbone only if held-out game results justify the added work.
Training a complete video encoder from scratch is not justified by the currently described collection.
This is an engineering recommendation, not a measured result.

There are additional downloadable hockey models: **HockeyOrient** and **HockeyRink**. [2,3]
They provide player orientation and rink landmarks, respectively. They do not classify saves or highlight interest.
I found no verified public specialist checkpoint that directly solves shots, saves, rebounds, scrambles, and highlight selection for this view.
That finding is limited to the inspected sources. It does not establish that no such checkpoint exists.

The user's paired raw footage and edits are valuable because they describe both the camera domain and the desired selections.
The first custom target should be **whether a complete play is worth keeping**, with separate event and visibility labels.
An editor's selection is not proof of a particular play type.

## What “available” means here

- **Public artifact:** an official listing contains a named weight file or an official model zoo links to it.
- **Supported inference:** the owner supplies loading code, a model definition, or an inference example.
- **Research only / unknown:** the paper describes a trained model, but this search did not verify a usable checkpoint.
- **Validated locally:** performance was measured on held-out footage from this project. No candidate received that test here.

All public-artifact claims below refer to source inspection, not successful downloads or execution.
No weights, datasets, or archives were downloaded. No model inference, training, installation, or video processing was run.

## Hockey specialists

| Candidate | Artifact and license evidence | Actual output | Fit and decision |
|---|---|---|---|
| **HockeyAI** | Official ungated repository lists `HockeyAI_model_weight.pt`; model card declares MIT. YOLOv8 loading is documented. [1] | Player, goalie, puck, goal frame, referee, faceoff dots, center ice boxes. | Retain for object context. `goal` means the frame, not a scored goal. Broadcast training differs from this camera view. |
| **HockeyOrient** | Official ungated repository lists `best_model_squezenet.pth`; card declares MIT. Official demo reconstructs SqueezeNet 1.1 and loads the state dictionary. [2] | Eight head-orientation classes from a cropped player image. | A real additional checkpoint. Test as optional context after player detection. Head direction does not prove possession, gaze at the puck, skating direction, or a shot. |
| **HockeyRink** | Official ungated repository lists `HockeyRink.pt`; card declares MIT and specifies YOLOv8-Large pose. [3] | 56 rink landmark coordinates with confidence. | A real additional checkpoint. Useful only if enough reliable landmarks are visible. Fisheye correction and mapping checks remain necessary. It is not a player-pose or event model. |
| **HARPET / temporal hockey action model** | Paper and dataset description verified. No author-linked, runnable hockey action checkpoint was verified. Weight/code license unknown. [4] | Forward skating, backward skating, passing, shooting, using player pose, stick keypoints, and flow. | Useful research, not a verified ready-to-run alternative. Individual-player crops and visible sticks are difficult in crowded behind-net footage. No save/rebound/scramble/goal classes. |
| **Vats puck multitask model / “PuckNet”** | Vats paper verified. No runnable official checkpoint or weight license verified. PLUCC calls this earlier approach “PuckNet.” [5,6] | Approximate rink-space puck location plus Faceoff, Advance, Play, Shot. | NHL broadcast training; approximate two-second targets. Not precise frame-level puck tracking or a save/outcome classifier. |
| **PLUCC** | Paper verified; VIP-PuckDataset is explicitly proprietary. Public specialist checkpoint, inference repository, and weight license remain unverified. [6] | Single-frame puck-location heatmap using full-image and segmented-player context. | Interesting future puck option if authors provide artifacts. It requires detection and segmentation upstream, not just its own network. No highlight/event output. |
| **`danbiagini/hockey_breeds_v2`** | Ungated listing contains `hockey_breeds-v2-101623.pt`; metadata declares MIT and YOLOv8. [7] | Card does not document classes, data, metrics, or a loading recipe. | A public community artifact, but insufficient evidence to recommend it over HockeyAI. Weight class mapping and usability were not inspected. |
| **`erwanln/videomae-base-finetuned-soccer-hockey`** | Listing contains only `.gitattributes`, with zero stored bytes. [8] | No published model artifact inspected. | The name suggests relevance, but this repository supplies no checkpoint. Do not count it as an available hockey action model. |

The HockeyAI, HockeyOrient, and HockeyRink MIT declarations concern their published repositories. [1–3]
The Ultralytics runtime has its own AGPL-3.0 license; a model-card declaration does not replace dependency terms. [17]
License declarations are source evidence, not a legal clearance for every training asset or distribution arrangement.

### Important specialist details

**HockeyOrient:** the dataset card describes 9,700 annotated player crops from the 2023–2024 SHL seasons. [2]
The model card reports about 75% accuracy/F1, without a documented local GoPro evaluation. [2]
The demo's label order differs from the card's prose order. Use the demo mapping when reproducing its loading procedure.
Verify the detector's player label before reproducing the demo's hardcoded class ID.
Image-space head orientation is especially sensitive to a new viewpoint and fisheye distortion.

**HockeyRink:** the model card describes SHL training footage and broadcast examples. [3]
It reports hardware speeds, including an M3 MacBook Pro result, but these were not reproduced here.
Those figures do not predict this project's runtime or landmark accuracy.
For a fixed camera, an occasional reliable calibration could be more useful than per-frame landmark inference.
That is a proposed use, not an established model capability on GoPro footage.

**HARPET:** the paper contains 424 three-image sequences from internet clips and instructional DVDs. [4]
Its random 70/15/15 split does not establish generalization to a new game or rink.
The reported 83.60% average action accuracy must not become a projected save or highlight accuracy.
The paper's use of pretrained LiteFlowNet does not establish release of the trained hockey classifier.

**Vats/PuckNet:** targets locate the puck approximately for an entire two-second clip. [5]
The reported 73.1% AUC integrates rink-distance tolerances from 5 to 50 feet.
It is not puck-box recall or goal-line crossing accuracy. Multitask Shot F1 is 58.4% on that dataset. [5]
The inspected GitHub repositories named `PuckNet` were unrelated projects; name matching is not checkpoint verification. [19]

**PLUCC:** the paper reports 114 million trainable parameters and H100 training. [6]
Table 1 reports about 53 FPS for the network alone on RTX6000, but about 6 FPS including context generation.
Neither speed describes the user's machine. The proprietary 150,000-image dataset is not a public training resource. [6]
The paper's distance-based AP differs from HockeyAI box AP and Vats tolerance AUC.
The paper's CC-BY publication license is not evidence of a weight or dataset license.

### “Sport” is not a verified hockey checkpoint name

No specific model named “Sport” was verified for this task.
**SportsMOT** is a player-tracking dataset covering basketball, football, and volleyball, not hockey. [16]
Its owner provides evaluation code and links to MixSort; that is not evidence of hockey event weights.
Its dataset license is CC-BY-NC-4.0, with access and redistribution terms. [16]
If another “Sport” candidate was intended, its exact owner or URL is needed to evaluate it.

## General pretrained features with supported loading paths

These are reusable encoders or generic action classifiers, not pretrained hockey outcome models.
Their published benchmark scores do not establish save recognition or the user's preferred highlight selections.

| Candidate | Exact practical checkpoint | Access and license evidence | Practical role and constraints |
|---|---|---|---|
| **X3D-S / X3D-M** | `X3D_S.pyth`, `X3D_M.pyth`; official PyTorchVideo model zoo and Hub support. [9] | Public first-party links. Repository license is Apache-2.0; a separate checkpoint-specific license was not found in the inspected zoo. [9] | Best first temporal baseline by documented model cost. Both list 3.79M parameters; single-view costs are 2.96G and 6.72G FLOPs. Train a local head on extracted features. |
| **SlowFast R50** | `SLOWFAST_8x8_R50.pyth`; `slowfast_r50` Hub loading is documented. [9] | Public first-party link; same repository-license evidence and checkpoint-license limit as X3D. | Useful motion comparison, not first choice for cost. Zoo lists 34.57M parameters and 65.71G single-view FLOPs. Requires the correct slow/fast pathway preprocessing. |
| **VideoMAE ViT-S** | K400 ViT-S, 1600-epoch pretrained checkpoint, official Drive ID `1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww`. [10] | Official zoo link; Drive access was not exercised. Original repository LICENSE is CC-BY-NC-4.0. A separate checkpoint grant was not verified. [10] | Valid feature/fine-tuning candidate for permitted use. Prefer the smaller V2 option below when its release terms fit better. Pretrained and action-fine-tuned weights are separate artifacts. |
| **VideoMAEv2 ViT-S distilled** | `OpenGVLab/VideoMAE2/distill/vit_s_k710_dl_from_giant.pth`. [11] | Official ungated Hugging Face listing and direct zoo link. HF card declares Apache-2.0; code repository LICENSE is MIT. | Best first transformer comparison. Published recipe uses 16 frames at 224×224. It learns generic action features, not hockey interest. Giant variants are unnecessary for the first benchmark. |
| **VideoPrism-B** | `google/videoprism-base-f16r288`; config `videoprism_public_v1_base`. [12] | Public official card declares Apache-2.0. Repository states Apache-2.0 for software and CC-BY for other materials; scope needs confirmation before redistribution. | Strong frozen-feature alternative. 114M parameters; documented checkpoint size 458MB. Official inference uses JAX/Flax; PyTorch support remains a listed TODO. Extra runtime integration makes it a second-stage choice here. |
| **VideoPrism-LvT-B** | `google/videoprism-lvt-base-f16r288`; config `videoprism_lvt_public_v1_base`. [12] | Official release and video-text loading example. Checkpoint-specific card terms were not separately inspected; repository terms apply as documented above. | Can compare text and video embeddings for exploratory retrieval. Prompt similarity is not a validated hockey event probability or keep score. 248M parameters, documented 991MB checkpoint. |
| **V-JEPA 2 ViT-L** | `facebook/vjepa2-vitl-fpc64-256`, or first-party `vitl.pt`. [13] | Official HF card declares MIT; repo documents public PyTorch/Hub/Transformers loading. | Frozen temporal features with a custom probe. 300M parameters and a documented 64-frame HF input. Heavier than the first candidates; published probes target SSv2, Diving48, and EPIC-KITCHENS, not hockey. |
| **V-JEPA 2.1 ViT-B** | `vjepa2_1_vitb_dist_vitG_384.pt`; Hub entry `vjepa2_1_vit_base_384`. [13] | Current official README lists the checkpoint and Hub entry; repository predominantly MIT with named Apache-2.0 portions. Separate weight-specific terms were not verified. | Current smaller alternative worth a later comparison: 80M parameters, 384 resolution. No inspected behind-net benchmark. The README announces its March 2026 release. |
| **DINOv2 ViT-S/14** | `dinov2_vits14_pretrain.pth`; Hub entry `dinov2_vits14`. [14] | Official repository explicitly licenses both code and model weights under Apache-2.0. Public first-party link. | Useful frame matching, image embeddings, and appearance baseline. 21M parameters. It is an image encoder; a sequence head is needed to model attack-response-outcome timing. |

FLOPs describe arithmetic work, not elapsed runtime. Published multi-view evaluation multiplies the single-view work. [9]
Cached frozen features avoid repeating encoder inference during small-head experiments.
Training only the head reduces training work, but does not remove encoder inference cost.
GPU memory, decoding cost, macOS support, and throughput require a local test.
The V-JEPA repository explicitly identifies macOS problems with its `decord` dependency. [13]
This does not prove that its encoder cannot run with a different decoding path.

VideoPrism's current public weights differ from the paper's original weights. The owner documents that distinction. [12]
DINOv2's README points to newer DINOv3 work, but DINOv3 artifacts and terms were not assessed here. [14]
The table selects supported practical candidates; it is not a complete catalog of all video foundation models.

## Objects, tracking, events, and interest are separate tasks

For this project, a proposed division is:

1. Use HockeyAI boxes for net, goalie, and player context.
2. Associate player/goalie boxes across short windows when trajectories add useful information.
3. Use video features to represent the buildup, response, and outcome.
4. Train a local ranker to predict the user's keep decision.
5. Add event-specific heads only where reviewed labels support them.

ByteTrack documents association of boxes from an external detector. It does not require adopting its pedestrian detector weights. [15]
Its bundled public detector checkpoints target pedestrian datasets, not goalies or pucks.
This report recommends testing the association method with HockeyAI boxes, not treating ByteTrack as a pretrained hockey model.
Association-only use has no separate learned checkpoint; dependency/license integration was not assessed.

Crowded scrambles can cause identity switches. A tracker cannot restore absent visual evidence.
Treat inferred puck positions during occlusion as estimates, not observations.
Puck detection should remain optional for candidate selection until visible-puck evaluation supports stronger use.
Keep missing, occluded, out-of-frame, and human-uncertain states separate.
Team assignment, player identities, stick contact, and possession do not follow automatically from boxes or head direction.

Broadcast-trained and web-trained models face different player scale, net obstruction, lens distortion, and event visibility here.
Low-resolution full-frame encoders can lose puck detail. Test a net-area crop alongside the full view.
Keep enough context to retain distant shots and the approach to the net.
These are engineering concerns to measure, not demonstrated failure rates for the listed weights.

## Using paired raw footage and edited highlights

### Convert edits into source-aligned labels

Prefer Resolve timelines, EDLs, XMLs, or other source-cut records when they exist.
Otherwise, propose raw-frame matches using image features, then verify each match and its temporal continuity.
DINOv2 offers a public image-feature option for this proposed matching step. [14]
Retain raw recording IDs, camera IDs, chapter offsets, source intervals, and alignment confidence.
Support multiple source intervals within one edited segment.

Edited highlights need normalization before they become training labels:

- Match repeated shots and replays to the same source play.
- Map slow motion and speed ramps back to source time.
- Account for crops, zooms, transitions, and camera switches.
- Exclude titles and graphics from visual training inputs.
- Use raw audio only if a separate audio experiment needs it.
- Do not train on added music or commentary absent from the source pipeline.

Train on aligned **raw clips**, not rendered highlight frames mixed with unedited negatives.
Otherwise, the model could learn editing style, score overlays, music, or compression instead of interesting play.
Avoid labeling a complete selected clip as one action: it contains buildup, the peak, and sometimes unrelated tail footage.

### Label selection separately from play type

Published selections are **weak positives** for editorial interest.
Unselected raw footage is **unlabeled**, not automatically bad: it may contain omitted good plays or poor views of good plays.
The clip budget, focus team, duplicate angle, or missing buildup can affect selection.

Use three explicit selection states: keep, reject after review, and uncertain/unreviewed.
For reviewed plays, record attack start, peak, outcome, useful end, and visibility.
Add multi-label event tags such as shot-like attempt, save, rebound, scramble, goal, clear, block, miss, and unknown.
A scramble can contain several attempts and outcomes.
Do not invent event labels from a model's embedding similarity.

Review hard negatives: ordinary skating, goalie adjustments, warmups, faceoff setup, post-whistle gatherings, and empty-net-area movement.
Include good plays absent from the published edit and poor clips of genuine goals.
Pairwise keep preferences can help rank similar candidates, but an unreviewed omission is not a reliable preference label.

### Split by the source event and whole game

Keep both camera views, adjacent windows, replays, and transformed copies of one play in the same split.
Split training, validation, and test by entire game, not randomly by frame or short clip.
Also evaluate a held-out rink or changed camera mount when the collection supports it.
If all games share one rink and angle, report that limit rather than claiming general camera-domain performance.
Fit normalization and thresholds on training/validation data only.

The ADR expects 10–15 paired games, but this research did not inventory the user's collection. [18]
There is no supported minimum game count, accuracy prediction, or training-time estimate for this footage.
Neighboring frames do not replace independent games or rare event examples.
Use learning curves and game-level uncertainty to decide whether more labels or a larger model would help.

## Proposed small benchmark before training expansion

This benchmark was not run. Its purpose is to identify the actual bottleneck at limited cost.

1. Inventory paired games, source-cut records, rink/mount variation, and available hardware.
2. Build a reviewed sample across games, both cameras, desired plays, and hard negatives.
3. Include reviewed continuous periods to measure interesting plays that candidate generation misses.
4. Reserve entire games before fitting any model or threshold.
5. Measure the current motion/object baseline at the existing Recap budget.
6. Compare frozen X3D-S features plus a small head against that baseline.
7. Compare frozen VideoMAEv2-S features plus the same labels and budget.
8. Measure full-view versus net-crop inputs and complete-play versus short-peak windows.
9. Review uncertainty and errors before increasing model size or fine-tuning.

Report these quantities separately:

| Question | Evidence |
|---|---|
| Does candidate generation miss useful plays? | Recall on reviewed continuous footage, including distant attempts and plays without whistles. |
| Does ranking improve selections? | Worth-keeping precision at the same clip/time budget; precision-recall curve and per-game results. |
| Does it recognize specific events? | Per-tag results with unknown outcomes preserved; do not substitute overall accuracy. |
| Are clips complete? | Boundary corrections, missing buildup/outcome, duplicate plays, and camera quality. |
| Do objects help? | Motion-only versus objects versus objects plus video features; visible-puck precision/recall separately. |
| Is compute practical? | Device, input resolution, sampling, frames decoded, wall time, peak memory, and caching policy. |
| Does it generalize? | Whole-game results, camera/mount/rink subsets, and uncertainty based on independent games. |

A ranker cannot recover plays that never become candidates.
If recall is the bottleneck, broaden candidate generation or test sparse temporal scanning on a limited reviewed period.
Do not start whole-game foundation-model inference merely because a checkpoint is public.
If player/goalie boxes are the bottleneck, annotate representative raw frames and fine-tune the detector separately.
Highlight intervals alone do not supply bounding-box, puck-center, or track-identity annotations.
If outcome recognition is the bottleneck, add reviewed event labels and fine-tune a small temporal encoder.

Keep Scoresheet-first goal obligations and the ADR's 80% non-goal worth-keeping target. [18]
The 80% value is an acceptance target, not an achieved result for any candidate.
Use game context for goal priority and visual evidence for non-goal interest, as the ADR specifies.
Optional whistle timing can support boundaries; do not require a whistle or restore global audio-energy weighting.
This preserves the conclusions of [non-goal-play-detection.md](non-goal-play-detection.md).

## Does a custom model make sense?

**Yes: a custom ranking head or a fine-tuned model makes sense as an experiment.**
The paired collection supplies unusually relevant training material for both this view and this editor's preferences.
Begin with a frozen encoder and a regularized linear or small temporal head.
Keep object/motion features as a comparison and as optional inputs.
Then use reviewed uncertain candidates and false positives to choose further annotation work.
Do not use the held-out test games for this active-learning loop.

Fine-tune a small encoder when enough independent reviewed examples exist and its learning curve supports the change.
Use local detector fine-tuning when object localization fails, not as a substitute for training highlight interest.
Consider self-supervised adaptation on training-game raw footage only after the simpler methods show a clear representation problem.
Several games do not establish the case for foundation-model pretraining from scratch.
The data count and diversity remain unknown, so a stronger sufficiency claim would be unsupported.

## Primary sources and verification limits

The links below were inspected on 2026-10-03. Official artifacts, code, papers, and licenses support the factual candidate claims.
GitHub and Hugging Face search APIs served as discovery tools; a search result alone was not treated as a validated model.
No background-agent tool was available; this research was performed directly.
Author contact, private releases, every community repository, and all possible commercial systems were not checked.
Weight integrity, actual binary contents, successful loading, dependencies, and local performance remain unverified.

1. **HockeyAI:** [model card](https://huggingface.co/SimulaMet-HOST/HockeyAI/blob/main/README.md), [artifact listing](https://huggingface.co/api/models/SimulaMet-HOST/HockeyAI). Inspected revision `438c4d6e430bcf0bb5eb048727fadac0cf6b1058`.
2. **HockeyOrient:** [model card](https://huggingface.co/SimulaMet-HOST/HockeyOrient/blob/main/README.md), [artifact listing](https://huggingface.co/api/models/SimulaMet-HOST/HockeyOrient), [dataset card](https://huggingface.co/datasets/SimulaMet-HOST/HockeyOrient/blob/main/README.md), [inference source](https://huggingface.co/spaces/SimulaMet-HOST/HockeyOrient/blob/main/app.py). Model revision `0d4e7c0a5b6833e12b03df1e47945234359affb2`.
3. **HockeyRink:** [model card](https://huggingface.co/SimulaMet-HOST/HockeyRink/blob/main/README.md), [artifact listing](https://huggingface.co/api/models/SimulaMet-HOST/HockeyRink). Revision `1d738004707d7dc7c4a4a15897ac544e273c7658`.
4. **Cai et al., HARPET:** [full paper](https://arxiv.org/html/1812.09533v1). Additional discovery: [GitHub repository search](https://api.github.com/search/repositories?q=HARPET&per_page=5). This search did not verify trained hockey weights.
5. **Vats et al., puck localization and multitask recognition:** [full paper](https://arxiv.org/html/2105.10563v1). Additional discovery: [repository search](https://api.github.com/search/repositories?q=hockey+puck+Vats&per_page=5).
6. **Salass et al., PLUCC:** [full paper](https://arxiv.org/html/2506.04365v1), especially §§2.1, 4.1–4.2 and Tables 1–2. Additional discovery: [repository search](https://api.github.com/search/repositories?q=PLUCC&per_page=5).
7. **Community YOLO artifact:** [card](https://huggingface.co/danbiagini/hockey_breeds_v2/blob/main/README.md), [listing](https://huggingface.co/api/models/danbiagini/hockey_breeds_v2). The referenced Space's `app.py` endpoint returned 404; no classes or inference behavior were established.
8. **Empty hockey-named VideoMAE repository:** [listing](https://huggingface.co/api/models/erwanln/videomae-base-finetuned-soccer-hockey).
9. **PyTorchVideo:** [official model zoo](https://github.com/facebookresearch/pytorchvideo/blob/main/docs/source/model_zoo.md), [Hub inference notebook](https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/torchhub_inference_tutorial.ipynb), [repository license](https://github.com/facebookresearch/pytorchvideo/blob/main/LICENSE). Exact checkpoints: [X3D-S](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth), [X3D-M](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth), [SlowFast R50](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOWFAST_8x8_R50.pyth). Checkpoint URLs were read from the zoo, not downloaded. Context7 also confirmed the documented Hub loading path.
10. **VideoMAE:** [official model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md), [repository license](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE), [selected pretrained ViT-S artifact link](https://drive.google.com/file/d/1nU-H1u3eJ-VuyCveU7v-WIOcAVxs5Hww/view).
11. **VideoMAEv2:** [README](https://github.com/OpenGVLab/VideoMAEv2/blob/master/README.md), [model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md), [HF card](https://huggingface.co/OpenGVLab/VideoMAE2/blob/main/README.md), [artifact listing](https://huggingface.co/api/models/OpenGVLab/VideoMAE2), [code license](https://github.com/OpenGVLab/VideoMAEv2/blob/master/LICENSE), [selected checkpoint](https://huggingface.co/OpenGVLab/VideoMAE2/blob/main/distill/vit_s_k710_dl_from_giant.pth). HF revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`; listing is ungated. Older zoo text mentions a request form for giant weights, but current official README points to the public HF release.
12. **VideoPrism:** [official README, loading examples, variants, and terms](https://github.com/google-deepmind/videoprism/blob/main/README.md), [base checkpoint card](https://huggingface.co/google/videoprism-base-f16r288/blob/main/README.md), [video-text base release](https://huggingface.co/google/videoprism-lvt-base-f16r288).
13. **V-JEPA 2 and 2.1:** [current official README](https://github.com/facebookresearch/vjepa2/blob/main/README.md), [ViT-L HF card](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/README.md), [ViT-L checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt), [2.1 ViT-B checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt). Direct checkpoint links were inspected in README text, not downloaded.
14. **DINOv2:** [official README, weights, license statement](https://github.com/facebookresearch/dinov2/blob/main/README.md), [selected ViT-S weight](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth). Weight link was read from README, not downloaded.
15. **ByteTrack:** [official README](https://github.com/ifzhang/ByteTrack/blob/main/README.md), especially “Combining BYTE with other detectors.”
16. **SportsMOT:** [official README](https://github.com/MCG-NJU/SportsMOT/blob/main/README.md). Dataset scope, access instructions, terms, and license inspected.
17. **Ultralytics runtime:** [official license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
18. **Project policy and recorded observations:** [ADR-0003](../adr/0003-game-sheet-first-selection.md). Its one-game HockeyAI and flow observations were read, not rerun. Its 18% puck-detection frequency is not a visible-puck recall benchmark.
19. **Discovery checks:** [PuckNet repository search](https://api.github.com/search/repositories?q=PuckNet), [hockey HF search](https://huggingface.co/api/models?search=hockey&limit=30), [hockey pretrained repository search](https://api.github.com/search/repositories?q=%22hockey%22+%22pretrained%22&per_page=5). These limited searches do not prove global absence of another model.
