# Cue Library

The Cue Library contains hand-approved Beds and cues for `audio_mix.py`.
No tracks are approved yet. `v3/cues/manifest.json` is an empty schema-1 template.
The tests create small synthetic WAV files. They do not establish approval for real tracks.

## Create the library

Run these commands from the project root:

```bash
.venv/bin/python v3/scripts/cue_library.py init ~/hockey/cues
.venv/bin/python v3/scripts/cue_library.py validate ~/hockey/cues/manifest.json
```

Initialization creates this structure without audio:

```text
~/hockey/cues/
  manifest.json
  beds/             # Stable track names; mood subfolders are optional.
  cues/             # Fixed cue IDs; one approved file per ID.
  provenance/       # Saved licenses and source evidence.
  rotation.json     # Created after the first successful mix with Beds.
```

An empty or partial library passes validation, but the tool reports each missing cue.
The mixer skips missing cues in manifest mode. Its existing no-manifest placeholder mode remains available.

## Approve and register a track

Use only these sources:

- YouTube Audio Library.
- Pixabay tracks without Content ID registration.
- Freesound tracks with a CC0 license.

Check the track page, license, and Content ID status before approval.
Save the source evidence under `provenance/`. Complete any required attribution for the source license.
The tool checks the approval record and media hash. It cannot determine a track's Content ID status from audio.

Place the approved media inside the library. Use stable Bed IDs, rather than period names, to support rotation.
Trim Beds to start on a checked downbeat before approval. The mixer loops complete bars from the start of each file.

Create metadata like this example. Replace every placeholder before setting `approved` and `content_id_safe` to `true`:

```json
{
  "id": "bed-energy-01",
  "file": "beds/energy/track.wav",
  "source": "Pixabay",
  "content_id_safe": false,
  "approved": false,
  "bpm": 120,
  "beats_per_bar": 4,
  "tags": ["energy"],
  "provenance": {
    "url": "https://pixabay.com/music/REPLACE-WITH-TRACK-PAGE/",
    "license": "REPLACE-WITH-EXACT-LICENSE",
    "proof": "provenance/track-license.txt",
    "approved_by": "REPLACE-WITH-REVIEWER",
    "approved_at": "YYYY-MM-DD"
  }
}
```

Register the completed metadata:

```bash
.venv/bin/python v3/scripts/cue_library.py add ~/hockey/cues/manifest.json beds /path/to/track.json
```

The tool measures `duration_s` with ffprobe and records the media's SHA-256 hash.
It writes the manifest only after validation succeeds. It does not download or copy audio.
For an untimed Bed, registration runs `beat_this`. Install the optional dependency before using this path.
An explicit `bar_s` can supply timing without BPM. Registration derives the corresponding regular beat grid.
For cues, use `cues` instead of `beds`. Omit Bed timing fields and use one fixed cue ID below.
Stings must last between two and four seconds. All cues also need source evidence and hand approval.
Freesound entries require the exact source `Freesound CC0` and license `CC0`.

## Schema 1

The manifest has `schema_version: 1`, a `beds` list, and a `cues` list.
Each entry requires a unique `id`, relative `file`, and finite positive `duration_s`.
Files and evidence must remain inside the library, including resolved symlink targets.

The authoring tool requires these additional fields on every entry:

- `source`, `content_id_safe: true`, and `approved: true`.
- `provenance.url`, `provenance.license`, and a relative `provenance.proof` file.
- `provenance.sha256`, `provenance.approved_by`, and ISO-date `provenance.approved_at`.

Each Bed requires `bpm`, `bar_s`, or at least two increasing `beats` timestamps within its duration.
`beats_per_bar` is a positive integer and defaults to four.
Optional `tags` contain strings. Optional `lufs` and cue `gain_db` must be finite numbers.
The tool preserves extra metadata fields when it updates grids.

These authoring checks are stricter than `audio_mix.load_manifest`.
The mixer still accepts the original schema-1 contract and ignores authoring metadata.
Validate the library before using it. Changing approved media bytes requires renewed approval and registration.

The mixer consumes exactly these 14 cue IDs:

```text
game_start           horn
close_win            close_loss          close_neutral
sting_penalty        sting_power_play    sting_fight
sting_comic_call     sting_neutral
sfx_goal_card        sfx_penalty_card
sfx_period_wipe      sfx_final_card
```

## Derive beat grids

For automatic Bed grids, install the optional inference dependency:

```bash
.venv/bin/python -m pip install -r requirements-beats.txt
.venv/bin/python v3/scripts/cue_library.py grid ~/hockey/cues/manifest.json
```

`beat_this` uses the `final0` checkpoint on CPU. The model may download weights on its first run.
Validation and manual BPM do not import `beat_this` or require its dependencies.
Check the detected timing against the approved audio before mixing.

For one manually checked Bed:

```bash
.venv/bin/python v3/scripts/cue_library.py grid ~/hockey/cues/manifest.json \
  --id bed-energy-01 --bpm 120 --beats-per-bar 4
```

Automatic grids retain beat timestamps and derive BPM from their median interval.
Manual BPM creates timestamps from zero to the media end, excluding the end timestamp.
Both methods produce `bpm`, `beats`, `beats_per_bar`, and `bar_s`.
Re-gridding keeps each Bed's stored `beats_per_bar`, or uses four if absent.
Use `--beats-per-bar` to change the meter explicitly.
Normally `bar_s` equals the beat interval times `beats_per_bar`. Use `--bar-s` for a checked override.
The mixer uses `bar_s` first, then `beats`, then `bpm`.
Grid updates preserve provenance and leave the manifest intact if analysis or validation fails.

## Rotate Beds between games

Keep at least six different approved Beds for three-period games.
The mixer selects different Beds per period and avoids all Beds from the previous successful game.
It records selections in `rotation.json` beside the manifest only after output verification and report writing.
Reruns use their saved selections, even if the manifest order changes.
The existing game key combines the output date and the Scoresheet teams.
Games must have distinct keys to receive independent selections.

If too few Beds remain, selection uses all available unused Beds before reusing a track and reports the reuse.
If a saved Bed disappears, selection reports its replacement.
A corrupt history produces a review flag. The mixer uses deterministic selection and preserves the corrupt file.
If history cannot be saved, the verified mix remains available with a flag that the next game may reuse Beds.
Run mixes that share a library sequentially so they observe the previous game's saved selection.

Set `audio.manifest` in per-game options to use another manifest path:

```json
{"audio": {"manifest": "/absolute/path/to/cues/manifest.json"}}
```
