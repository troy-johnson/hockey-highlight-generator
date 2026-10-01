# Whistle and stoppage detection from rink audio (hhg-3r5.29)

Measured on one game: `wild_03012026` (two GoPro cameras, AAC 48 kHz stereo,
4231 s and 5077 s of audio). All times are on the detection timeline (see
below).

## Timeline

`audio_signals.json` and `music_spans.json` use the detection timeline, the same
time base as `events.csv` and `markers.csv`: cam1 concat time, with cam2 moved
by the sync offset. The audio is decoded through each camera's Concat Manifest
with the same `-ss` seek and `# recording` start times as `signals.py`.

## What a referee whistle looks like here

- Spectrograms of 11 whistles in the first 700 s of cam1 (374, 400.7, 418.9, 454,
  473.6, 480, 511.1, 583, 604, 620, 694 s) show one flat band, 150–300 Hz wide,
  between 2.0 and 2.5 kHz. Some have a faint second band at 4.4–5 kHz.
- Duration is 0.3–2.3 s. The pitch stays steady inside one whistle (standard
  deviation of the peak frequency 40 Hz or less on clean whistles).
- Different whistles have different pitch (about 2.1 kHz and about 2.35 kHz in
  this game), so a fixed single frequency is not safe.
- GoPro audio falls off steeply above 4 kHz (about −76 dB at 4 kHz, −98 dB at
  8 kHz), so the fundamental is the useful part.
- YAMNet class "Whistle" (396) peaked at 0.004 on these whistles. It is not
  usable for referee whistles. The scoreboard horn shows as a stack of
  harmonics (about 1.5 kHz fundamental) and YAMNet calls it "Buzzer".

## Detector

1. Decode mono 16 kHz. STFT 512 samples (31 Hz bins), 10 ms hop.
2. Band contrast per frame: mean dB of a 250 Hz band minus the louder of the two
   bands next to it (shifted by band width + 2 bins). Take the best band in
   1.8–3.0 kHz and its frequency.
3. Median filter 90 ms. Runs above 6 dB, joined across gaps below 0.3 s, at
   least 0.3 s long.
4. Keep a run if the median frequency is 1.9–2.7 kHz, peak contrast is 10 dB
   or more, and the frequency standard deviation is 300 Hz or less. A run with
   a standard deviation above 150 Hz must also have a peak of 12 dB or more.
5. Join whistles of the two cameras that are within 0.5 s. Keep whistles heard
   by one camera only, and list the cameras.

Contrast percentiles on 0–700 s of cam1: p50 3.4 dB, p90 6.4, p99 12.0,
p99.9 20.5. Whistle peaks were 14–29 dB. The frequency-steadiness check removes
horns and voices that pass the contrast check (for example the horn at 288.6 s,
sd 275 Hz).

First rule (peak 12 dB, sd below 150 Hz, 0.2 s): 83 whistles. 10 of the 11
hand-found whistles were found. The miss (583 s) had a peak of 17.8 dB, but
crowd noise in the same run made the frequency sd 269 Hz.

### Check by ear (wild_03012026)

The user listened to 4 s of audio at each time.

- Round 1: 15 detected whistles (biased to hard cases): 15 yes. 6 rejected
  runs from a looser rule: 6 yes. So the first rule missed whistles.
- Round 2 (peak 10 dB, sd 300 Hz): 16 new whistles: 9 yes, 1 probably yes,
  3 no (goalie hits the post, a short squeak, a bench door), 2 after the game,
  1 unsure. 12 runs still rejected by a very loose rule: 2 yes.
- Estimates: first rule about 100% precision, 77% recall. Peak 10 / sd 300
  about 94% precision, 92% recall.

The current rule (step 4) is between the two. On the full game it gives 99
whistles (91 on both cameras, 6 cam1 only, 2 cam2 only). It keeps all 30
whistles that the user confirmed and that this detector can find. It drops 3
of the 5 "no" answers. It still misses 2 confirmed whistles (2337.5 s, and
2615.1 s with sd about 370 Hz). Two whistles after the game end (4134.5 s,
4233.0 s) stay in the output; finding the game end is hhg-3r5.30.

## Stoppages

ROI flow (net + slot boxes from `signals.py`) does not show stoppages. After
whistles the combined flow rank stayed high (0.75–0.88 at 454, 480 and 620 s),
because whistles often come while players are at the net. Flow dips happen
every 20–30 s during play too.

Rink audio activity does show stoppages. Activity is the spectral flux in
500–3500 Hz (positive change of log magnitude), 3 s mean, ranked 0..1 within
the game, then averaged over the cameras. After 8 of the 11 whistles it fell
below 0.3 for 10–20 s and jumped at the faceoff.

Rule:

- A whistle starts a stoppage when mean activity from 1 s to 10 s after the
  whistle is below 0.35.
- The stoppage ends at the first time activity stays above 0.6 for 2 s (the
  restart), or after 120 s (`restart_found: false`).
- A later whistle inside a stoppage joins it (`role: in_stoppage`). In this game
  most stoppages have such a whistle about 2 s before the restart, likely the
  faceoff whistle.
- Other whistles get `role: no_stoppage`.

The cached flow is kept only as an extra field (`flow_vs_median`).

Result on the full game (current whistle rule): 46 stoppages; 47 whistles
inside stoppages; 6 whistles without a stoppage. The stoppage times were not
checked by ear or by video.

## Limits

- Tuned and checked on one game and one rink.
- A whistle during loud play or crowd noise can fail the steadiness check.
- Activity rank is relative to the game, so a very quiet or very loud game
  shifts what "low" means less than a fixed level would, but this is untested.
- The audio stage needs the sync stage, and sync needs two cameras. A game
  with one usable camera gets no whistles or stoppages. The code can analyse
  one camera, so a later change can drop that need.
