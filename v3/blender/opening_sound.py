"""Black Ice opening sound, v12 (hhg-3r5.86). Output frames at 30 fps, 148 frames.

A rebuild after a review against professional stingers (user: the break sounds like
cracking, not glass; the scene 2 lake-ice 'lasers' are real but not separate from the
break; the whole video sounds contained, as if in a building, but the scene is outdoors).

- Space: an outdoor winter night (378048, light wind 352436), no room reverb at all; the
  foley is the driest available (249833); a faint valley echo only under the two big hits.
- One sound family per story beat: wake-up = skates and puck on open ice; heat = real
  lake-ice 'lasers' (737896 chirps) over the sting's riser and the hiss of hot metal;
  breakthrough = glass (861043, 981, 853703, 422633) on one short crack (426204);
  debris = tonal ice tinkles (735353); reveal = the steel crest settling, steam.
- Contrast: everything drops away for two frames before the cut (f54-56), then the impact.
- Sync (v12, new retime): the cut (f56) is story 80, the breakthrough itself (plates leave
  at story 79.7-81.5 = out 56-57.8), so the impact and a sub hit land on the cut (v7 review:
  the energy dropped at the cut). Plates: into the water at 63.8, 66.5, 67.3, 68.1; onto the
  ice at 72.5 and 77.5 (sync_probe). Reveal (story 100) at 80.5; fizzles at 85.1, 93.3,
  100.5, 107.7 (Scene 4 now 0.969x).
- Three bands at the impact: sub (sting drop), body (one crack), top (glass), then a tail.

Drafts use Freesound HQ previews where the original is not downloaded yet.
Usage: python opening_sound.py OUT.wav
The source sounds are on the NAS (FX below); the drafts' Freesound previews are in its
ice_dl/, glass_dl/ and sting_dl/ folders.
"""
import subprocess
import sys

import numpy as np
from scipy.signal import butter, sosfilt

SR, FPS, FRAMES = 48000, 30, 148
N = round(FRAMES / FPS * SR)
HIT = 56.0                        # the impact frame: the cut
FX = "/Volumes/Partition_1/Ext. Projects/Ice Pak/Black Ice Graphics/opening/soundfx/"
ICE, GLASS, STING = FX + "ice_dl/", FX + "glass_dl/", FX + "sting_dl/"
BUS = {name: np.zeros((N, 2)) for name in ("amb", "foley", "ice", "heat", "glass", "music", "echo")}
_cache = {}


def sec(frame):
    return frame / FPS


def db(g):
    return 10 ** (g / 20)


def at(frame):
    return round(sec(frame) * SR)


def _read(path, ch):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(float)
    x = x.reshape(-1, 2) if ch == 2 else x
    return x / np.max(np.abs(x))


def load(name, stereo=False):
    path = name if name.startswith("/") else FX + name
    key = (path, stereo)
    if key not in _cache:
        _cache[key] = _read(path, 2 if stereo else 1)
    return _cache[key]


def cut(x, a, b, fade=0.005, fade_out=0.01):
    y = x[round(a * SR):round(b * SR)].copy()
    for f, head in ((min(round(fade * SR), len(y) // 2), True), (min(round(fade_out * SR), len(y) // 2), False)):
        if f:
            ramp = np.linspace(0, 1, f) if head else np.linspace(1, 0, f) ** 2
            if y.ndim == 2:
                ramp = ramp[:, None]
            if head:
                y[:f] *= ramp
            else:
                y[-f:] *= ramp
    return y


def filt(x, kind, hz, order=2):
    return sosfilt(butter(order, hz, btype=kind, fs=SR, output="sos"), x, axis=0)


def shelf(x, kind, hz, gain_db):
    return x + (db(gain_db) - 1) * filt(x, kind, hz, 2)


def pan(x, p):
    m = x.mean(1) if x.ndim == 2 else x
    return np.column_stack([m * np.sqrt((1 - p) / 2), m * np.sqrt((1 + p) / 2)])


def place(x, start_s, gain_db, bus, p=0.0, echo=None):
    if x.ndim == 1:
        x = pan(x, p)
    i = round(start_s * SR)
    a, b = max(0, i), min(N, i + len(x))
    if b > a:
        BUS[bus][a:b] += x[a - i:b - i] * db(gain_db)
        if echo is not None:
            BUS["echo"][a:b] += x[a - i:b - i] * db(gain_db + echo)


def hit(name, onset, frame, gain_db, bus, p=0.0, a=None, b=None, tail=0.01, echo=None, prep=None, stereo=False):
    """Place a file so its onset (seconds in the file) lands on an output frame."""
    x = load(name, stereo)
    a = 0.0 if a is None else a
    b = len(x) / SR if b is None else b
    y = cut(x, a, b, fade_out=tail)
    if prep:
        y = prep(y)
    place(y, sec(frame) - (onset - a), gain_db, bus, p, echo)


# ---------------------------------------------------------------- the open air
# A still winter night on the pond; a breath of wind that rises a little with the heat.
night = load(GLASS + "g_378048.mp3", stereo=True)
place(cut(night, 4.0, 4.0 + N / SR, 0.3, 0.05)[:N], 0, -19, "amb")   # a still night: barely there
wind = filt(cut(load(GLASS + "g_352436.mp3", stereo=True), 10.0, 10.0 + N / SR, 0.4, 0.05)[:N], "lowpass", 1800)
wind *= np.interp(np.arange(N), [0, at(22), at(56), at(80.5), N], [0.5, 0.6, 1.0, 0.7, 0.7])[:, None]
place(wind, 0, -28, "amb")

# ---------------------------------------------------------------- the sting
# Its riser climbs through the heat; the drop (2.95 s in the file, where the riser's top cuts
# off) lands on the impact. It breathes out at the cut, comes back as the drop, then falls away
# so the hero hold stays quiet; its sub clears after the first hit so the glass tail is clean.
STING_DROP = 2.95
sting = load(STING + "px/px_562152.mp3", stereo=True)
sting = filt(sting[round((STING_DROP - sec(HIT)) * SR):][:N], "highpass", 45, 2)
# Under the impact the sting gives only weight (its sub, below 150 Hz) so the glass leads;
# after it, only its upper drone (above 120 Hz) carries on under the tinkles and the reveal.
sub = filt(sting, "lowpass", 150, 4)
thin = filt(sting, "highpass", 120, 4)
w_sub = np.interp(np.arange(N), [at(f) for f in (0, HIT - 0.3, HIT, HIT + 10.4, HIT + 16.4)], [0, 0, 1, 1, 0])[:, None]
w_thin = np.interp(np.arange(N), [at(f) for f in (0, HIT + 10.4, HIT + 16.4)], [0, 0, 1])[:, None]
sting = sting * (1 - w_sub - w_thin) + sub * w_sub + thin * w_thin
# The build stays under the hit (a professional hit stands well above its build).
duck = np.interp(np.arange(N), [at(f) for f in (0, 40, 52, 53.3, 53.9, 55.75, HIT, 62.4, 80.5, 97.5, 147)],
                 [db(v) for v in (-14, -12, -9, -8, -26, -26, -2, -6, -10, -15, -17)])
place(sting * duck[:, None], 0, -4, "music")
# The sub hit on the cut: a short sine drop (58 -> 34 Hz) under the sting's own sub, so the
# cut lands as a blow in the chest, not only as glass.
t = np.arange(round(1.2 * SR)) / SR
boom = np.sin(2 * np.pi * np.cumsum(34 + 24 * np.exp(-t / 0.12)) / SR) * np.exp(-t / 0.42)
boom *= np.minimum(1, t / 0.003)
place(boom, sec(HIT), -5, "music")

# ---------------------------------------------------------------- 1. wake-up (f0-21)
# Skates and a puck on open ice: the driest recordings (249833 stops in 5-10 ms).
SKATE = "249833__splodge123__skate-fx.mp3"
hit(SKATE, 43.4, 6, -14, "foley", 0.45, 43.35, 43.75, tail=0.15)          # a stride, far right
hit(SKATE, 43.8, 17, -16, "foley", 0.25, 43.75, 44.15, tail=0.15)         # the next
hit("141284__mannhawks__stone-on-thin-ice.wav", 0.34, 8.5, -10, "foley", -0.3, 0.2, 1.6, tail=0.4)   # the puck
hit(SKATE, 30.3, 12, -11, "foley", 0.2, 30.25, 31.0, tail=0.3)            # the carve over the crest
hit("826199__audioalchemydesigns__ice-hockey-sound-library_stick-handles.wav", 0.95, 12.5, -18, "foley", -0.2,
    0.85, 1.1, tail=0.08)                                                  # the stick blade, tail trimmed

# ---------------------------------------------------------------- 2. heat (f22-56)
# Hot metal hisses and sizzles, growing; the lake ice 'sings' under the shoves: the real
# descending chirps of pond ice (737896), small to big, with a faint creak under each.
for name, a, g in (("50620__vibe_crc__metal_hissing.wav", 0.5, -20), ("436470__rabban625__shovelsizzle8_mod.mp3", 0.3, -19)):
    x = cut(load(name), a, a + sec(56 - 22), 0.05, 0.02)
    x = filt(x, "highpass", 600) * np.linspace(0.2, 1, len(x)) ** 1.3
    place(x, sec(22), g, "heat", 0.15)
LAKE = ICE + "ice_737896.mp3"


def sing(y):
    return filt(y, "highpass", 250, 4)


hit(LAKE, 16.91, 27, -6, "ice", -0.3, 16.85, 17.4, tail=0.15, prep=sing)
hit(LAKE, 16.43, 40, -3, "ice", 0.3, 16.37, 16.9, tail=0.15, prep=sing)
hit(LAKE, 9.23, 51, -4, "ice", -0.1, 9.17, 9.95, tail=0.2, prep=sing)
hit(LAKE, 9.71, 53.2, -11, "ice", 0.35, 9.68, 10.1, tail=0.15, prep=sing)
for onset, frame, g, p in ((4.37, 27.4, -16, 0.2), (2.77, 40.4, -14, -0.25), (15.23, 51.3, -12, 0.3)):
    hit(ICE + "ice_855116.mp3", onset, frame, g, "ice", p, onset - 0.03, onset + 0.3, tail=0.1,
        prep=lambda y: filt(y, "highpass", 600, 4))

# ---------------------------------------------------------------- 3. the breath (f54-56)
# Just before the cut, at the brightest glow, everything drops away for two frames.
gap = np.interp(np.arange(N), [at(f) for f in (0, 53.4, 53.9, 55.8, HIT)], [1, 1, db(-22), db(-22), 1])[:, None]
for name in ("amb", "foley", "ice", "heat"):
    BUS[name] *= gap

# ---------------------------------------------------------------- 4. the breakthrough (f56)
# Body: one short crack. Top: glass, dry and bright. Tail: tonal ice tinkles falling away.
hit("426204__kev_durr__chipping-large-ice-block-2.wav", 0.05, HIT, -7, "glass", 0.0, 0.045, 0.33, tail=0.1, echo=-6)
hit(GLASS + "g_861043.mp3", 0.06, HIT, 0, "glass", 0.0, 0.045, 1.06, tail=0.25, echo=-4)   # 15 ms lead-in: the breath stays clean
hit("422633__aurelon__break-something-iceglass.flac", 0.30, HIT + 0.15, -7, "glass", -0.2, 0.25, 1.1, tail=0.2)
hit(GLASS + "g_981.mp3", 0.0, HIT + 0.45, -4, "glass", 0.35, 0.0, 1.77, tail=0.4)
hit(GLASS + "g_853703.mp3", 0.195, HIT + 1.6, -9, "glass", -0.4, 0.15, 1.98, tail=0.4)
tink = cut(load(GLASS + "g_735353.mp3", stereo=True), 41.0, 46.0, 0.05, 0.6)
tink = filt(tink, "highpass", 1200, 2) * np.interp(np.arange(len(tink)), [0, round(0.2 * SR), round(2.0 * SR), len(tink)],
                                                     [0.2, 1, 0.35, 0])[:, None]
place(tink, sec(HIT + 2), -10, "glass")
ICICLES = "677716__augustsandberg__breaking-icicles-2.wav"
for onset, frame, g, p in ((1.32, HIT + 3.2, -14, -0.45), (12.78, HIT + 5.5, -16, 0.5), (3.09, HIT + 8.5, -19, -0.2)):
    hit(ICICLES, onset, frame, g, "glass", p, onset - 0.02, onset + 0.8, tail=0.3)

# The plates come down (sync_probe): four into the water, then the rest onto the ice.
hit(ICE + "ice_560886.mp3", 1.06, 63.8, -10, "foley", -0.4, 0.95, 2.4, tail=0.4)
hit(ICE + "ice_560886.mp3", 3.05, 66.5, -13, "foley", 0.35, 2.95, 4.2, tail=0.4)
hit(ICE + "ice_347389.mp3", 7.10, 67.3, -15, "foley", 0.25, 7.0, 7.9, tail=0.3)
hit(ICE + "ice_347389.mp3", 11.27, 68.1, -17, "foley", -0.2, 11.17, 12.0, tail=0.3)
THUD = "675856__craigsmith__s10-16-dropping-large-piece-of-ice-onto-cement-dull.wav"
hit(THUD, 0.40, 72.5, -11, "foley", -0.3, 0.3, 1.6, tail=0.3)
hit(THUD, 0.40, 77.5, -14, "foley", 0.4, 0.3, 1.6, tail=0.3)

# ---------------------------------------------------------------- 5. the reveal (f80.5)
REVEAL = 80.5
hit("697758__sounddesignforyou__metlimpt_metal-resonant-scrape-impact_sdfy_metal-resonant.wav", 0.86, REVEAL, -10,
    "foley", 0.1, 0.6, tail=0.4, echo=-6)
hit("647712__unfa__braam.flac", 0.03, REVEAL, -16, "music")
for name, a, g in (("50620__vibe_crc__metal_hissing.wav", 0.0, -12), ("436470__rabban625__shovelsizzle8_mod.mp3", 0.3, -14)):
    x = cut(load(name), a, a + 2.5, 0.03, 0.3)
    x = filt(x, "highpass", 400) * np.exp(-np.arange(len(x)) / (1.4 * SR))
    place(x, sec(REVEAL), g, "heat", -0.15)
hit("234782__wubitog__steamhiss.wav", 0.53, 83, -12, "heat", 0.2, tail=0.2)
FIRE = "870180__bassimat__water-poured-onto-a-burning-fireplace-hissing-steam-and-sizzling-embers.wav"
x = filt(cut(load(FIRE), 6.0, 6.0 + sec(139 - 86), 0.2, 0.3), "highpass", 500) * np.linspace(1, 0.35, round(sec(139 - 86) * SR))
place(x, sec(86), -25, "heat", 0.1)
for f, (name, a), p in zip((85.1, 93.3, 100.5, 107.7),
                           (("436470__rabban625__shovelsizzle8_mod.mp3", 1.0), (FIRE, 2.85),
                            ("436470__rabban625__shovelsizzle8_mod.mp3", 0.38), (FIRE, 4.65)),
                           (-0.3, 0.25, 0.1, -0.15)):
    x = cut(load(name), a, a + 0.3, 0.005, 0.05) * np.exp(-np.arange(round(0.3 * SR)) / (0.08 * SR))
    place(filt(x, "highpass", 900), sec(f), -15, "heat", p)

# ---------------------------------------------------------------- the mix
# The far side of the valley answers the two big hits: late, dark, faint. No room reverb.
e = BUS["echo"]
valley = np.zeros_like(e)
for delay, g in ((0.31, -12), (0.63, -19)):
    d = round(delay * SR)
    valley[d:] += e[:-d] * db(g)
valley = filt(filt(valley, "lowpass", 2500, 2), "highpass", 200, 2)
foley = shelf(shelf(BUS["foley"] + BUS["ice"], "highpass", 3500, -4), "lowpass", 250, 2)
glass = shelf(BUS["glass"], "highpass", 6000, 3)
heat = filt(BUS["heat"], "lowpass", 9000, 2)
mix = BUS["amb"] + foley + glass * db(5) + heat * db(4) + BUS["music"] + valley

# The dissolve (f139-147): the sound gives way to the first play.
mix *= np.interp(np.arange(N), [0, at(139), N - 1], [1, 1, 0.25])[:, None]
mix[-round(0.01 * SR):] *= np.linspace(1, 0, round(0.01 * SR))[:, None]
mix *= 0.89 / np.max(np.abs(mix))

subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                "-af", "volume=6dB,alimiter=limit=0.79:attack=1:release=60:level=disabled",
                "-c:a", "pcm_s24le", sys.argv[1]], input=mix.astype(np.float32).tobytes(), check=True)
print(f"{sys.argv[1]}: {N / SR:.3f} s")
