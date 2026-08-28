# patina

Analog coloration for digital music: tube harmonics, vinyl and cassette
artefacts, applied to whole albums, with tags and artwork preserved.

*Patina* is what time leaves on a surface. That is what this does to a
recording — nothing is restored, something is added.

It is a self-contained Python script with a command line interface, plus an
optional Tk front end. Point it at a folder of FLAC files, pick a preset, and it
writes a coloured copy alongside the original — same file names, same bit depth,
same metadata.

```bash
python3 patina.py "~/Music/Pablo Honey" -o "~/Music/Pablo Honey [tube]" --preset tube
```

---

## Why a waveshaper and not a filter

A filter cannot do this. FIR, IIR, biquad — anything linear and time-invariant
takes a 1 kHz sine in and gives a 1 kHz sine back, with a different amplitude and
phase. It can never *create* a frequency that was not already there.

Harmonics need a non-linearity: a function `y = f(x)` applied sample by sample.
Which harmonics you get is decided entirely by the symmetry of that function:

- **Odd symmetry**, `f(-x) = -f(x)`, such as `tanh(kx)` → **odd** harmonics
  (3rd, 5th…). This is push-pull and solid state in clipping: the hard sound.
- **Asymmetric** → **even** harmonics. This is the single-ended triode: the 2nd
  harmonic dominates and grows with level.

This tool uses a biased hyperbolic tangent:

```
y = tanh(k·(x + b)) − tanh(k·b)
```

`b` is the bias that breaks the odd symmetry and generates the even harmonics;
the subtracted constant removes the DC offset that the bias would otherwise
introduce. `k` is the drive. At `b = 0` it collapses back to a plain `tanh` and
you get odd harmonics only — which is exactly how `--bias` lets you dial the
character from "valve" to "transistor".

## Aliasing, and why oversampling is not optional

A non-linearity generates harmonics without limit. Everything above Nyquist
folds back into the audio band at **non-harmonic** frequencies — inharmonic,
dissonant, and unrelated to the music. At 44.1 kHz a 6 kHz fundamental puts its
4th harmonic at 24 kHz, which folds down to 20.1 kHz. That is the metallic,
"digital" edge that makes cheap saturation plugins sound bad.

So the waveshaper runs in an oversampled domain: upsample 8×, distort, low-pass,
downsample. The effect is measurable — at 6 kHz with heavy drive:

| oversampling | THD | non-harmonic residue |
|---|---|---|
| 1× (off) | 7.26 % | 0.227 % |
| 2× | 7.26 % | 0.025 % |
| 8× | 7.26 % | 0.026 % |

Same harmonics, 19 dB less rubbish.

---

## Install

```bash
pip install numpy scipy soundfile mutagen
```

`soundfile` bundles libsndfile, which reads and writes FLAC natively — no ffmpeg
needed. `mutagen` handles the tags. Python 3.9 or newer.

## Use

```bash
# One album
python3 patina.py "~/Music/Album" -o "~/Music/Album [tube]" --preset tube

# A whole library, recursively
python3 patina.py ~/Music -o ~/Music_vinyl --preset vinyl --recursive

# See what it would do, without doing it
python3 patina.py ~/Music --dry-run --recursive

# Measure the harmonic spectrum of a setting, writing no files
python3 patina.py --analyze --preset tube
```

The source is never touched: output always goes to a separate directory, and the
tool refuses to run if the destination is the same as the source. Files that
already exist are skipped unless you pass `--force`.

### Graphical interface

```bash
python3 patina_gui.py
```

A Tk window with the same options: source and destination, preset, every
parameter in editable fields, and Process / Dry run / Analyze / Stop. It is a
front end and nothing more — it builds a command line, runs `patina.py` as a
subprocess and streams its output into the log, printing the exact command at
the top so it can be copied into a terminal. Tk ships with Python, so there is
nothing else to install.

### What it preserves

- **Bit depth**: 16→16, 24→24, with TPDF dither applied only on 16-bit output.
- **Tags**: every Vorbis comment, copied verbatim.
- **Artwork**: embedded pictures copied byte for byte.
- **Companion files**: loose covers, booklets, cue sheets, rip logs. The cue
  sheet stays valid because file names do not change.
- **ReplayGain**: `*_PEAK` fields are recomputed from the processed audio. The
  gain value stays correct because the chain matches the RMS of the original.
- **Directory structure**: multi-disc folders are recreated as they were.

---

## Presets

| Preset | What it is |
|---|---|
| `tube-soft` | Barely there: ~0.6 % THD, second harmonic |
| `tube` | Single-ended triode at a normal listening level, ~2 % THD |
| `tube-hot` | Driven hard, ~5 % THD, audible compression on peaks |
| `vinyl-mint` | A new record on a good deck: bass mono, mild wow, no noise |
| `vinyl` | Ordinary pressing, ordinary turntable |
| `vinyl-worn` | Played a lot: hiss, rumble, 4 clicks per second |
| `vinyl-trashed` | Charity shop copy: heavy crackle, a pop every revolution |
| `vinyl-tube` | Turntable into a valve preamp: the full vinyl chain, then a valve stage |
| `cassette-chrome` | Fresh chrome/metal tape, Dolby B tracking almost perfect |
| `cassette` | Ordinary ferric compact cassette, Dolby B |
| `cassette-worn` | Old, oft-played tape: wow, hiss, mistracked Dolby, print-through |
| `cassette-no-dolby` | Same tape played back with the decoder switched off: bright and hissy |
| `cassette-tube` | Cassette deck into a valve preamp: the full tape chain, then a valve stage |

Any preset is a starting point — individual options override it:

```bash
python3 patina.py ~/Music/Album --preset vinyl-worn --click-rate 8 --noise-db -58
```

### The valve stage after vinyl

`--mode vinyl-tube` runs the complete vinyl chain and then a second
waveshaper, with its own `--tube-drive` and `--tube-bias`. It sits at the end
deliberately: what reaches a preamp is the cartridge output, hiss and clicks
included, so the valve stage colours those too rather than being applied to
clean music that then gets noise glued on top.

(`--mode both` / `--preset both` still work — they are the original name for
this mode, from before cassette support existed, kept as a deprecated alias
so old command lines do not break.)

It has its own oversampled round, because the surface artefacts are generated at
base rate after the first one, so `vinyl-tube` costs roughly a vinyl pass plus a
tube pass. `--tube-drive 0` disables it and gives you plain vinyl back, bit for
bit.

The two non-linearities add up. Measured at −6 dBFS, 1 kHz:

| | THD | 2nd harmonic |
|---|---|---|
| `--preset vinyl` | 1.79 % | 1.73 % |
| `--preset vinyl-tube` | 3.50 % | 3.38 % |

So set `--drive` for the record and `--tube-drive` for the amplifier, and expect
the sum:

```bash
python3 patina.py ~/Music/Album --preset vinyl-trashed --mode vinyl-tube --tube-drive 0.2
```

### The valve stage after cassette

Same idea, `--mode cassette-tube`: the complete tape chain — saturation, wow &
flutter, EQ, hiss, Dolby, print-through — followed by a second waveshaper. It
sits downstream for the same reason as `vinyl-tube`: the tape hiss and
print-through should pick up the valve's colour too, not just the clean signal
underneath them.

| | THD | 2nd harmonic |
|---|---|---|
| `--preset cassette` | 0.67 % | 0.59 % |
| `--preset cassette-tube` | 2.26 % | 2.13 % |

```bash
python3 patina.py ~/Music/Album --preset cassette-worn --mode cassette-tube --tube-drive 0.2
```

## The vinyl model

Vinyl is not mainly a distortion effect. Distortion is perhaps the fourth thing
you notice, so the model covers the rest:

| Artefact | How it is modelled | Option |
|---|---|---|
| Bass summed to mono | Mid/side split, side high-passed with a linear-phase filter | `--bass-mono-hz` |
| Channel crosstalk | 25–30 dB, against >90 dB for digital | `--crosstalk-db` |
| Wow & flutter | Resampling at modulated time positions: 0.55 Hz for the platter revolution, 9.3 Hz for flutter | `--wow-pct`, `--flutter-pct` |
| Tracking distortion | Rises with frequency: pre-emphasis → waveshaper → de-emphasis | `--tilt-db` |
| HF roll-off | Gentle shelf at 10 kHz | `--hf-rolloff-db` |
| Rumble | Low-passed noise below 45 Hz | `--rumble-db` |
| Surface hiss | High-passed noise above 2 kHz | `--noise-db` |
| Clicks and pops | See below | `--click-rate`, `--click-db` |
| A defect that repeats | One pop per platter revolution | `--tick-db`, `--rpm` |

### Clicks

Click events follow a **Poisson process**: the interval from one click to the
next is random and independent, as it is for defects scattered over a surface.
A regular pattern would be recognised as artificial within seconds.

Each click is a burst of band-limited noise under an exponentially decaying
envelope, with a sharpened leading edge. There are two families:

| | duration | spectral peak | share |
|---|---|---|---|
| fine crackle | 3 ms | 6.4 kHz | 95 % of events |
| pop (scratch) | 24 ms | 2.3 kHz | 5 % |

The same defect appears on both channels at slightly different amplitudes and
with up to 24 samples of delay, so it does not sound like a mono event glued to
the centre of the image.

The level is **absolute**, not relative to the music. Clicks stay masked in loud
passages and emerge in quiet ones — which is what a record does, and which comes
for free from the fact that they are additive.

## The cassette model

Same idea as vinyl — most of what you hear from a cassette is not distortion —
but a different transport and a different set of failure modes:

| Artefact | How it is modelled | Option |
|---|---|---|
| Channel crosstalk | Head-gap bleed | `--crosstalk-db` |
| Wow & flutter | Same modulated resampling as vinyl, tuned to a capstan drive instead of a platter | `--wow-pct`, `--flutter-pct` |
| Tape saturation | Level-dependent, not frequency-tilted like vinyl tracking distortion | `--drive`, `--bias` |
| Bass bump | Low shelf around the head-EQ corner | `--lf-bump-hz`, `--lf-bump-db` |
| HF roll-off | Shelf at the tape/head bandwidth limit | `--hf-rolloff-db`, `--hf-corner-hz` |
| Head azimuth loss | Extra top-end loss, worse on one channel — decks are rarely perfectly aligned | `--azimuth-db`, `--azimuth-corner-hz` |
| Tape hiss | Broadband, top-heavy noise | `--hiss-db` |
| Dolby tracking | See below | `--dolby-type`, `--dolby-mismatch-pct` |
| Print-through | See below | `--print-db`, `--print-ms` |

### Dolby B/C

Real Dolby noise reduction is a sliding-band compander: quiet high-frequency
content is boosted going onto the tape, and pulled back down by the same
amount on playback — taking the hiss recorded in that pass down with it
whenever the music is loud. A decoder that tracks the encoder perfectly is
inaudible. A decoder that does not — wrong type, or the level slightly off —
leaves a residual gain error that is heard as the noise floor **breathing** in
time with the music, the single most recognisable Dolby artefact.

`--dolby-type` picks `off` (plain tape hiss, no tracking), `b` or `c` (`c`
compresses a wider band, so a mismatch is more audible). `--dolby-mismatch-pct`
sets how far the simulated decoder misses the ideal: 0 is perfect and
inaudible, 100 is no reduction at all. `cassette-no-dolby` demonstrates the
other classic case — a tape encoded with Dolby but played back with the
decoder off, the bright, hissy sound of a boombox mixtape without noise
reduction.

### Print-through

A faint, muffled copy of a loud passage, transferred magnetically onto the
tape layer wound next to it, heard *before* the passage itself rather than
after. Only the pre-echo is modelled — forward masking is much weaker than
backward masking, so on a real tape that is the audible one.
`--print-ms` sets how far ahead of the sound it is heard, `--print-db` its
level.

---

## Verifying what it does

`--analyze` runs a sine through the chain and prints the harmonic spectrum. The
test frequency is locked to the FFT grid so there is no leakage and no window is
needed, and it is deliberately *not* a whole number of samples per period —
otherwise aliasing products would land exactly on the harmonics and hide.

```
$ python3 patina.py --analyze --preset tube
Harmonic analysis — mode=tube drive=0.35 bias=0.25 oversample=8
Sine at 999.9 Hz, -6.0 dBFS, fs=44100 Hz

  harmonic    freq (Hz)    dB rel        %
  ----------------------------------------
  1 (fund.)        1000       0.0   100.00
  2                2000     -34.5    1.893
  3                3000     -44.3    0.609
  4                4000     -72.5    0.024
  5                5000     -87.3    0.004
  ----------------------------------------
  THD = 1.989 %   |   non-harmonic residue (aliasing/noise) = 0.0015 %
  even/odd ratio = 3.11 (EVEN dominant (tube-like))
```

That is the signature of a single-ended triode: around 2 % THD dominated by the
second harmonic, falling away quickly, and growing with signal level.

The `--drive` mapping is calibrated against this measurement:

| drive | THD at −6 dBFS |
|---|---|
| 0.2 | 0.6 % |
| 0.35 | 2 % |
| 0.6 | 5 % |
| 1.0 | 11 % |

To inspect the surface artefacts on their own, run a file of digital silence
through the vinyl chain — the gain compensation is applied before the additive
noise, so silence in gives you exactly the artefacts out.

---

## Performance

Measured on an Apple M-series laptop with 16 GB of RAM, a 42-minute album of
16-bit/44.1 kHz FLAC:

| mode | time |
|---|---|
| `tube` | 21 s |
| `vinyl` | 50 s |
| `vinyl-tube` | 63 s |

In the oversampled domain a four-minute track is 85 million samples per channel,
so memory, not CPU, is the limit. Two things follow from that:

- Wow and flutter is computed in blocks of 1 M samples, and the waveshaper and
  shelf filters operate in place. Without this the chain allocated around 4 GB
  per track and the machine spent its time swapping.
- `--jobs` defaults to a worker count derived from the physical RAM and the
  oversampling factor. Pass `--jobs N` to override it.

## Notes on material

Heavily limited modern masters are the worst case for a tube model. If a record
peaks at −0.14 dBFS and sits near full scale continuously, the drive is applied
almost uniformly instead of dynamically, which is the opposite of what a valve
does. Well-recorded material with real dynamics responds far better.

In vinyl and cassette mode, `--mix` below 100 should be used with wow and
flutter set to zero. The dry path is not pitch-modulated, so blending it with
the wet path produces a chorus effect.

---

## Why I built it

I have a passion for music, for audio signal processing, and for the search for
emotion in sound — for what makes you stop whatever you were doing and simply
listen. This tool is an attempt at that: it makes listening a little more
analogue, and carries you back to a time further away.

All of it was made possible in part by Anthropic's recent AI technology: the
code was written together with Claude.

— Carlo A.

---

## Licence

MIT — see [LICENSE](LICENSE).
