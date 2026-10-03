# Drum Sampler

Turns songs into a drum sample pack. Each song's drums are separated (on your Mac with Demucs by default, or with lalal.ai), then cut at their transients into one-shots and loops.

## Separator
| `--separator` | Cost | Notes |
|---|---|---|
| `demucs` (default) | free | Demucs htdemucs_ft on your Mac, ~90 s for a 5.5-min song on an M1 Pro. On "SubLab - Maera" it measured as clean as lalal.ai (5.5% melody/vocal bleed vs 6.3%). No API key needed. |
| `lalal` | ~1 lalal.ai minute per song minute | Needs `LALAL_API_KEY`. Songs whose drums are one loop are cut to a 60 s excerpt first to save minutes (see below). |

Stems already made by either separator are reused, so switching doesn't redo old songs.

## Setup
Needs Python 3.10+ (Beat This). On Apple Silicon, use a native build so the GPU is used:
```
pip3 install uv
uv venv --managed-python --python cpython-3.11-macos-aarch64-none .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
export LALAL_API_KEY=your_activation_code   # only for --separator lalal: lalal.ai → Profile → activation code
```
Check the key works (prints minutes left):
```
curl -s -X POST https://www.lalal.ai/api/v1/limits/minutes_left/ -H "X-License-Key: $LALAL_API_KEY"
```
Each song uses its length in processing minutes (a 4-minute song ≈ 4 minutes).

## Run
```
./.venv/bin/python drum_sampler.py song1.mp3 song2.mp3 -o my_pack
./.venv/bin/python drum_sampler.py ~/Music/to_sample/ -o my_pack      # every song in a folder
```

## Songs you don't have yet (Soulseek)
Give a song as `"Artist - Title"` and it's fetched as an mp3 first:
```
export SLSK_USERNAME=your_soulseek_name SLSK_PASSWORD=your_soulseek_password
./.venv/bin/python drum_sampler.py "Bonobo - Kerala" "Four Tet - Baby" -o my_pack
./.venv/bin/python drum_sampler.py --songs-file wanted.txt -o my_pack     # one "Artist - Title" per line
```
1. It looks in `~/Soulseek Downloads` for an mp3 (or flac) you already have (title in the file name, artist in the path or tags).
2. Otherwise it searches Soulseek for "Artist Title". If nothing fits, it searches the artist and filters by title.
3. mp3s are preferred; if neither search turns up an mp3 it takes a flac instead. Other versions (remix, live, edit, instrumental…) are skipped unless they're in the title you asked for.
4. The best source is picked by free slot, bitrate (320 preferred), queue and speed. If a download stalls for 60 s it tries the next source (up to 5).

Downloads go to `~/Soulseek Downloads/complete/drum_sampler/` (`--download-dir` to change).
This uses its own Soulseek connection, not the SoulseekQt app. **Soulseek allows one login per account, so quit SoulseekQt while it runs** (or use a second account). It doesn't share any of your files.

## Output
One folder per artist (from the song's tags, or "Artist - Title" in the file name). Leading track numbers are dropped.
```
my_pack/
  Bonobo/
    oneshots/
      kicks/          shokoufeh_001.wav, uncasually_001.wav, …
      snares_claps/   (claps and snares sound too alike to split reliably)
      hats/
      cymbals/
      percussion/
    loops/            shokoufeh_118.3bpm_1bar_001.wav, shokoufeh_118.3bpm_2bar_001.wav, …
    drum_stems/       shokoufeh.wav  (cached: re-running a song won't use more lalal.ai minutes)
    no_drum_stems/    shokoufeh.wav  (the song with the drums removed)
```
All audio is 24-bit **FLAC** (lossless: identical to WAV, about half the size; Logic and Ableton Live 11+ read it) at the song's sample rate, stereo kept, samples peak-normalised to −1 dBFS. Apple Loops are CAF files with Apple Lossless audio. `python to_flac.py drum_samples vocal_samples` converts an older WAV library, checking every file is sample-identical before deleting the WAV.

## Drum parts (DrumSep)
Each drum stem is also split into kick, snare, toms, hi-hat, ride and crash with MDX23C DrumSep (by jarredou & aufr33, ~1 min per song on the Mac's GPU; the 432 MB model lives in `~/.cache/drum_sampler/models`). Parts that are silent in a song aren't kept. Each part gets its own stem, one-shots and loops, alongside the full-drum ones:
```
drum_samples/<Artist>/drum_parts/
  kick/    <song>.flac   oneshots/<song>_001.flac   loops/<song>_126bpm_1bar_001.flac
  snare/ …  hihat/ …  toms/ …  ride/ …  crash/ …
```
Part loops also go to Apple Loops under `Drum Sampler/<Artist>/<part>/`. Options: `--no-drum-parts`, `--max-part-oneshots 16`.

Once a song's one-shots, loops and drum parts are all cut, its full drum stem and its drum-part stems are **deleted** to save space (~75 MB per song). Each artist folder keeps a hidden `.drum_sampler/` folder with small records per song (bar grid, kept parts, a "sliced" marker), so a song is never separated again. `--keep-drum-stem` / `--keep-part-stems` keep them. `--split-existing` runs only DrumSep + part slicing (and the stem clean-up) on songs already in the library. Tested against LarsNet on live, electronic and pop drums; DrumSep separated snares and hats far better.

## Storage
Samples (one-shots, loops, drum-part samples, vocal phrases, Apple Loops) stay on the Mac so Logic and Ableton load them instantly: ~55 MB per song.
The full-length stems you rarely open — the song without drums and the acapella, ~80 MB per song — go to **iCloud Drive/DrumSlicer Stems/<Artist>/{no_drums,acapellas}/**. Once uploaded they're evicted from the Mac (`brctl evict`): still visible in Finder, downloaded when opened, no local space used. Without iCloud Drive they stay in a local `stems/` folder. Drum and drum-part stems are deleted after slicing.
Source mp3s are preferred over FLAC when downloading (a third of the size, no audible difference after separation). Don't use MP3 for samples: encoders pad the start with silence, which shifts attacks and breaks sample-exact loops.

## Vocals
With Demucs (the default), each song's vocals are separated too and kept in their own library next to the drums:
```
vocal_samples/
  ROSALÍA/
    phrases/     reliquia_phrase_001.flac  cut at the pauses, 0.5-12 s, repeats removed
iCloud Drive/DrumSlicer Stems/ROSALÍA/acapellas/reliquia.flac   the full vocal track
```
Songs without singing are skipped: a song counts as vocal if the vocal track is within 12 dB of the full song for at least 15 s (instrumentals measure 0-1 s, Reliquia 144 s). Skipped songs are listed in `vocal_samples/.no_vocals.txt`. Songs already separated for drums only get the vocals step (~20 s).
Always saved to `vocal_samples/` next to the drum library. Options: `--no-vocals`, `--max-vocal-phrases 24`, `--vocal-similarity 0.85` (lower = fewer, more varied phrases).

## Apple Loops (Logic / GarageBand)
Every loop is also saved as an Apple Loop in `~/Library/Audio/Apple Loops/User Loops/SingleFiles/Drum Sampler/<Artist>/`, so it shows up in Logic's Loop Browser (category Drums) and follows the project tempo. The file carries the beat count (4 per bar, 4/4) and the drum hits' positions for clean time-stretching, in the same layout as Apple's own loops. Turn off with `--no-apple-loops`. To convert loops made earlier: `./.venv/bin/python apple_loops.py drum_samples`. If new loops don't appear, right-click in the Loop Browser → Re-index All Loops.

## How it slices
- **One-shots:** transient detection with backtracking to the true start of each hit. Each hit runs to the next transient (1.5 s max) and is trimmed where it decays. Near-identical hits are removed so you don't get 200 copies of the same kick. The loudest version is the one kept, and the `--max-oneshots` slots are shared out across kick/snare/hat/cymbal/perc so quiet hats aren't crowded out by loud kicks.
- **Labels:** a spectral guess (kick / snare / hat / cymbal / perc) based only on the energy each hit *adds*, so the tail of the previous kick doesn't make a hat look like a kick. On test mixes built from labelled samples it's right about 80% of the time; a kick + hat played together is labelled kick. "Perc" is the catch-all (toms, rims, shakers, odd sounds).
- **Loops:** bars are found with Beat This (a neural beat/downbeat tracker) on the **full song**, not the drum stem: downbeats are mostly heard in the chords, bass and vocals. On a test of 33 drum loops with music underneath it got the downbeat right 33/33, vs 10/33 for the old guess from the drums alone. The grid is saved as `drum_stems/<song>.beats.json`. Loops (1 and 2 bars) run from one downbeat to the next, snapped to the drum hit's start. On songs locked to a fixed tempo they're sample-exact at the song's BPM; with live drummers each loop keeps its own bar length and the file name gives that loop's BPM. Quiet/gappy sections and duplicate bars are skipped. Tempo: measured over many short stretches (one tracker slip can't bend it), halved above 180 BPM (the tracker counting half-beats), then the whole number whose steady grid fits best — a .5 (e.g. 87.5) only when it fits clearly better; low-fit tempos are flagged with a `confidence` in the song's `.beats.json`. The grid is re-anchored where a song's beat shifts (e.g. after a breakdown). Songs whose tempo really drifts keep the tracked beats.

## Useful knobs
| Flag | Default | Effect |
|---|---|---|
| `--sensitivity` | 0.07 | Lower finds more (quieter) hits |
| `--max-oneshots` | 48 | Per song |
| `--oneshot-similarity` | 0.90 | Lower means stricter de-duplication (fewer, more varied hits) |
| `--loop-bars` | 1 2 | For example `--loop-bars 2 4` |
| `--max-loops` | 12 | Per song, per loop length |
| `--loop-similarity` | 0.85 | Lower means fewer, more varied loops |
| `--extraction` | clear_cut | `deep_extraction` keeps more detail but lets more of the other instruments bleed in |
| `--splitter` | auto | Force a lalal.ai model |
| `--already-drums` | – | Inputs are already drum stems, so lalal.ai is skipped |
| `--no-normalize` | – | Keep the original levels |
