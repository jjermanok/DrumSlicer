"""
apple_loops.py — save loops as Apple Loops, so Logic/GarageBand show them in the Loop Browser
and play them at the project tempo.

An Apple Loop is a CAF file with two extra 'uuid' chunks (layout read from Apple's own loops
in /Library/Audio/Apple Loops):
  * metadata  29819273-b5bf-4aef-b78d-62d1ef90bb2c: uint32 count, then key\\0value\\0 pairs,
    e.g. "beat count" "8", "time signature" "4/4", "category" "Drums". Logic derives the
    loop's tempo from beat count / duration.
  * transients 0352811b-9d5d-42e1-882d-6af61a6b330c: 20-byte header (…, sensitivity 50,
    resolution 16 = 16th notes, count) then per hit: uint32 0x00010000 + int64 sample
    position. Logic stretches at these points when the tempo changes.

Usage as a script: convert every loop already in a sample library
  python apple_loops.py drum_samples
"""
from __future__ import annotations

import io
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

META_UUID = bytes.fromhex("29819273b5bf4aefb78d62d1ef90bb2c")
TRANSIENT_UUID = bytes.fromhex("0352811b9d5d42e1882d6af61a6b330c")
USER_LOOPS = Path("~/Library/Audio/Apple Loops/User Loops/SingleFiles/Drum Sampler").expanduser()


def _chunk(kind: bytes, body: bytes) -> bytes:
    return kind + struct.pack(">q", len(body)) + body


def _metadata(beats: int, artist: str, genre: str | None) -> bytes:
    pairs = [("descriptors", "Part,Grooving,Processed"), ("time signature", "4/4"),
             ("subcategory", "Drum Kit"), ("beat count", str(beats)), ("category", "Drums")]
    if genre:
        pairs.append(("genre", genre))
    body = struct.pack(">I", len(pairs)) + b"".join(k.encode() + b"\0" + v.encode() + b"\0" for k, v in pairs)
    return _chunk(b"uuid", META_UUID + body)


def _transients(positions: list[int]) -> bytes:
    head = struct.pack(">IHHHHII", 0, 1, 0, 50, 16, 0, len(positions))
    recs = b"".join(struct.pack(">Iq", 0x00010000, int(p)) for p in positions)
    return _chunk(b"uuid", TRANSIENT_UUID + head + recs)


def _info(title: str, artist: str) -> bytes:
    pairs = [("title", title), ("artist", artist), ("comments", "Creator: drum_sampler")]
    body = struct.pack(">I", len(pairs)) + b"".join(k.encode() + b"\0" + v.encode() + b"\0" for k, v in pairs)
    return _chunk(b"info", body)


def _caf_chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    chunks, pos = [], 8
    while pos + 12 <= len(raw):
        kind = raw[pos:pos + 4]
        size = struct.unpack(">q", raw[pos + 4:pos + 12])[0]
        if size < 0:  # "until end of file" (only allowed for data)
            size = len(raw) - pos - 12
        chunks.append((kind, raw[pos + 12:pos + 12 + size]))
        pos += 12 + size
    return chunks


def _apple_lossless(pcm_caf: Path, work: Path) -> bytes | None:
    """Encode with macOS's own ALAC encoder and check, by decoding with macOS too (as Logic
    will), that every sample survives. libsndfile's ALAC codec is not used: it corrupted some
    loops (clicks) and also mis-decodes Apple's files. None if anything differs."""
    alac, back = work / "alac.caf", work / "back.caf"
    try:
        subprocess.run(["afconvert", "-f", "caff", "-d", "alac", str(pcm_caf), str(alac)],
                       check=True, capture_output=True)
        subprocess.run(["afconvert", "-f", "caff", "-d", "BEI24", str(alac), str(back)],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    a = dict(_caf_chunks(pcm_caf.read_bytes())).get(b"data", b"")[4:]
    b = dict(_caf_chunks(back.read_bytes())).get(b"data", b"")[4:]
    return alac.read_bytes() if a == b else None


def write_apple_loop(path: Path, audio: np.ndarray, sr: int, beats: int, transients: list[int],
                     title: str, artist: str, genre: str | None = None) -> None:
    """audio: (channels, samples) or (samples,). Writes a CAF Apple Loop, Apple Lossless when it
    verifies bit-exact, else plain 24-bit PCM."""
    with tempfile.TemporaryDirectory(prefix="apple_loop_") as tmp:
        work = Path(tmp)
        pcm = work / "pcm.caf"
        sf.write(pcm, audio.T if audio.ndim == 2 else audio, sr, format="CAF", subtype="PCM_24")
        raw = _apple_lossless(pcm, work) or pcm.read_bytes()
    out = [raw[:8]]
    for kind, body in _caf_chunks(raw):
        if kind == b"data":
            out += [_info(title, artist), _metadata(beats, artist, genre), _transients(transients)]
        if kind not in (b"info", b"free"):
            out.append(_chunk(kind, body))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".part")
    tmp_path.write_bytes(b"".join(out))
    tmp_path.replace(path)


def transients_of(mono: np.ndarray, sr: int) -> list[int]:
    import librosa

    on = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=256, backtrack=True, units="samples")
    return sorted({0, *[int(x) for x in on if 0 <= x < len(mono)]})


LOOP_NAME = re.compile(r"^(?P<song>.+)_(?P<bpm>[\d.]+)bpm_(?P<bars>\d+)bar_(?P<n>\d+)\.(wav|flac)$")


def convert_library(library: Path, dest: Path = USER_LOOPS) -> int:
    """Every <library>/<Artist>/loops/*.(flac|wav) -> <dest>/<Artist>/<name>.caf (skips existing)."""
    import librosa

    count = 0
    sources = sorted([p for pat in ("*/loops/*.flac", "*/loops/*.wav", "*/drum_parts/*/loops/*.flac",
                                    "*/drum_parts/*/loops/*.wav") for p in library.glob(pat)])
    for wav in sources:
        m = LOOP_NAME.match(wav.name)
        if not m:
            continue
        if wav.parent.parent.parent.name == "drum_parts":  # <Artist>/drum_parts/<part>/loops
            artist, folder = wav.parents[3].name, f"{wav.parents[3].name}/{wav.parent.parent.name}"
        else:
            artist = folder = wav.parent.parent.name
        out = dest / folder / (wav.stem + ".caf")
        if out.exists():
            continue
        audio, sr = sf.read(wav, always_2d=True)
        audio = audio.T
        write_apple_loop(out, audio, sr, beats=4 * int(m["bars"]),
                         transients=transients_of(librosa.to_mono(audio), sr),
                         title=wav.stem, artist=artist)
        count += 1
    return count


if __name__ == "__main__":
    lib = Path(sys.argv[1] if len(sys.argv) > 1 else "drum_samples")
    n = convert_library(lib)
    print(f"{n} Apple Loops written to {USER_LOOPS}")
    print("Logic indexes new loops on launch; if they don't appear, right-click in the Loop Browser "
          "and choose 'Re-index All Loops'.")
