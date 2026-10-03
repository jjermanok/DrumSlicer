"""
to_flac.py — shrink an existing library without changing a single sample:
  * every .wav under the given folders -> .flac (same bit depth), verified sample-for-sample
    identical before the .wav is deleted
  * Apple Loops (.caf, plain PCM) -> Apple Lossless inside the same CAF, tempo metadata and
    transient markers kept

Usage: python to_flac.py drum_samples vocal_samples
       (Apple Loops in ~/Library/Audio/Apple Loops/User Loops/SingleFiles/Drum Sampler are
        always included)
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from apple_loops import USER_LOOPS

SUBTYPES = {"PCM_16": "PCM_16", "PCM_24": "PCM_24", "PCM_S8": "PCM_S8", "PCM_U8": "PCM_S8"}


def wav_to_flac(wav: Path) -> tuple[int, int, str]:
    """Returns (bytes before, bytes after, status)."""
    info = sf.info(wav)
    subtype = SUBTYPES.get(info.subtype)
    if subtype is None:  # float/32-bit WAVs aren't lossless in FLAC; leave them
        return wav.stat().st_size, wav.stat().st_size, f"kept (subtype {info.subtype})"
    flac = wav.with_suffix(".flac")
    tmp = wav.with_name(wav.stem + ".part.flac")
    a, sr = sf.read(wav, always_2d=True, dtype="int32")
    sf.write(tmp, a, sr, format="FLAC", subtype=subtype)
    b, _ = sf.read(tmp, always_2d=True, dtype="int32")
    if not np.array_equal(a, b):
        tmp.unlink()
        return wav.stat().st_size, wav.stat().st_size, "MISMATCH, kept wav"
    tmp.replace(flac)
    before = wav.stat().st_size
    wav.unlink()
    return before, flac.stat().st_size, "ok"


def _chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    out, pos = [], 8
    while pos + 12 <= len(raw):
        kind = raw[pos:pos + 4]
        size = struct.unpack(">q", raw[pos + 4:pos + 12])[0]
        if size < 0:
            size = len(raw) - pos - 12
        out.append((kind, raw[pos + 12:pos + 12 + size]))
        pos += 12 + size
    return out


def caf_to_alac(caf: Path) -> tuple[int, int, str]:
    """Re-encode a PCM Apple Loop to Apple Lossless with macOS's own encoder, verified with
    macOS's decoder (libsndfile's ALAC codec corrupts some files)."""
    import tempfile
    from apple_loops import _apple_lossless, _caf_chunks, _chunk

    raw = caf.read_bytes()
    chunks = _caf_chunks(raw)
    if dict(chunks).get(b"desc", b"")[8:12] != b"lpcm":
        return len(raw), len(raw), "already compressed"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pcm = work / "pcm.caf"
        pcm.write_bytes(raw)
        alac = _apple_lossless(pcm, work)
    if alac is None:
        return len(raw), len(raw), "not bit-exact, kept PCM"
    keep = [(k, v) for k, v in chunks if k in (b"info", b"uuid")]  # loop metadata + transients
    audio = [(k, v) for k, v in _caf_chunks(alac) if k not in (b"free", b"info")]
    body = [c for c in audio if c[0] != b"data"] + keep + [c for c in audio if c[0] == b"data"]
    out = raw[:8] + b"".join(_chunk(k, v) for k, v in body)
    tmp_path = caf.with_name(caf.name + ".part")
    tmp_path.write_bytes(out)
    tmp_path.replace(caf)
    return len(raw), len(out), "ok"


def main() -> None:
    roots = [Path(p) for p in (sys.argv[1:] or ["drum_samples", "vocal_samples"])]
    wavs = [p for r in roots for p in r.rglob("*.wav") if not p.name.endswith(".part.wav")]
    cafs = list(USER_LOOPS.rglob("*.caf")) if USER_LOOPS.exists() else []
    print(f"{len(wavs)} WAV files and {len(cafs)} Apple Loops to convert", flush=True)
    before = after = 0
    problems = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, ((b, a, status), path) in enumerate(zip(
                pool.map(wav_to_flac, wavs, chunksize=16), wavs), 1):
            before += b
            after += a
            if status != "ok":
                problems.append(f"{path}: {status}")
            if i % 2000 == 0:
                print(f"  {i}/{len(wavs)} WAV done, {(before - after) / 1e9:.1f} GB saved so far", flush=True)
        for (b, a, status), path in zip(pool.map(caf_to_alac, cafs, chunksize=16), cafs):
            before += b
            after += a
            if status not in ("ok", "already compressed"):
                problems.append(f"{path}: {status}")
    print(f"done: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB (saved {(before - after) / 1e9:.1f} GB)")
    for p in problems[:20]:
        print("  !", p)


if __name__ == "__main__":
    main()
