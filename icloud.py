"""
icloud.py — keep the big, rarely opened full-length stems in iCloud Drive.

No-drums tracks and acapellas (~80 MB per song) go to iCloud Drive/DrumSlicer Stems/<Artist>/
{no_drums,acapellas}/; the samples you audition in Logic/Ableton stay on the Mac. Once a stem
has uploaded it's evicted (`brctl evict`): it still shows in Finder and downloads when opened,
but takes no space on the Mac. Without iCloud Drive, stems stay in a local "stems" folder.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ICLOUD = Path("~/Library/Mobile Documents/com~apple~CloudDocs").expanduser()


def stems_root(library: Path) -> Path:
    """Where full-length stems live: iCloud Drive if set up, else next to the library."""
    return ICLOUD / "DrumSlicer Stems" if ICLOUD.is_dir() else library.parent / "stems"


def no_drums_path(root: Path, artist: str, song: str) -> Path:
    return root / artist / "no_drums" / f"{song}.flac"


def acapella_path(root: Path, artist: str, song: str) -> Path:
    return root / artist / "acapellas" / f"{song}.flac"


def on_disk_bytes(path: Path) -> int:
    """Space the file actually takes on this Mac (0 once evicted to iCloud)."""
    try:
        return path.stat().st_blocks * 512
    except OSError:
        return 0


def evict_uploaded(root: Path) -> tuple[int, int]:
    """Evict every stem under `root` that is still taking space locally. Files that haven't
    finished uploading stay for now (a later run, or macOS itself, evicts them).
    Returns (files evicted, bytes freed)."""
    if not str(root).startswith(str(ICLOUD)) or not root.exists():
        return 0, 0
    n = freed = 0
    for f in root.rglob("*.flac"):
        before = on_disk_bytes(f)
        if before == 0:
            continue
        subprocess.run(["brctl", "evict", str(f)], capture_output=True)
        after = on_disk_bytes(f)
        if after < before:
            n += 1
            freed += before - after
    return n, freed
