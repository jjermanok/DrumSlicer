"""
soulseek.py — find a song as an mp3 (or flac if there's no mp3): in your local downloads
first, else on the Soulseek network.

Search strategy for "Artist - Title":
  1. search "Artist Title", keep mp3s whose file name contains the title
  2. if nothing usable, search "Artist" and filter those results by title
  3. still no mp3: take a flac from either search
Candidates are ranked by free upload slot, bitrate, queue length and speed; if a
download stalls or fails, the next candidate is tried.

Uses its own Soulseek connection (aioslsk), not the SoulseekQt app. Only one login per
account is allowed, so quit SoulseekQt first or use a second account.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Words that mean "a different version of the song" unless they're in the requested title.
VERSION_WORDS = {"remix", "rmx", "edit", "live", "instrumental", "acapella", "acappella", "mix",
                 "version", "karaoke", "cover", "demo", "remaster", "remastered", "vip", "dub", "rework"}


def log(msg: str) -> None:
    print(msg, flush=True)


def words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’]", "", s)
    return re.findall(r"[a-z0-9]+", s)


# Suffixes that still mean the normal track, so "mix"/"version" in them isn't a different version.
SAME_VERSION = re.compile(r"(?<![a-z0-9])(original|extended|album|single)[\s_-]+(mix|version|edit)(?![a-z0-9])", re.I)


def parse_query(q: str) -> tuple[str, str]:
    """'Artist - Title' -> (artist, title). Without ' - ', the whole thing is the title."""
    if " - " in q:
        a, t = q.split(" - ", 1)
        return a.strip(), t.strip()
    return "", q.strip()


def basename(remote_path: str) -> str:
    return re.split(r"[\\/]", remote_path)[-1]


# Words in file names that don't make it a different song
FILLER = {"original", "extended", "explicit", "clean", "official", "audio", "lyric", "lyrics", "video",
          "hq", "hd", "remaster", "remastered", "mp3", "flac", "kbps", "320", "web", "single", "album"}


def _extra_words(name: str, artist: str, title: str) -> set[str]:
    """Words in the file name that are neither the title, the artist, track numbers, hashes,
    featured-artist credits nor filler: what makes it possibly a different song."""
    name = re.sub(r"[(\[]?\b(feat|ft|featuring)\b\.?[^)\]]*[)\]]?", " ", name, flags=re.I)
    known = set(words(title)) | set(words(artist)) | FILLER
    return {w for w in words(name) if w not in known and not w.isdigit() and not re.fullmatch(r"[0-9a-f]{8}", w)}


def matches(path: str, artist: str, title: str, need_artist: bool) -> bool:
    """Title words must all be in the file name; artist words anywhere in the path
    (Soulseek shares are usually Artist/Album/NN Title.mp3). The file name may add at most one
    other word ("08 You Know Who I Am" is not "You"), unless it also names the artist, in which
    case the extras are usually collaborators ("hessian_and_molly_and_afnan_prince-shine")."""
    # "Extended Mix" etc. count as the same song, so drop them from BOTH sides: the title asked
    # for ("Shapeshift (Extended Mix)") and the file ("Rezident - Shapeshift.mp3")
    stem = SAME_VERSION.sub(" ", Path(basename(path)).stem)
    title = SAME_VERSION.sub(" ", title)
    name_w = set(words(stem))
    title_w = words(title)
    if not title_w or not set(title_w) <= name_w:
        return False
    if (name_w & VERSION_WORDS) - set(title_w):
        return False
    if need_artist and artist and not set(words(artist)) <= set(words(path)):
        return False
    # "Bonobo - Migration - 08 - Kerala": with a track number, the title must come after it, or
    # an album name earlier in the file name passes for the song ("Migration"). The first
    # number counts, so titles that start with one still match ("03-christian_loffler-7_tepe").
    numbered = list(re.finditer(r"(?:^|[\s_.-])\d{1,2}[\s._-]+", stem))
    if numbered and not set(title_w) <= set(words(stem[numbered[0].end():])):  # after the first number
        return False
    extras = _extra_words(stem, artist, title)
    names_artist = bool(artist) and set(words(artist)) <= name_w
    if len(extras) > (6 if names_artist else 1):
        return False
    return True


# --------------------------------------------------------------------------- #
# Local library
# --------------------------------------------------------------------------- #
def _tag_artists(path: str) -> str:
    try:
        import mutagen
        tags = mutagen.File(path, easy=True) or {}
        return " ".join(str(v) for k in ("artist", "albumartist") for v in tags.get(k, []))
    except Exception:
        return ""


def find_local(query: str, roots: list[Path]) -> Path | None:
    for ext in (".mp3", ".flac"):
        found = _find_local(query, roots, ext)
        if found:
            return found
    return None


def _find_local(query: str, roots: list[Path], ext: str) -> Path | None:
    artist, title = parse_query(query)
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(ext):
                    continue
                full = os.path.join(dirpath, f)
                if not matches(full, artist, title, need_artist=False):
                    continue
                # Soulseek download folders are named after the uploader, so the artist is
                # often only in the tags, not the path.
                if not artist or matches(full, artist, title, need_artist=True) or \
                        set(words(artist)) <= set(words(_tag_artists(full))):
                    return Path(full)
    return None


# --------------------------------------------------------------------------- #
# Soulseek network
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    username: str
    path: str
    size: int
    bitrate: int
    free_slot: bool
    queue: int
    speed: int

    def score(self) -> tuple:
        # prefer: free slot, 320k (but not absurd VBR numbers), short queue, fast peer.
        # flac bitrates are all well above 320, so for flac this just ranks on slot/queue/speed.
        return (self.free_slot, min(self.bitrate, 320), -self.queue, self.speed)


def _candidates(requests: list, artist: str, title: str, ext: str) -> list[Candidate]:
    from aioslsk.protocol.primitives import AttributeKey

    out = []
    for r in (r for req in requests for r in req.results):
        for item in r.shared_items:
            if not item.filename.lower().endswith(ext) or item.filesize < 1_000_000:
                continue
            if not matches(item.filename, artist, title, need_artist=False):
                continue
            attrs = item.get_attribute_map()
            out.append(Candidate(r.username, item.filename, item.filesize, attrs.get(AttributeKey.BITRATE, 0),
                                 r.has_free_slots, r.queue_size, r.avg_speed))
    return sorted(out, key=Candidate.score, reverse=True)


async def _search(client, query: str, wait_s: float, tag: str = "") -> object:
    request = await client.searches.search(query)
    await asyncio.sleep(wait_s)
    files = sum(len(r.shared_items) for r in request.results)
    locked = sum(len(r.locked_results) for r in request.results)
    log(f"{tag} '{query}': {len(request.results)} users replied, {files} files" + (f" (+{locked} locked)" if locked else ""))
    return request


def _near_misses(requests: list, title: str, limit: int = 8) -> list[str]:
    """Audio files that mention the title but were filtered out, to explain a 'not found'."""
    tw = set(words(title))
    out = []
    for r in (r for req in requests for r in req.results):
        for item in r.shared_items:
            if tw & set(words(basename(item.filename))) and \
                    item.filename.lower().endswith((".mp3", ".flac", ".wav", ".m4a", ".aiff", ".ogg")):
                out.append(f"{r.username}: {item.filename}")
    return out[:limit]


async def _download(client, cand: Candidate, stall_s: float, max_s: float, tag: str = "") -> Path | None:
    from aioslsk.transfer.state import TransferState

    transfer = await client.transfers.download(cand.username, cand.path)
    start = last_change = time.time()
    last_bytes, last_pct = -1, -1
    while True:
        await asyncio.sleep(1)
        state = transfer.state.VALUE
        if state == TransferState.COMPLETE:
            return Path(transfer.local_path) if transfer.local_path else None
        if state in (TransferState.FAILED, TransferState.ABORTED, TransferState.INCOMPLETE):
            log(f"{tag}   failed: {transfer.fail_reason or transfer.abort_reason or state.name.lower()}")
            return None
        if transfer.bytes_transfered != last_bytes:
            last_bytes, last_change = transfer.bytes_transfered, time.time()
            pct = int(100 * last_bytes / cand.size) if cand.size else 0
            if pct // 25 != last_pct // 25:
                log(f"{tag}   {pct}%")
                last_pct = pct
        now = time.time()
        if now - last_change > stall_s or now - start > max_s:
            where = f" (queued, place {transfer.place_in_queue})" if transfer.place_in_queue else ""
            log(f"{tag}   no progress for {int(now - last_change)}s{where}, trying another source")
            await client.transfers.abort(transfer)
            return None


async def _fetch_all(queries: list[str], download_dir: Path, username: str, password: str, listen_port: int,
                     search_wait: float, tries: int, stall_s: float, max_s: float, parallel: int = 3,
                     on_ready=None) -> dict[str, Path]:
    from aioslsk.client import SoulSeekClient
    from aioslsk.settings import CredentialsSettings, Settings

    from aioslsk.exceptions import AuthenticationError, ListeningConnectionFailedError

    download_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(credentials=CredentialsSettings(username=username, password=password))
    settings.shares.scan_on_start = False
    settings.shares.download = str(download_dir)
    settings.network.listening.port = listen_port
    settings.network.listening.obfuscated_port = listen_port + 1
    # aioslsk logs every peer it can't reach (normal on Soulseek); keep the output readable
    logging.getLogger("aioslsk").setLevel(logging.CRITICAL)

    results: dict[str, Path] = {}
    gate = asyncio.Semaphore(parallel)

    async def fetch_one(q: str) -> None:
        async with gate:
            path = await _fetch_one(client, q, search_wait, tries, stall_s, max_s)
        if path:
            results[q] = path
            if on_ready:
                on_ready(q, path)

    # Start listening on the requested port; if something already uses it (SoulseekQt open,
    # a previous run still closing), try a few others rather than giving up.
    for port in (listen_port, listen_port + 10, 50300, 50310, 50320):
        settings.network.listening.port = port
        settings.network.listening.obfuscated_port = port + 1
        client = SoulSeekClient(settings)
        try:
            await client.start()
            break
        except (OSError, ListeningConnectionFailedError) as e:
            log(f"[soulseek] port {port} unavailable ({getattr(e, 'strerror', None) or e}); trying another")
            try:
                await client.stop()
            except Exception:
                pass
    else:
        raise RuntimeError("couldn't open a listening port for Soulseek (is SoulseekQt running?)")
    try:
        try:
            await client.login()
        except AuthenticationError as e:
            raise RuntimeError(f"Soulseek login failed ({e.reason or 'no reason given'}): check the password, "
                               f"and quit SoulseekQt if it's logged in as {username}") from None
        log(f"[soulseek] logged in as {username} (port {settings.network.listening.port}); "
            f"downloading up to {parallel} songs at a time")
        await asyncio.gather(*(fetch_one(q) for q in queries))
    finally:
        await client.stop()
    return results


async def _fetch_one(client, q: str, search_wait: float, tries: int, stall_s: float, max_s: float) -> Path | None:
    artist, title = parse_query(q)
    tag = f"[soulseek] {q}:"
    # The server sends a list of phrases it blocks (usually at a label's request); searches
    # containing one get no replies at all, so say so instead of searching in vain.
    blocked = [p for p in client.searches.excluded_search_phrases if p and p.lower() in q.lower()]
    if blocked:
        log(f"{tag} Soulseek blocks searches containing '{blocked[0]}'; it can't be downloaded here")
        return None
    plain = lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()  # "Essáy" -> "Essay"
    reqs = [await _search(client, plain(f"{artist} {title}").strip(), search_wait, tag)]
    cands, fmt = _candidates(reqs, artist, title, ".mp3"), "mp3"
    if not cands and artist:
        log(f"{tag} no mp3 yet; searching artist '{artist}' and filtering by title")
        reqs.append(await _search(client, plain(artist), search_wait, tag))
        cands = _candidates(reqs, artist, title, ".mp3")
    if not cands:
        cands, fmt = _candidates(reqs, artist, title, ".flac"), "flac"
        if cands:
            log(f"{tag} no mp3 anywhere; falling back to flac")
    if not cands:
        log(f"{tag} no mp3 or flac found")
        for m in _near_misses(reqs, title):
            log(f"{tag}   didn't match: {m}")
        return None
    log(f"{tag} {len(cands)} {fmt} candidates")
    tried = 0
    for cand in cands:
        if tried >= tries:
            break
        tried += 1
        kbps = f"{cand.bitrate}kbps" if cand.bitrate else "?kbps"
        log(f"{tag} trying {cand.username}: {basename(cand.path)} ({kbps}, {cand.size / 1e6:.1f} MB)")
        path = await _download(client, cand, stall_s, max_s, tag)
        if path and path.exists():
            log(f"{tag} downloaded -> {path.name}")
            return path
    if fmt == "mp3":  # every mp3 source failed: the flacs found by the same searches are next
        flacs = _candidates(reqs, artist, title, ".flac")
        if flacs:
            log(f"{tag} no mp3 source worked; trying {len(flacs)} flac candidates")
            for cand in flacs[:tries]:
                log(f"{tag} trying {cand.username}: {basename(cand.path)} ({cand.size / 1e6:.1f} MB)")
                path = await _download(client, cand, stall_s, max_s, tag)
                if path and path.exists():
                    log(f"{tag} downloaded -> {path.name}")
                    return path
    log(f"{tag} couldn't download any source")
    return None


def fetch_songs(queries: list[str], download_dir: Path, local_roots: list[Path], username: str | None,
                password: str | None, listen_port: int = 2234, search_wait: float = 20.0, tries: int = 5,
                stall_s: float = 60.0, max_s: float = 600.0, parallel: int = 3, on_ready=None) -> dict[str, Path]:
    """Resolve each 'Artist - Title' query to a local mp3/flac, downloading it from Soulseek if needed.
    on_ready(query, path) is called as soon as each song is available (local ones first), so the
    caller can start separating it while the rest download."""
    found: dict[str, Path] = {}
    missing = []
    for q in queries:
        local = find_local(q, local_roots)
        if local:
            log(f"[local] {q}: already have {local}")
            found[q] = local
            if on_ready:
                on_ready(q, local)
        else:
            missing.append(q)
    if missing:
        if not (username and password):
            log("[soulseek] set SLSK_USERNAME and SLSK_PASSWORD to download: " + "; ".join(missing))
            return found
        found.update(asyncio.run(_fetch_all(missing, download_dir, username, password, listen_port,
                                            search_wait, tries, stall_s, max_s, parallel, on_ready)))
    return found
