"""Unified transcript source for fallacyscoreboard.py.

TranscriptSource turns either a local media file or a live stream URL
(m3u8/rtmp/...) into a stream of time-sliced text segments that llm.py's
detect_fallacies() can consume. Local files are chunked up front; live
streams are pulled and sliced incrementally through a ring buffer so
analysis can start before the stream ends.
"""

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# Rolling cache cap for Files/ -- long-running live sessions must not grow
# the cache directory without bound.
MAX_CACHE_FILES = 50

# Default sliding-window size and overlap for chunking audio into
# transcribable pieces. Overlap keeps a sentence that straddles a window
# boundary from being cut in half on both sides.
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 2.5

SAMPLE_RATE = 16_000
CHANNELS = 1

SUBPROCESS_MAX_RETRIES = 3
SUBPROCESS_RETRY_BACKOFF = 1.0


def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg not found on PATH. Install ffmpeg (or the static_ffmpeg "
        "package) before using TranscriptSource."
    )


def _is_stream_url(source: str) -> bool:
    lowered = source.lower()
    if lowered.startswith(("rtmp://", "rtmps://", "http://", "https://")):
        return True
    if lowered.endswith((".m3u8", ".vtt", ".srt")):
        return True
    return False


def _is_subtitle_file(source: str) -> bool:
    return source.lower().endswith((".srt", ".vtt"))


@dataclass
class TextSegment:
    """One time-sliced piece of transcript text, ready for llm.py."""

    start: float
    end: float
    speaker: str
    text: str

    def dedup_key(self) -> str:
        bucket = round(self.start, 1)
        digest = hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()[:16]
        return f"{bucket}:{digest}"

    def as_line(self) -> str:
        """Render in the "start-end speaker: text" format llm.py expects."""
        return f"{self.start}-{self.end} {self.speaker}: {self.text}"


class Transcriber:
    """Pluggable transcription backend.

    Swap in a real engine by passing a callable to TranscriptSource, e.g.:
        TranscriptSource(path, transcribe_fn=whisper_transcribe)
    or point WHISPER_CLI_COMMAND at a CLI binary to shell out to it. The
    default stub lets the ring-buffer/dedup/backpressure plumbing be
    exercised and verified without a GPU or model weights installed.
    """

    def __init__(self, transcribe_fn: Optional[Callable[[str, float, float], str]] = None,
                 cli_command: Optional[list] = None):
        self.transcribe_fn = transcribe_fn
        self.cli_command = cli_command

    def transcribe(self, wav_path: str, start: float, end: float) -> str:
        if self.transcribe_fn is not None:
            return self.transcribe_fn(wav_path, start, end)
        if self.cli_command is not None:
            return self._transcribe_via_cli(wav_path)
        return self._stub_transcribe(wav_path, start, end)

    def _transcribe_via_cli(self, wav_path: str) -> str:
        # Reserved hook: e.g. cli_command = ["whisper-cli", "--model", "base"].
        # The wav path is appended; stdout (stripped) becomes the segment text.
        cmd = list(self.cli_command) + [wav_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip()

    @staticmethod
    def _stub_transcribe(wav_path: str, start: float, end: float) -> str:
        # Deterministic placeholder so the pipeline is end-to-end testable
        # without a real ASR model. Real deployments pass transcribe_fn.
        name = os.path.splitext(os.path.basename(wav_path))[0]
        return f"[stub transcript for {name} segment {start:.1f}-{end:.1f}s]"


def parse_subtitle_file(path: str) -> list:
    """Parse an external .srt/.vtt file directly, skipping transcription."""
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    is_vtt = path.lower().endswith(".vtt")
    blocks = [b.strip() for b in content.replace("\r\n", "\n").split("\n\n") if b.strip()]
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if is_vtt and lines and lines[0].upper().startswith("WEBVTT"):
            lines = lines[1:]
        lines = [l for l in lines if l.strip()]
        if not lines:
            continue
        # Optional leading numeric index line (SRT) or cue id (VTT).
        time_line_idx = 0
        if "-->" not in lines[0]:
            time_line_idx = 1
        if time_line_idx >= len(lines) or "-->" not in lines[time_line_idx]:
            continue
        time_line = lines[time_line_idx]
        start_str, end_str = [p.strip().split(" ")[0] for p in time_line.split("-->")]
        start = _timestamp_to_seconds(start_str)
        end = _timestamp_to_seconds(end_str)
        text = " ".join(lines[time_line_idx + 1:]).strip()
        if text:
            segments.append(TextSegment(start=start, end=end, speaker="SPEAKER_00", text=text))
    return segments


def _timestamp_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    while len(parts) < 3:
        parts.insert(0, "0")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# Process-exit ledger subprocesses report into, so callers (and verify.py)
# can confirm every ffmpeg child was reaped with a final exit code and none
# was left running or zombied. Thread-safe append-only list of
# (pid, returncode) tuples.
_subprocess_ledger_lock = threading.Lock()
subprocess_exit_ledger = []


def _run_ffmpeg_with_retry(args: list, max_retries: int = SUBPROCESS_MAX_RETRIES,
                            timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run an ffmpeg subprocess, retrying on abnormal exit.

    Ensures the process is always waited on (no zombies) even when it is
    killed after a timeout, and retries transient failures (e.g. a dropped
    network connection to a stream) with backoff before giving up.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        finally:
            # proc.communicate() already waits, but guard explicitly so a
            # process can never be left as a zombie even on unexpected path.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            with _subprocess_ledger_lock:
                subprocess_exit_ledger.append((proc.pid, proc.returncode))

        if timed_out:
            last_error = f"timed out after {timeout}s"
        elif proc.returncode == 0:
            return subprocess.CompletedProcess(args, 0, stdout, stderr)
        else:
            last_error = f"exit code {proc.returncode}: {stderr[-500:]}"

        if attempt < max_retries:
            time.sleep(SUBPROCESS_RETRY_BACKOFF * attempt)
    raise RuntimeError(f"ffmpeg failed after {max_retries} attempts: {last_error}")


def _enforce_cache_limit(cache_dir: str, max_files: int = MAX_CACHE_FILES,
                          keep: Optional[set] = None) -> None:
    """Roll off the oldest cached WAV chunks once Files/ exceeds max_files."""
    keep = keep or set()
    try:
        entries = [
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if name.lower().endswith(".wav") and os.path.join(cache_dir, name) not in keep
        ]
    except FileNotFoundError:
        return
    entries.sort(key=lambda p: os.path.getmtime(p))
    excess = len(entries) + len(keep) - max_files
    for path in entries[:max(0, excess)]:
        try:
            os.remove(path)
        except OSError:
            pass


def _wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


@dataclass
class _ReconnectStats:
    reconnect_count: int = 0
    dedup_hits: int = 0
    chunks_emitted: int = 0


class TranscriptSource:
    """Produces time-sliced text segments from a local file or a live stream.

    Local media: ffmpeg extracts a 16kHz mono WAV into Files/, which is then
    cut into fixed windows with a short overlap and transcribed window by
    window.

    Live stream (m3u8/rtmp/...): a background thread continuously pulls the
    stream with ffmpeg's segment muxer, dropping fixed-length WAV chunks
    into Files/ as they become available. A ring buffer (bounded queue)
    hands each chunk to the transcription stage as soon as it lands, so
    analysis is never blocked on the stream finishing. On a dropped
    connection the puller retries and resumes; timestamp+content-hash
    dedup prevents a chunk that was already transcribed before the drop
    from being analyzed twice after reconnect.

    Backpressure policy (pick one, documented trade-off):
      DROP_OLDEST_ON_FULL=True  -> when the ring buffer is full, the oldest
        pending chunk is discarded to make room for the new one. Keeps the
        pipeline realtime under sustained overload, at the cost of losing
        a chunk's worth of transcript if the consumer falls far behind.
      DROP_OLDEST_ON_FULL=False -> the puller blocks until the consumer
        frees a slot. Guarantees no transcript is ever lost, at the cost of
        falling behind live (and, for rtmp, of the upstream source itself
        stalling if it can't buffer).
    """

    DROP_OLDEST_ON_FULL = True

    def __init__(self, source: str, cache_dir: str = "Files",
                 window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
                 ring_buffer_size: int = 6,
                 transcriber: Optional[Transcriber] = None,
                 max_cache_files: int = MAX_CACHE_FILES,
                 force_stream: Optional[bool] = None):
        self.source = source
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds
        self.ring_buffer_size = ring_buffer_size
        self.transcriber = transcriber or Transcriber()
        self.max_cache_files = max_cache_files

        self.is_stream = force_stream if force_stream is not None else _is_stream_url(source)
        self.stats = _ReconnectStats()
        self._session_prefix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]

        # Dedup state is persisted next to the cache so that a *new*
        # TranscriptSource instance created after a real process-level
        # reconnect (ffmpeg/network died and the caller re-launched the
        # pull) still recognizes chunks it already emitted before the
        # drop, instead of re-analyzing them.
        self._dedup_state_path = os.path.join(
            self.cache_dir, f"{self._session_prefix}_dedup_state.json"
        )
        self._seen_keys = self._load_dedup_state()

    def _load_dedup_state(self) -> set:
        if os.path.exists(self._dedup_state_path):
            try:
                with open(self._dedup_state_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except (OSError, ValueError):
                return set()
        return set()

    def _save_dedup_state(self) -> None:
        try:
            with open(self._dedup_state_path, "w", encoding="utf-8") as f:
                json.dump(sorted(self._seen_keys), f)
        except OSError:
            pass

    # -- public API ---------------------------------------------------

    def segments(self) -> Iterator[TextSegment]:
        """Yield TextSegment objects, deduplicated, as they become ready."""
        if _is_subtitle_file(self.source):
            for seg in parse_subtitle_file(self.source):
                if self._register(seg):
                    yield seg
            return

        if self.is_stream:
            yield from self._stream_segments()
        else:
            yield from self._local_file_segments()

    def lines(self) -> Iterator[str]:
        """Yield segments pre-formatted for llm.py's read_line_from_file()."""
        for seg in self.segments():
            yield seg.as_line()

    # -- dedup ----------------------------------------------------------

    def _register(self, segment: TextSegment) -> bool:
        """Return True if this is a new segment, False if it's a dup."""
        key = segment.dedup_key()
        if key in self._seen_keys:
            self.stats.dedup_hits += 1
            return False
        self._seen_keys.add(key)
        self.stats.chunks_emitted += 1
        self._save_dedup_state()
        return True

    # -- local file path --------------------------------------------------

    def _local_file_segments(self) -> Iterator[TextSegment]:
        wav_path = self._extract_wav(self.source)
        try:
            duration = _wav_duration_seconds(wav_path)
            step = max(self.window_seconds - self.overlap_seconds, 1.0)
            start = 0.0
            index = 0
            while start < duration:
                end = min(start + self.window_seconds, duration)
                chunk_path = os.path.join(
                    self.cache_dir, f"{self._session_prefix}_local_{index:05d}.wav"
                )
                self._extract_window(wav_path, chunk_path, start, end - start)
                try:
                    text = self.transcriber.transcribe(chunk_path, start, end)
                    seg = TextSegment(start=start, end=end, speaker="SPEAKER_00", text=text)
                    if self._register(seg):
                        yield seg
                finally:
                    # Keep the full source wav alive; it's still needed for
                    # later windows even if the chunk cache rolls over.
                    _enforce_cache_limit(self.cache_dir, self.max_cache_files, keep={wav_path})
                if end >= duration:
                    break
                start += step
                index += 1
        finally:
            _enforce_cache_limit(self.cache_dir, self.max_cache_files, keep={wav_path})

    def _extract_wav(self, media_path: str) -> str:
        base = os.path.splitext(os.path.basename(media_path))[0]
        wav_path = os.path.join(self.cache_dir, f"{base}_16k_mono.wav")
        if os.path.exists(wav_path):
            return wav_path
        ffmpeg = _resolve_ffmpeg()
        args = [
            ffmpeg, "-y", "-i", media_path,
            "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            "-vn", wav_path,
        ]
        _run_ffmpeg_with_retry(args)
        return wav_path

    def _extract_window(self, wav_path: str, out_path: str, start: float, duration: float) -> None:
        ffmpeg = _resolve_ffmpeg()
        args = [
            ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", wav_path, "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            out_path,
        ]
        _run_ffmpeg_with_retry(args)

    # -- live stream path -------------------------------------------------

    def _stream_segments(self) -> Iterator[TextSegment]:
        chunk_queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=self.ring_buffer_size)
        stop_event = threading.Event()

        puller = threading.Thread(
            target=self._pull_stream_loop,
            args=(chunk_queue, stop_event),
            daemon=True,
        )
        puller.start()
        try:
            index = 0
            while True:
                chunk_path = chunk_queue.get()
                if chunk_path is None:  # sentinel: puller finished/gave up
                    break
                try:
                    duration = _wav_duration_seconds(chunk_path)
                    start = index * max(self.window_seconds - self.overlap_seconds, 1.0)
                    end = start + duration
                    text = self.transcriber.transcribe(chunk_path, start, end)
                    seg = TextSegment(start=start, end=end, speaker="SPEAKER_00", text=text)
                    if self._register(seg):
                        yield seg
                    index += 1
                finally:
                    _enforce_cache_limit(self.cache_dir, self.max_cache_files)
        finally:
            stop_event.set()
            puller.join(timeout=10)

    def _pull_stream_loop(self, chunk_queue: "queue.Queue", stop_event: threading.Event) -> None:
        """Continuously segment the live stream into WAV chunks.

        Each ffmpeg invocation captures one window-with-overlap worth of
        audio directly from the stream (ffmpeg -t <window>) and exits; this
        loop immediately relaunches it with -ss advanced by (window -
        overlap), which is what gives the 2-3s overlap between consecutive
        chunks. A stream drop surfaces as a non-zero ffmpeg exit, which is
        retried with backoff (_run_ffmpeg_with_retry) and counted as a
        reconnect; dedup on the consumer side guards against re-emitting
        content captured just before the drop.
        """
        ffmpeg = _resolve_ffmpeg()
        step = max(self.window_seconds - self.overlap_seconds, 1.0)
        elapsed = 0.0
        index = 0

        while not stop_event.is_set():
            chunk_path = os.path.join(
                self.cache_dir, f"{self._session_prefix}_live_{index:05d}.wav"
            )
            args = [
                ffmpeg, "-y",
                "-ss", f"{elapsed:.3f}",
                "-i", self.source,
                "-t", f"{self.window_seconds:.3f}",
                "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
                "-vn", chunk_path,
            ]
            try:
                _run_ffmpeg_with_retry(args, max_retries=SUBPROCESS_MAX_RETRIES,
                                        timeout=self.window_seconds + 30)
            except RuntimeError:
                self.stats.reconnect_count += 1
                if os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
                if stop_event.is_set():
                    break
                time.sleep(SUBPROCESS_RETRY_BACKOFF)
                continue

            if not os.path.exists(chunk_path) or _wav_duration_seconds(chunk_path) <= 0:
                # End of stream (file finished / vod exhausted at this offset).
                # A header-only wav (0 frames) is ffmpeg's signature for
                # "nothing left to read at this -ss offset", not an error.
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                break

            self._enqueue_chunk(chunk_queue, chunk_path, stop_event)
            elapsed += step
            index += 1

        # Best-effort sentinel: if the consumer already walked away (e.g.
        # the caller stopped iterating early) and the queue is still full,
        # don't block this thread forever trying to deliver it.
        try:
            chunk_queue.put_nowait(None)
        except queue.Full:
            pass

    def _enqueue_chunk(self, chunk_queue: "queue.Queue", chunk_path: str,
                        stop_event: threading.Event) -> None:
        if self.DROP_OLDEST_ON_FULL:
            while True:
                try:
                    chunk_queue.put_nowait(chunk_path)
                    return
                except queue.Full:
                    try:
                        dropped = chunk_queue.get_nowait()
                        if dropped and os.path.exists(dropped):
                            os.remove(dropped)
                    except queue.Empty:
                        pass
        else:
            while not stop_event.is_set():
                try:
                    chunk_queue.put(chunk_path, timeout=1)
                    return
                except queue.Full:
                    continue
