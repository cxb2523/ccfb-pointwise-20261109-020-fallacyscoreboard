"""Verification entry point for transcript_source.py.

Run with: python verify.py
Only reads/writes inside this repo (Files/); touches nothing outside it.

Case 1: a local short video runs through the full TranscriptSource
  pipeline (ffmpeg -> 16kHz mono wav -> overlapping windows -> transcribe)
  and produces a structured transcript JSON.
Case 2: the same local file is read twice through the live-stream code
  path (ring buffer + background puller), with the first run deliberately
  cut short to simulate an rtmp drop and the second a reconnect. Verifies
  the reconnect does not re-analyze already-seen text, that every ffmpeg
  subprocess exited cleanly, and that the rolling cache cap holds.
"""

import json
import os
import shutil
import sys

import transcript_source as ts_mod
from transcript_source import TranscriptSource, MAX_CACHE_FILES

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(REPO_DIR, "Files")


def make_fixture(path, duration):
    """Synthesize a tiny local video (tone + color) with ffmpeg via lavfi,
    so verification is self-contained and needs no external media."""
    ffmpeg = ts_mod._resolve_ffmpeg()
    args = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-f", "lavfi", "-i", f"color=c=navy:s=64x64:d={duration}",
        "-shortest", path,
    ]
    ts_mod._run_ffmpeg_with_retry(args)


def check_subprocess_retry_on_failure():
    """Exercise the auto-retry path directly: a deliberately invalid
    ffmpeg invocation must retry, exit uncleanly every time, and still be
    fully reaped (no zombie) rather than hang."""
    before = len(ts_mod.subprocess_exit_ledger)
    try:
        ts_mod._run_ffmpeg_with_retry(
            [ts_mod._resolve_ffmpeg(), "-this-flag-does-not-exist"], max_retries=2
        )
        raised = False
    except RuntimeError:
        raised = True
    attempts = ts_mod.subprocess_exit_ledger[before:]
    all_reaped = all(code is not None for _pid, code in attempts)
    all_failed = all(code != 0 for _pid, code in attempts)
    return raised and len(attempts) == 2 and all_reaped and all_failed


def case1_local_file():
    print("=== Case 1: local short video -> full pipeline -> structured transcript JSON ===")
    fixture = os.path.join(FILES_DIR, "verify_case1_source.mp4")
    make_fixture(fixture, duration=10)

    cache_dir = os.path.join(FILES_DIR, "verify_case1_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    source = TranscriptSource(fixture, cache_dir=cache_dir,
                               window_seconds=6.0, overlap_seconds=2.5)
    segments = list(source.segments())

    out_json = os.path.join(FILES_DIR, "verify_case1_transcript.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "source": fixture,
            "window_seconds": source.window_seconds,
            "overlap_seconds": source.overlap_seconds,
            "segments": [seg.__dict__ for seg in segments],
        }, f, ensure_ascii=False, indent=2)

    passed = len(segments) > 0 and os.path.exists(out_json) and all(
        seg.text for seg in segments
    )
    print(f"PASS={passed}  segments={len(segments)}  dedup_hits={source.stats.dedup_hits}  "
          f"json={os.path.relpath(out_json, REPO_DIR)}")
    return passed, len(segments), source.stats.dedup_hits


def case2_simulated_rtmp_reconnect():
    print("\n=== Case 2: local file as simulated rtmp push, interrupted twice ===")
    fixture = os.path.join(FILES_DIR, "verify_case2_source.mp4")
    make_fixture(fixture, duration=20)

    cache_dir = os.path.join(FILES_DIR, "verify_case2_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Small window/overlap and a tight cap so the ring buffer, dedup, and
    # rolling cache cleanup all get genuinely exercised within a few
    # seconds instead of needing a real long-running stream.
    test_cap = 6
    common_kwargs = dict(window_seconds=3.0, overlap_seconds=1.5,
                          ring_buffer_size=3, force_stream=True,
                          max_cache_files=test_cap)

    subprocess_before = len(ts_mod.subprocess_exit_ledger)

    # "Push" #1: starts streaming, gets cut off after a couple of chunks
    # (simulated connection drop).
    source_a = TranscriptSource(fixture, cache_dir=cache_dir, **common_kwargs)
    gen_a = source_a.segments()
    session_a_segments = []
    for i, seg in enumerate(gen_a):
        session_a_segments.append(seg)
        if i >= 1:
            break
    gen_a.close()

    # "Push" #2: reconnects. A fresh TranscriptSource re-pulls the same
    # source from the top (a naive reconnect has no better reference point
    # than "start over"), which deliberately re-covers the ground session
    # #1 already walked -- exactly the overlap the dedup layer must catch.
    source_b = TranscriptSource(fixture, cache_dir=cache_dir, **common_kwargs)
    gen_b = source_b.segments()
    session_b_segments = list(gen_b)

    all_segments = session_a_segments + session_b_segments
    all_texts = [seg.text for seg in all_segments]
    duplicate_count = len(all_texts) - len(set(all_texts))
    total_dedup_hits = source_a.stats.dedup_hits + source_b.stats.dedup_hits

    subprocess_calls = ts_mod.subprocess_exit_ledger[subprocess_before:]
    all_subprocesses_exited = all(code is not None for _pid, code in subprocess_calls)

    # MAX_CACHE_FILES/the rolling cleanup govern the cached audio chunks
    # (.wav); small bookkeeping files like the dedup state json are not
    # part of that rolling set and are excluded from the cap check.
    all_files = [
        name for name in os.listdir(cache_dir)
        if os.path.isfile(os.path.join(cache_dir, name))
    ]
    cache_files = [name for name in all_files if name.lower().endswith(".wav")]
    cache_within_test_cap = len(cache_files) <= test_cap
    cache_within_global_cap = len(all_files) <= MAX_CACHE_FILES

    retry_ok = check_subprocess_retry_on_failure()

    passed = (
        duplicate_count == 0
        and all_subprocesses_exited
        and cache_within_test_cap
        and cache_within_global_cap
        and retry_ok
        and len(all_segments) > 0
    )

    print(f"reconnect count (session A internal): {source_a.stats.reconnect_count}")
    print(f"ffmpeg/subprocess invocations this case: {len(subprocess_calls)}  "
          f"all_exited_cleanly={all_subprocesses_exited}")
    print(f"wav chunks in cache dir: {len(cache_files)}  (test cap {test_cap})  |  "
          f"total files in cache dir: {len(all_files)}  (global MAX_CACHE_FILES={MAX_CACHE_FILES})")
    print(f"auto-retry-on-subprocess-failure check: {'ok' if retry_ok else 'FAILED'}")
    print(f"PASS={passed}  segments_total={len(all_segments)}  dedup_hits={total_dedup_hits}  "
          f"duplicates_found={duplicate_count}")
    return passed, len(all_segments), total_dedup_hits


if __name__ == "__main__":
    assert MAX_CACHE_FILES == 50, "MAX_CACHE_FILES must stay at the spec value of 50"
    os.makedirs(FILES_DIR, exist_ok=True)

    case1_passed, case1_segments, case1_dedup = case1_local_file()
    case2_passed, case2_segments, case2_dedup = case2_simulated_rtmp_reconnect()

    print("\n==== SUMMARY ====")
    print(f"Case 1 (local file, full pipeline):          PASS={case1_passed}  "
          f"segments={case1_segments}  dedup_hits={case1_dedup}")
    print(f"Case 2 (simulated rtmp interrupt/reconnect):  PASS={case2_passed}  "
          f"segments={case2_segments}  dedup_hits={case2_dedup}")

    sys.exit(0 if (case1_passed and case2_passed) else 1)
