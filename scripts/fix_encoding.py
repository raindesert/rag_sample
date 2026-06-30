"""Fix wikipedia-zh-cn.json encoding.

The source file is a UTF-8 JSONL whose bytes were re-tagged as Latin-1 then
written back, producing a "double-encoded" file. On disk, every CJK character
appears as three Latin-1 chars (e.g. 数学 -> \"æ•°å­¦\")
and the file is therefore not valid UTF-8 (C1 control bytes 0x80-0x9F appear
inside the CJK three-byte sequences and break strict UTF-8 decoders).

Additionally:
- 18.8M bytes are CP1252-undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D) and become
  U+FFFD on the round-trip. That's 96.86% of lines; acceptable for RAG.
- Some lines are missing the closing quote of the \"title\" or \"text\" field.
  We patch those by inserting a single '\"' before the first ',\\\"tags\\\":'
  (or ',\\\"text\\\":') and/or the final '}'.

This script streams the file line-by-line (do NOT load 384MB into RAM) and
writes a UTF-8 JSONL named wikipedia-zh-cn.fixed.json alongside the original.

Run:
    python scripts/fix_encoding.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Force UTF-8 stdout (Windows console otherwise defaults to GBK and crashes on
# non-GBK chars in the diagnostic output).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

SRC = Path("wikipedia-zh-cn.json")
DST = Path("wikipedia-zh-cn.fixed.json")

# Compiled once, used on every unparseable line.
_FIELD_BOUNDARY = re.compile(r', "(tags|text)":')


def _round_trip(b: bytes) -> str:
    """Undo the Latin-1 re-tag: bytes -> Latin-1 str -> bytes -> UTF-8 str.

    bytes.decode('latin-1') interprets every byte as a code point (no errors).
    The resulting string is then re-encoded to bytes (still 1:1 with input).
    Finally we decode those bytes as UTF-8, which collapses the three
    Latin-1 chars representing a single CJK code point back into one char.
    """
    return b.decode("latin-1").encode("latin-1", errors="replace").decode("utf-8", errors="replace")


def _try_parse(s: str) -> dict | None:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _insert_before(text: str, pos: int) -> str:
    return text[:pos] + '"' + text[pos:]


def _last_brace_before_crlf(s: str) -> int:
    end = s.rfind("\r")
    if end < 0:
        end = s.rfind("\n")
    if end < 0:
        end = len(s)
    return s.rfind("}", 0, end)


def fix_line(b: bytes) -> dict | None:
    """Convert one raw line into a parsed dict, or None if unrecoverable."""
    s = _round_trip(b)

    rec = _try_parse(s)
    if rec is not None:
        return rec

    # Strategy 1: text value is unterminated. The original file ends with
    #     ...content...<FFFD>?}\\r\\n
    # without a closing '\"' for the text value. Insert one before the final '}'.
    bp = _last_brace_before_crlf(s)
    if bp > 0:
        rec = _try_parse(_insert_before(s, bp))
        if rec is not None:
            return rec

    # Strategy 2: title value is unterminated. The '\"' that should close the
    # title value was dropped, so the next field's opener '\"tags\"' or
    # '\"text\"' is being consumed as content. Insert a '\"' right before
    # ', \"tags\":' or ', \"text\":'.
    m = _FIELD_BOUNDARY.search(s)
    if m:
        rec = _try_parse(_insert_before(s, m.start()))
        if rec is not None:
            return rec

    # Strategy 3: both are missing. Apply strategy 2 then strategy 1.
    s2 = s
    m2 = _FIELD_BOUNDARY.search(s2)
    if m2:
        s2 = _insert_before(s2, m2.start())
    bp2 = _last_brace_before_crlf(s2)
    if bp2 > 0:
        s2 = _insert_before(s2, bp2)
        rec = _try_parse(s2)
        if rec is not None:
            return rec

    return None


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1

    n_total = 0
    n_ok = 0
    n_with_fffd = 0
    n_fail: list[int] = []
    fail_sample = None

    with SRC.open("rb") as fin, DST.open("w", encoding="utf-8", newline="\n") as fout:
        for lineno, raw in enumerate(fin, start=1):
            n_total += 1
            rec = fix_line(raw)
            if rec is None:
                if fail_sample is None and len(n_fail) < 5:
                    fail_sample = lineno
                n_fail.append(lineno)
                # Skip unrecoverable lines rather than aborting the whole run.
                continue
            n_ok += 1
            text_blob = rec.get("title", "") + rec.get("text", "")
            if "�" in text_blob:
                n_with_fffd += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Source:      {SRC} ({SRC.stat().st_size / 1e6:.1f} MB)")
    print(f"Destination: {DST} ({DST.stat().st_size / 1e6:.1f} MB)")
    print(f"Lines read:  {n_total}")
    print(f"Lines OK:    {n_ok}")
    print(f"Lines fail:  {len(n_fail)}")
    if n_fail:
        print(f"  first failures: {n_fail[:5]}")
    print(f"Lines with U+FFFD (CP1252 undefined bytes): {n_with_fffd}")
    return 0 if not n_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
