"""Flask PDF annotation app with genuine neural pen-stroke handwriting.

Install: python3 -m pip install -U flask pymupdf pillow pydantic google-genai
Also install Node.js 18+ (node -v).
Run: python3 app.py
"""

import asyncio
import os
import hashlib
import io
import importlib.util
import tempfile
from contextvars import ContextVar
import math
import json
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request
import uuid
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Literal

import fitz
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance
from flask import Flask, jsonify, render_template, request, send_file
from google import genai
from google.genai import types
from pydantic import BaseModel
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
HOSTED = os.environ.get("VERCEL") == "1" or os.environ.get("AUTOANNOTATOR_HOSTED") == "1"
HOSTED_MAX_UPLOAD = 100 * 1024 * 1024
HOSTED_MAX_PAGES = 100
HOSTED_MAX_OUTPUT = 100 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
request_work = ContextVar("annotation_request_work", default=None)


def work_setting(name, default=None):
    return (request_work.get() or {}).get(name, default)


def model_name():
    return work_setting("model", MODEL_NAME)


def cache_directory():
    return work_setting("cache_dir", CACHE_DIR)


def engine_directory():
    return work_setting("engine_dir", ENGINE_DIR)


# ------------------------------------------------------------
# SETTINGS: exact settings from the Handwriter website
# ------------------------------------------------------------
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # Keep the model that works with your key.
HAND_NUMBER = 11                   # Displayed Hand 11 = zero-based index 10.
NEATNESS = 0.10
LINE_SPACING = 2.70        # Extra breathing room between actual handwritten lines
INK_RGB = (0, 0, 0)             # Solid black, NOT the site's blue-black.
PEN_WIDTH_POINTS = 0.98    # Thin ballpoint/gel pen rather than Sharpie
SOFTNESS = 0.22                 # Subtle scan edge only, no gray tint.
# Margin-first layout: preserve original-size reading whenever possible.
# Only shrink slightly when the right side cannot fit handwritten notes.
MIN_NOTE_WIDTH_PT = 145.0  # only a small reduction when existing right margin is genuinely too narrow
NOTE_LAYOUT_PADDING = 8.0
MAX_MARGIN_SHRINK = 0.90  # At most 12% EXTRA reduction beyond mandatory paper fit
RENDER_RESOLUTION = 5

# The writing uses REAL neural pen strokes rather than Pillow fonts.
# Let handwriting begin in the original PDF's empty right-hand margin,
# rather than wasting that space and starting beyond the page edge.
NOTE_BORROW_MAX = 120       # Legacy setting, no fixed reserve in margin-first layout
NOTE_TEXT_GAP = 3         # Minimum gap between printed prose and ink
NOTE_OUTER_PADDING = 3    # Blank space at the new outer page edge
NOTE_TARGET_SCALE = 2.18  # Keep handwriting large across PDFs
NOTE_SCALE_JITTER = 0.12  # Small natural size variation note-to-note

# Optional, subtle scanned-paper finish across the entire output.
# This flattens the final PDF into page images (best for printing).
SCAN_FINISH = True
SCAN_DPI = 165
SCAN_BLUR_PX = 0.55
SCAN_PAPER_COLOR = (255, 255, 255)  # neutral white paper
SCAN_TINT_STRENGTH = 0.0            # remove warm tint while keeping blur + grain
SCAN_CONTRAST = 1.035                # keep black pen ink strong
SCAN_GRAIN_SIGMA = 23               # fine-grained scanning texture
SCAN_GRAIN_STRENGTH = 0.32          # multipliers map noise to paper only
SCAN_JPEG_QUALITY = 88

# Every exported page is US Letter PORTRAIT: 8.5 by 11 inches.
LETTER_PORTRAIT_WIDTH = 8.5 * 72   # 612 points
LETTER_PORTRAIT_HEIGHT = 11 * 72   # 792 points
LETTER_BORDER_PT = 9               # small safe print margin

MIN_SECONDS_BETWEEN_REQUESTS = 1.0
REQUEST_TIMEOUT_SECONDS = 75
MAX_ATTEMPTS = 5
MAX_PAGE_CHARS = 16000
CACHE_VERSION = 16  # same Gemini annotations; safer placement is rendering-only
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")

BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / ".annotation_cache"
ENGINE_DIR = BASE / ".handwriter_engine"

# Pin to verified published version instead of allowing a future update
# to change the output without warning. MIT licensed.
HANDWRITER_COMMIT = "3c11fe12014d510bd5e74a91aca96aa5f772ca57"
HANDWRITER_BASE = (
    "https://raw.githubusercontent.com/Eeman1113/handwriter/"
    + HANDWRITER_COMMIT
)
ENGINE_URL = HANDWRITER_BASE + "/src/lib/handwriting.js"
WEIGHTS_URL = HANDWRITER_BASE + "/public/d.bin"

# Handheld fluorescent highlighter appearance (drawn shapes, not PDF highlight widgets).
# 0.35–0.55 is deliberately translucent, so printed text stays legible.
MARKER_OPACITY = 0.63
MARKER_END_OVERLAP = 0.29
MARKER_WOBBLE_PT = 1.45
MARKER_EDGE_OPACITY = 0.32    # pigment pooling along the upper/lower borders
MARKER_EDGE_THICKNESS_PT = 1.55
MARKER_TIP_OPACITY = 0.38     # stronger felt-tip deposit at both ends

HIGHLIGHT_COLORS = {
    "pink": (1.00, 0.18, 0.72),
    "yellow": (1.00, 0.95, 0.00),
    "blue": (0.00, 0.82, 1.00),
}
STOP_HEADINGS = {"NOTES", "ENDNOTES", "REFERENCES", "BIBLIOGRAPHY", "WORKS CITED"}

# This is a small Node adapter to the *actual* neural Handwriter engine.
# The original open-source model and weights are downloaded once locally.
# No font is used to draw the margin notes.
NODE_BRIDGE = r'''
import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { loadModel, generate, bounds, styleCount } from './handwriting.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const modelBytes = await fs.readFile(join(here, 'd.bin'));
const ab = modelBytes.buffer.slice(
  modelBytes.byteOffset, modelBytes.byteOffset + modelBytes.byteLength
);
// loadModel() calls fetch(); feed it the downloaded local weights.
globalThis.fetch = async () => ({ ok: true, arrayBuffer: async () => ab });
await loadModel('handwriter-local-model');

const input = JSON.parse(readFileSync(0, 'utf8'));
const styleIndex = input.hand - 1;
if (styleIndex < 0 || styleIndex >= styleCount()) {
  throw new Error(`Hand ${input.hand} not available (have ${styleCount()})`);
}

function wrapWords(text, target=19) {
  // Prefer at least two words per line, even in a narrow margin.
  // A single trailing word is folded into the preceding line.
  const words = text.trim().split(/\s+/).filter(Boolean);
  const rows = [];
  let current = [];
  for (const word of words) {
    if (current.length >= 2 && (current.join(' ').length + 1 + word.length > target)) {
      rows.push(current.join(' '));
      current = [word];
    } else {
      current.push(word);
    }
  }
  if (current.length === 1 && rows.length) {
    let last = rows.pop().split(' ');
    if (last.length > 2) {
      current.unshift(last.pop());
      rows.push(last.join(' '));
    } else {
      current.unshift(...last);
    }
  }
  if (current.length) rows.push(current.join(' '));
  return rows;
}

const output = [];
for (const item of input.notes) {
  try {
    const wrapped = wrapWords(item.text, item.wrap_chars ?? 34);
    const result = generate(wrapped.join('\n'), {
      style: styleIndex,
      bias: input.neatness,
      seed: item.seed >>> 0,
    });

    const gap = 10 * input.line_spacing;
    let ascent = 0, descent = 0, maxWidth = 0;
    const boxes = result.lines.map(l => {
      const b = bounds(l.strokes);
      ascent = Math.max(ascent, b.maxY);
      descent = Math.max(descent, -b.minY);
      maxWidth = Math.max(maxWidth, b.width);
      return b;
    });
    const pad = 5;
    const firstBaseline = pad + ascent;
    const strokes = [];
    result.lines.forEach((line, i) => {
      const b = boxes[i];
      const xOffset = pad - b.minX;
      const baseline = firstBaseline + i * gap;
      line.strokes.forEach(stroke => {
        if (stroke.length > 1) {
          strokes.push(stroke.map(([x,y]) => [
            Math.round((xOffset + x) * 100) / 100,
            Math.round((baseline - y) * 100) / 100
          ]));
        }
      });
    });
    output.push({
      id: item.id,
      lines: result.lines.length,
      width: maxWidth + 2 * pad,
      height: ascent + descent + gap * Math.max(0, result.lines.length - 1) + 2 * pad,
      strokes,
    });
  } catch (e) {
    output.push({ id: item.id, error: String(e) });
  }
}
process.stdout.write(JSON.stringify(output));
'''


class HighlightItem(BaseModel):
    quote: str
    color: Literal["pink", "yellow", "blue"]
    line_number: int  # required 1-based source line; no ungrounded position guessing


class MarginNote(BaseModel):
    anchor_quote: str
    note_text: str
    line_number: int  # required 1-based source line for this note


class PageAnnotation(BaseModel):
    page_number: int
    highlights: list[HighlightItem]
    margin_notes: list[MarginNote]


class ChunkAnnotations(BaseModel):
    pages: list[PageAnnotation]


jobs = {}
jobs_lock = threading.Lock()
processing_lock = threading.Lock()


def update_job(job_id, **data):
    with jobs_lock:
        jobs[job_id].update(data)


def status_for(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return {k: v for k, v in job.items() if k != "pdf_bytes"} if job else None


def page_done(job_id, page_num, cached=False):
    with jobs_lock:
        job = jobs[job_id]
        job["done"] += 1
        job["message"] = (
            f"Page {page_num}: reused saved annotations" if cached
            else f"Page {page_num}: generated and saved"
        )
        print(f"[{job['done']}/{job['total']}] {job['message']}", flush=True)


# ------------------------------------------------------------
# INSTALL THE REAL HANDWRITER SOURCE AND MODEL WEIGHTS
# ------------------------------------------------------------

def ensure_handwriter():
    node = shutil.which("node")
    if not node:
        spec = importlib.util.find_spec("nodejs_wheel")
        if spec and spec.origin:
            bundled = Path(spec.origin).parent / "bin" / "node"
            if bundled.is_file():
                node = str(bundled)
    if not node:
        raise RuntimeError(
            "Handwriter requires Node.js 18+. Install it from https://nodejs.org/ "
            "(or brew install node), then restart Flask."
        )
    version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
    if version.returncode != 0:
        raise RuntimeError("Could not run Node.js: " + version.stderr)
    major = int(re.search(r"v(\d+)", version.stdout).group(1))
    if major < 18:
        raise RuntimeError("Please install Node.js version 18 or later")

    engine_dir = engine_directory()
    engine_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, min_bytes in (
        ("handwriting.mjs", ENGINE_URL, 10000),
        ("d.bin", WEIGHTS_URL, 100000),
    ):
        target = engine_dir / filename
        if target.exists() and target.stat().st_size >= min_bytes:
            continue
        vendored = BASE / "vendor" / "handwriter" / filename
        if vendored.is_file() and vendored.stat().st_size >= min_bytes:
            shutil.copyfile(vendored, target)
            continue
        if work_setting("hosted"):
            raise RuntimeError("Bundled handwriting model is unavailable")
        print("Downloading Handwriter component:", filename, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "PDF-Annotator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                content = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from the open-source Handwriter "
                f"project: {exc}. Check internet access."
            ) from exc
        if len(content) < min_bytes:
            raise RuntimeError(f"Handwriter download incomplete: {filename}")
        temp = target.with_suffix(".download")
        temp.write_bytes(content)
        temp.replace(target)

    bridge = engine_dir / "render_notes.mjs"
    if not bridge.exists() or bridge.read_text(encoding="utf-8") != NODE_BRIDGE:
        bridge.write_text(NODE_BRIDGE, encoding="utf-8")
    return node, bridge


def generate_neural_strokes(notes, node, bridge):
    """Call the original JS RNN once for all notes, returning stroke geometry."""
    if not notes:
        return {}
    payload = {
        "hand": HAND_NUMBER,
        "neatness": NEATNESS,
        "line_spacing": LINE_SPACING,
        "notes": notes,
    }
    result = subprocess.run(
        [node, str(bridge)],
        input=json.dumps(payload, ensure_ascii=True),
        capture_output=True,
        text=True,
        timeout=60 if work_setting("hosted") else 600,
        cwd=str(bridge.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Handwriter neural engine failed: " + result.stderr[-1500:]
        )
    try:
        generated = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("Handwriter did not return valid stroke data") from exc
    bad = [g for g in generated if g.get("error")]
    if bad:
        print("Handwriter skipped some notes" if work_setting("hosted") else f"Warnings from Handwriter: {bad[:3]}", flush=True)
    return {g["id"]: g for g in generated}


# ------------------------------------------------------------
# FIND THE BODY TEXT, SKIPPING TITLES / REFERENCES / ENDNOTES
# ------------------------------------------------------------

def _line_entries(page, textpage=None):
    """Ordered visible text lines with their precise PDF coordinates."""
    kw = {"textpage": textpage} if textpage is not None else {}
    result = []
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, **kw)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = " ".join(span["text"].strip() for span in line.get("spans", []))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                result.append((text, fitz.Rect(line["bbox"]), line))
    result.sort(key=lambda item: (round(item[1].y0 / 4), item[1].x0))
    return result


def _is_citation_section(lines, title_index):
    """A standalone NOTES heading is not enough to stop the whole PDF.

    Require bibliographic-looking material following it, to avoid stopping
    on main-text headings such as 'Notes on the experiment'.
    """
    after = [t for t, *_ in lines[title_index + 1: title_index + 18]]
    citation_signals = sum(bool(re.search(
        r"(?:\b(?:18|19|20)\d{2}\b|\b(?:ibid|vol\.?|press|publisher|"
        r"university|journal|doi|http|et al|pp?\.)\b|^[\[\(]?\d+[\]\).])",
        line, re.I)) for line in after)
    return citation_signals >= 2


def _endnotes_cutoff(lines, page_index, total_pages):
    """Only end the reading when a LATE, confirmed bibliography starts.

    No more early stopping just because 'NOTES' is used as an ordinary heading.
    """
    if page_index < max(2, int(total_pages * 0.45)):
        return None
    for i, (text, rect, _) in enumerate(lines):
        upper = re.sub(r"\s+", " ", text).strip().upper().rstrip(":")
        if upper in STOP_HEADINGS and _is_citation_section(lines, i):
            return rect.y0
    return None


def _get_tessdata():
    """Locate English OCR data on typical macOS Homebrew installations."""
    candidates = [
        Path("/opt/homebrew/share/tessdata"),
        Path("/usr/local/share/tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    env = __import__('os').environ.get('TESSDATA_PREFIX')
    if env:
        candidates.insert(0, Path(env))
    for item in candidates:
        if (item / 'eng.traineddata').is_file():
            return str(item)
    return None


def _ocr_page(page, page_number):
    """Get an OCR TextPage; critically, it includes word bounding boxes.

    It supports exact position-based highlighting on *image-only* PDFs.
    """
    params = {"language": "eng", "dpi": 185, "full": True}
    tessdata = _get_tessdata()
    if tessdata:
        params['tessdata'] = tessdata
    try:
        return page.get_textpage_ocr(**params)
    except Exception as exc:
        raise RuntimeError(
            f"Page {page_number} needs OCR, but Tesseract could not run: {exc}. "
            "On macOS run `brew install tesseract`; restart Terminal and Flask."
        ) from exc


def _is_prose_line(text, rect, page, *, ocr=False):
    """Avoid metadata, running heads, page numbers, source code and captions."""
    if rect.y0 < page.rect.height * 0.065 or rect.y1 > page.rect.height * 0.94:
        return False
    text = text.strip()
    if not text or len(text) < (24 if ocr else 12):
        return False
    if any(phrase in text.lower() for phrase in (
        'this content downloaded', 'all use subject to',
        'jstor is a not-for-profit', 'stable url:',
        'chapter title:', 'book title:', 'published by:',
    )):
        return False
    if text.upper() == text and len(text) < 90:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < 0.48:
        return False
    if ocr and len(re.findall(r"[A-Za-z]{2,}", text)) < 4:
        return False
    if re.match(r"^(?:FIGURE|PATTERN|TABLE)\s+(?:[A-Z0-9]|ONE|TWO|THREE|FOUR)", text, re.I):
        return False
    # A long standalone title can still have enough letters to pass the
    # thresholds above. Short title-cased lines are not prose sentences.
    words = re.findall(r"[A-Za-z]{2,}", text)
    if (3 <= len(words) <= 12 and not re.search(r"[.!?;]", text)
            and sum(w[0].isupper() for w in words) / len(words) > 0.60):
        return False
    return True


def typical_body_size(doc):
    """Find the dominant font on selectable-text PDFs."""
    scores = Counter()
    for page in doc:
        native = page.get_text('text')
        if len(native.strip()) < 150:
            continue
        for text, rect, line in _line_entries(page):
            if len(text) < 35 or not _is_prose_line(text, rect, page):
                continue
            for span in line.get('spans', []):
                if 4.0 <= span['size'] <= 14:
                    scores[round(span['size'] * 2) / 2] += len(span['text'])
        if len(scores) > 8 and sum(scores.values()) > 12000:
            break
    return scores.most_common(1)[0][0] if scores else 10.0


def native_layer_has_broken_word_spacing(text):
    """Detect old scans with a *searchable* layer of separated letters.

    Example from the user's Noll PDF: "The d i g i t a l c o m p u t e r".
    This must be OCR'd despite having selectable characters, otherwise
    Gemini and the line-level highlight match cannot agree on words.
    """
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 90:
        return False
    singles = sum(len(word) == 1 for word in words)
    return singles / len(words) > 0.38


def collect_prose_pages(doc, job_id=None):
    """Read ordinary PDFs and scan-only PDFs via on-demand OCR.

    Returns (reading, confirmed_endnotes_page, ocr_textpages, ocr_page_count).
    Tuples in reading keep the original 3-field format used by the rest
    of the app. OCR TextPages are kept in memory for search + exact positioning.
    """
    typical = typical_body_size(doc)
    reading = []
    ocr_textpages = {}
    scanned_pages = 0
    stopped_at = None
    total = len(doc)

    for index, page in enumerate(doc):
        page_num = index + 1
        native_text = page.get_text('text').strip()
        textpage = None
        uses_ocr = len(native_text) < 180 or native_layer_has_broken_word_spacing(native_text)
        if uses_ocr:
            if job_id:
                update_job(job_id, message=f'OCR scanning page {page_num}/{total} ...')
            textpage = _ocr_page(page, page_num)
            scanned_pages += 1
            # Keep the actual page wrapper alive: OCR TextPage is bound to it.
            ocr_textpages[page_num] = (page, textpage)

        entries = _line_entries(page, textpage=textpage)
        cutoff = _endnotes_cutoff(entries, index, total)
        text_all = ' '.join(t for t, *_ in entries)
        if index == 0 and any(term in text_all.upper() for term in (
            'COVER SHEET FOR TECHNICAL MEMORANDUM',
            'JSTOR IS A NOT-FOR-PROFIT SERVICE',
        )):
            print(f'Skipping cover page {page_num}', flush=True)
            continue

        # OCR usually places each source line into a separate PDF text block.
        # So never require a 90-character multi-line *block* for scanned PDFs.
        prose = []
        rectangles = []
        for text, rect, line in entries:
            if cutoff is not None and rect.y0 >= cutoff - 1:
                continue
            if not _is_prose_line(text, rect, page, ocr=uses_ocr):
                continue
            if not uses_ocr:
                size_count = sum(len(s['text']) for s in line['spans'])
                matching = sum(
                    len(s['text']) for s in line['spans']
                    if abs(s['size'] - typical) <= 1.25
                )
                if size_count and matching / size_count < 0.60:
                    continue
            rectangles.append(fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1))
            prose.append(text)

        joined = '\n'.join(f'[L{i:03d}] {line}' for i, line in enumerate(prose, 1)).strip()
        # A few misrecognized caption words do not make a figure into a reading.
        threshold = 200 if uses_ocr else 180
        if len(joined) >= threshold and len(prose) >= 3:
            reading.append((page_num, joined[:MAX_PAGE_CHARS], rectangles))
            print(f'Page {page_num}: {"OCR" if uses_ocr else "text"} — {len(joined)} body characters', flush=True)
        else:
            print(f'Page {page_num}: no substantial prose ({len(joined)} chars), skipped', flush=True)

        if cutoff is not None:
            stopped_at = page_num
            print(f'Confirmed bibliography/endnotes on PDF page {page_num}', flush=True)
            break

    return reading, stopped_at, ocr_textpages, scanned_pages


def rect_area(rect):
    """Rectangle area compatible with older and newer PyMuPDF versions."""
    r = fitz.Rect(rect)
    return max(0.0, float(r.width)) * max(0.0, float(r.height))


def _normalized_words(text):
    """Turn scanned/OCR words into comparable tokens without punctuation."""
    text = text.casefold().replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.findall(r"[a-z0-9]+", text)


def _fuzzy_match_prose(page, quote, prose_rects, textpage=None):
    """Recover an OCR highlight when a few characters/word boundaries differ.

    Only considers individual WORD COORDINATES inside approved prose, and
    requires a strong textual similarity. Never highlights headers or art.
    """
    needle = _normalized_words(quote)
    if len(needle) < 2:
        return []

    opts = {"textpage": textpage} if textpage is not None else {}
    words = page.get_text("words", sort=False, **opts)
    accepted = []
    for w in words:
        box = fitz.Rect(w[:4])
        if not any(rect_area(box & r) >= 0.30 * max(rect_area(box), 0.1)
                   for r in prose_rects):
            continue
        for token in _normalized_words(str(w[4])):
            accepted.append((token, box))

    n = len(needle)
    if len(accepted) < 2:
        return []

    target = " ".join(needle)
    # Very short quotes demand particularly close matches.
    threshold = 0.85 if n <= 3 else (0.77 if n <= 6 else 0.74)
    low = max(2, round(n * 0.75))
    high = min(len(accepted), max(low, round(n * 1.28) + 1))
    best = None
    best_score = 0.0
    best_loc = None

    for size in range(low, high + 1):
        for i in range(len(accepted) - size + 1):
            window = accepted[i:i + size]
            # Don't accidentally match a phrase spanning unrelated sections
            # or across the physical gap between two columns.
            if any(abs(window[j + 1][1].y0 - window[j][1].y0) > 43
                   for j in range(len(window) - 1)):
                continue
            # A new line is allowed; a backwards jump to a different column isn't.
            if any(window[j + 1][1].y0 < window[j][1].y0 - 6
                   for j in range(len(window) - 1)):
                continue
            candidate = " ".join(tok for tok, _ in window)
            ratio = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
            if ratio > best_score:
                best_score = ratio
                best_loc = i
                best = window

    if not best or best_score < threshold:
        return []

    # Return a separate rectangle per physical line so one highlight
    # doesn't paint a huge diagonal region over blank space.
    line_boxes = []
    for _, b in best:
        if line_boxes and abs(b.y0 - line_boxes[-1][-1].y0) <= 5:
            line_boxes[-1].append(b)
        else:
            line_boxes.append([b])
    out = []
    for line in line_boxes:
        out.append(fitz.Rect(min(b.x0 for b in line), min(b.y0 for b in line),
                             max(b.x1 for b in line), max(b.y1 for b in line)))
    return out


def find_at_numbered_line(page, quote, prose_rects, line_number, *, textpage=None):
    """Only highlight the numbered OCR line Gemini actually selected.

    This avoids accidental matches elsewhere on the page. If OCR has small
    spelling errors, compare only within that particular line's word boxes.
    Never extend a highlight into another line or a picture.
    """
    if not isinstance(line_number, int) or not 1 <= line_number <= len(prose_rects):
        return []
    rect = prose_rects[line_number - 1]
    opts = {'textpage': textpage} if textpage is not None else {}
    exact = page.search_for(quote.strip(), clip=rect, quads=True, **opts)
    # Some PyMuPDF/OCR versions ignore clip when a textpage is supplied.
    # Verify geometrically that every returned hit belongs to this line.
    exact = [hit for hit in exact
             if rect_area(hit.rect & rect) >= 0.82 * max(rect_area(hit.rect), 0.01)]
    if exact:
        return exact

    wanted = _normalized_words(quote)
    if len(wanted) < 2:
        return []
    contained = []
    for word in page.get_text('words', sort=True, **opts):
        b = fitz.Rect(word[:4])
        if rect_area(b & rect) < 0.60 * max(rect_area(b), 0.01):
            continue
        toks = _normalized_words(str(word[4]))
        if len(toks) == 1:
            contained.append((toks[0], b))
    if not contained:
        return []

    target = ' '.join(wanted)
    n = len(wanted)
    best_score = 0.0
    best_span = None
    for size in range(max(2, n - 1), min(len(contained), n + 1) + 1):
        for i in range(len(contained) - size + 1):
            segment = contained[i:i+size]
            candidate = ' '.join(w for w, _ in segment)
            score = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
            first = SequenceMatcher(None, wanted[0], segment[0][0]).ratio()
            last = SequenceMatcher(None, wanted[-1], segment[-1][0]).ratio()
            if score > best_score and first >= 0.65 and last >= 0.65:
                best_score = score
                best_span = segment
    # High threshold on a single physical line prevents plausible-looking
    # but incorrect fuzzy matches. Missing highlights are reported, not guessed.
    if best_span is None or best_score < 0.72:
        return []
    return [fitz.Rect(min(b.x0 for _, b in best_span),
                      min(b.y0 for _, b in best_span),
                      max(b.x1 for _, b in best_span),
                      max(b.y1 for _, b in best_span))]


def find_in_prose(page, quote, rects, quads=False, textpage=None):
    quote = quote.strip()
    if not quote or not rects:
        return []

    if textpage is None:
        # Search each paragraph/line then the whole page, restricting hits
        # to the known prose coordinates so headings can't be marked.
        for rect in rects:
            hits = page.search_for(quote, clip=rect, quads=quads)
            if hits:
                return hits

    params = {"quads": quads}
    if textpage is not None:
        params["textpage"] = textpage
    hits = page.search_for(quote, **params)
    accepted = []
    for hit in hits:
        box = hit.rect if hasattr(hit, "rect") else fitz.Rect(hit)
        if any(box.intersects(r) and (
            rect_area(box & r) / max(rect_area(box), 0.01) > 0.35
        ) for r in rects):
            accepted.append(hit)
    if accepted:
        return accepted

    # OCR frequently swaps letters, breaks words, and loses hyphens.
    # Use the text's actual OCR word boxes rather than giving up.
    return _fuzzy_match_prose(page, quote, rects, textpage=textpage)



# ------------------------------------------------------------
# SAFETY NET FOR SCANS WITH FEW VISIBLE HIGHLIGHTS
# ------------------------------------------------------------

_COMMON_WORDS = {
    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'were', 'into',
    'have', 'been', 'their', 'then', 'when', 'also', 'will', 'what',
    'where', 'which', 'they', 'more', 'some', 'these', 'there', 'such',
    'about', 'after', 'before', 'very', 'just', 'only', 'each', 'our',
    'who', 'are', 'but', 'not', 'any', 'was', 'had', 'his', 'her', 'you',
}
_IMPORTANCE_CUES = {
    'however', 'because', 'instead', 'although', 'therefore',
    'purpose', 'reason', 'result', 'difference', 'discovered',
    'created', 'original', 'research', 'important', 'computer',
    'patterns', 'artistic', 'design', 'method', 'conclusion',
    'process', 'experiment', 'demonstrates', 'produce', 'possible',
    'develop', 'development', 'problem', 'limitation', 'significant',
    'new', 'first', 'artists', 'future', 'evidence', 'argument',
    'example', 'similar', 'contrast', 'changed',
}


def _line_words(page, rect, textpage=None):
    """Words in one already-approved prose line, in left-to-right order."""
    opts = {'textpage': textpage} if textpage is not None else {}
    found = []
    for word in page.get_text('words', sort=False, **opts):
        box = fitz.Rect(word[:4])
        if rect_area(box & rect) >= 0.55 * max(0.1, rect_area(box)):
            token = str(word[4]).strip()
            if token and re.search(r'[A-Za-z0-9]', token):
                found.append((token, box))
    found.sort(key=lambda item: item[1].x0)
    # OCR can produce two overlapping copies of a word. Ignore near duplicates.
    unique = []
    for text, box in found:
        if unique and text == unique[-1][0] and rect_area(box & unique[-1][1]) > 0.65 * min(rect_area(box), rect_area(unique[-1][1])):
            continue
        unique.append((text, box))
    return unique


def _best_verified_span(words, quote):
    """Choose a true word span in a given OCR line. Never guess coordinates."""
    target_tokens = _normalized_words(quote)
    if len(target_tokens) < 2 or len(words) < 2:
        return None, 0.0
    target = ' '.join(target_tokens)
    length = len(target_tokens)
    best_score = 0.0
    best_rect = None
    # Handle one extra split/combined OCR word, but not unrelated lines.
    for window in range(max(2, length - 2), min(len(words), length + 2) + 1):
        for start in range(len(words) - window + 1):
            segment = words[start:start + window]
            actual = ' '.join(' '.join(_normalized_words(w)) for w, _ in segment)
            if not actual.strip():
                continue
            score = SequenceMatcher(None, target, actual, autojunk=False).ratio()
            if score > best_score:
                boxes = [b for _, b in segment]
                best_rect = fitz.Rect(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                                      max(b.x1 for b in boxes), max(b.y1 for b in boxes))
                best_score = score
    return best_rect, best_score


def _choose_key_phrase(words):
    """Select a short actual phrase from a line if Gemini provided no usable span.

    This uses text coordinates only: never paints guessed words or images.
    """
    if len(words) < 5:
        return None, -1.0
    count = min(10, max(5, round(len(words) * 0.65)))
    best = None
    best_score = -1e6
    for start in range(len(words) - count + 1):
        window = words[start:start + count]
        toks = [_normalized_words(w)[0] for w, _ in window if _normalized_words(w)]
        if len(toks) < 5:
            continue
        content = [w for w in toks if w not in _COMMON_WORDS and len(w) >= 4]
        if len(content) < 2:
            continue
        cue = sum(w in _IMPORTANCE_CUES for w in toks)
        long_word = sum(len(w) >= 7 for w in content)
        boxes = [b for _, b in window]
        heights = sorted(b.height for b in boxes)
        typical_h = heights[len(heights) // 2]
        midpoints = sorted((b.y0 + b.y1) / 2 for b in boxes)
        baseline = midpoints[len(midpoints) // 2]
        # Scanned OCR sometimes merges TWO printed lines into one large word
        # box. Refuse that box instead of painting over both lines and the gap.
        if typical_h > 21 or any(abs(m - baseline) > 6.0 for m in midpoints):
            continue
        # Favour informative phrasing and natural clause boundaries.
        score = len(content) + 1.7 * cue + 0.35 * long_word
        if len(toks) >= 2 and toks[0] in {'following', 'paragraphs', 'section'}:
            score -= 4
        score -= 0.015 * abs(start - (len(words) - count) * 0.48)
        if score > best_score:
            # Keep the marker to one handwritten/typed baseline even if the
            # OCR engine reports slightly oversized glyph bounding boxes.
            half_h = min(6.7, max(4.2, typical_h * 0.49))
            best = fitz.Rect(min(b.x0 for b in boxes), baseline - half_h,
                             max(b.x1 for b in boxes), baseline + half_h)
            best_score = score
    return best, best_score


def fill_sparse_highlights(page, prose_rects, existing_line_ids, page_number,
                           *, textpage=None, over_image=True):
    """Prevent genuine prose pages from being left unhighlighted.

    Reuse Gemini-selected lines wherever possible. For OCR phrases whose exact
    spellings cannot be recovered, use short, real word spans on other readable
    prose lines. Never mark titles, images or pages with only a few prose lines.
    """
    if not prose_rects:
        return 0
    word_lines = []
    for line_id, rect in enumerate(prose_rects):
        words = _line_words(page, rect, textpage=textpage)
        line_text = ' '.join(w for w, _ in words).casefold()
        # Transitional/navigation sentences are not substantive claims.
        if any(phrase in line_text for phrase in (
            'following paragraphs', 'the next section', 'the paragraphs below',
            'shown in the following', 'see appendix', 'see reverse side',
        )):
            continue
        if len(words) >= 6:
            phrase, score = _choose_key_phrase(words)
            if phrase is not None and phrase.width >= 15:
                word_lines.append((line_id, phrase, score))
    if len(word_lines) < 5:
        return 0

    # At least one highlight across most two-to-three-line stretches of prose.
    # Cap at 12 on a very dense page; never force color onto short pages.
    desired = min(12, max(4, round(len(word_lines) * 0.42)))
    current = len(existing_line_ids)
    needed = max(0, desired - current)
    if not needed:
        return 0

    used = set(existing_line_ids)
    picked = []
    color_names = ('yellow', 'blue', 'pink')
    for _ in range(needed):
        options = []
        for line_id, phrase, score in word_lines:
            if line_id in used:
                continue
            # Prefer useful lines spaced apart from already-marked lines.
            distance = min((abs(line_id - old) for old in used), default=3)
            spacing_bonus = min(distance, 5) * 0.9
            options.append((score + spacing_bonus, line_id, phrase))
        if not options:
            break
        _, line_id, phrase = max(options, key=lambda row: (row[0], -row[1]))
        picked.append((line_id, phrase))
        used.add(line_id)

    for index, (line_id, phrase) in enumerate(picked):
        color = HIGHLIGHT_COLORS[color_names[(page_number + index) % len(color_names)]]
        seed = int.from_bytes(hashlib.sha256(
            f'auto|{page_number}|{line_id}'.encode('utf-8')
        ).digest()[:8], 'big')
        draw_handheld_marker(page, phrase, color, seed, over_image=over_image)

    print(f'Page {page_number}: added {len(picked)} verified prose highlights '
          f'to fill sparse coverage (target {desired}, original {current})', flush=True)
    return len(picked)

# ------------------------------------------------------------
# PERSIST COMPLETED AI ANNOTATIONS
# ------------------------------------------------------------

def load_cache(pdf_hash):
    cache_dir = cache_directory()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdf_hash}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                data.get("hash") == pdf_hash
                and data.get("model") == model_name()
                and data.get("version") == CACHE_VERSION
            ):
                return data, path
        except (OSError, ValueError):
            pass
        # A new prompt version should create fresh annotations, but preserve
        # the previous cache (which may contain many finished pages).
        old_path = cache_dir / (
            f"{pdf_hash}_older_cache_{datetime.now():%Y%m%d_%H%M%S}_"
            f"{uuid.uuid4().hex[:6]}.json"
        )
        shutil.copy2(path, old_path)
        print("Older annotation cache backed up:", old_path, flush=True)
    return {
        "hash": pdf_hash, "model": model_name(),
        "version": CACHE_VERSION, "pages": {},
    }, path


def save_cache(data, path):
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def clean_note(text):
    text = re.sub(r"^[\s\-•*]+", "", text.strip())
    text = text.replace("’", "'").replace("—", "-").replace("–", "-")
    text = re.sub(r"[.!;:]", "", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(", -")
    # Only keep a question mark when the note is a question.
    is_question = text.endswith("?")
    text = text.replace("?", "").strip()
    return (text + "?") if is_question else text


def retry_delay(error, attempt=1):
    match = re.search(r"retry in\s+([\d.]+)\s*s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    return min(30.0, (2.0 ** attempt) + random.uniform(1.0, 3.0))


# ------------------------------------------------------------
# GEMINI: INFORMAL, NATURAL MARGIN NOTES WITH DENSITY-BASED COUNTS
# ------------------------------------------------------------

async def annotate_page(client, page_number, text, job_id, last_request):
    highlight_target = min(18, max(5, round(len(text) / 185)))
    # Dense full pages need four distinct sentence-style margin notes.
    # Short readings receive fewer so the margins do not look crowded.
    if len(text) >= 2600:
        note_target = 4
    elif len(text) >= 1500:
        note_target = 3
    else:
        note_target = 2
    prompt = f"""
Annotate page {page_number} of an academic reading using structured JSON.
Return a pages list containing exactly one page_number={page_number}.
The supplied text contains only the actual reading paragraphs.
Each physical line is labeled [L001], [L002], etc. The label is an index,
not part of the printed text. Use it to give exact placement coordinates.

CRITICAL COORDINATE RULES:
For every highlight include line_number as the integer shown in the [L...] label
on the SAME line as the quoted text. Choose highlights that are contained
within a single labeled line (no wrapping to the next line).
For every margin note include line_number of the line it discusses.
Quotes must NOT include any [L...] label. line_number MUST be correct.
If an OCR line contains minor mistakes, quote exactly the OCR line text.
Never guess a different line to make a quote fit.

HIGHLIGHTS: Aim for {highlight_target} useful exact quotes, usually 3–12
consecutive words, using pink/yellow/blue. Focus on important arguments,
evidence, examples, dates that actually matter, and significant distinctions.
Do not highlight random phrases. Never highlight title, metadata, headers,
figure captions, endnotes, bibliographies or page numbers.

MARGIN NOTES: Write {note_target} distinct casual student margin notes.
On this page, use exactly {note_target} notes if there are enough different ideas.
For a full page of prose there should be FOUR notes, each mostly a complete
casual sentence of 9–20 words. On shorter pages use the smaller target shown.
At least three of four full-page notes should be real sentences rather than
short labels. The remaining note can be a short reaction or genuine question.
Place their anchor quotes across DIFFERENT PARTS of the reading: ideally
an early paragraph, a middle paragraph, a later paragraph, and near the end.
Do NOT cluster notes around a single paragraph or repeat the same idea.
Write in very ordinary language, as if summarizing to yourself while reading.
Each note should make one clear observation tied directly to its paragraph.
Do not add filler just to make it longer. Use no periods or exclamation points;
only questions end in a question mark. Never prefix literal bullet symbols.
Good full-sentence examples:
  There were financial limits so they had to think about what they could actually afford
  He keeps saying women matter but seems worried about changing the university
  I think their evidence is weaker than they want everyone to believe
  They were scared calling it art would make their research look less serious
  A lot of this decision seems to come down to money and reputation
Good shorter notes:
  the report assumes too much here
  nobody really agrees on the cost
  why did they ignore this evidence?
Bad notes:
  fear of change
  important point
  reflects the broader narrative of technological transformation
Avoid: demonstrates, underscores, embodies, highlights, resonates, reflects,
fundamental, broader narrative, serves as, nuanced, complex interplay.

QUOTES: Each highlight.quote and each margin_note.anchor_quote must occur
verbatim in the supplied reading, with no made-up text. Return fewer if needed.

READING:
{text}
"""
    max_attempts = MAX_ATTEMPTS
    timeout = REQUEST_TIMEOUT_SECONDS
    for attempt in range(1, max_attempts + 1):
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last_request[0])
        if wait > 0:
            await asyncio.sleep(wait)
        update_job(job_id, message=f"Analyzing page {page_number} (attempt {attempt}/{max_attempts})")
        last_request[0] = time.monotonic()
        try:
            response = await asyncio.wait_for(
                client.models.generate_content(
                    model=model_name(), contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ChunkAnnotations,
                        temperature=0.28,
                        max_output_tokens=4000,
                    ),
                ),
                timeout=timeout,
            )
            if not response or not response.text:
                raise ValueError("Gemini returned empty text")

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

            parsed = ChunkAnnotations.model_validate_json(raw_text)
            if not parsed.pages:
                raise ValueError("Gemini returned no page annotations")

            page_data = parsed.pages[0]
            if page_data.page_number != page_number:
                page_data.page_number = page_number

            page_data.margin_notes = [
                MarginNote(anchor_quote=n.anchor_quote, note_text=clean_note(n.note_text),
                           line_number=n.line_number)
                for n in page_data.margin_notes if clean_note(n.note_text)
            ]
            return page_data
        except Exception as exc:
            err_name = type(exc).__name__
            err = str(exc) or err_name
            lower = err.lower()
            print(f"Gemini page {page_number} attempt {attempt}: {err_name} - {err}", flush=True)

            if any(p in lower for p in (
                "invalid api key", "api key not valid", "permission_denied", "401", "403"
            )):
                raise RuntimeError(f"Gemini API authentication error: {err}") from exc
            if "429" in lower and any(p in lower for p in ("per day", "daily", "requests per day")):
                raise RuntimeError("Daily Gemini quota reached. Completed pages saved.") from exc

            # If all attempts exhausted on this page, fall back gracefully so rest of 100-page document completes
            if attempt == max_attempts:
                print(f"Warning: Page {page_number} could not be annotated after {max_attempts} attempts ({err_name}). Continuing to preserve rest of document.", flush=True)
                return PageAnnotation(page_number=page_number, highlights=[], margin_notes=[])

            # Backoff and retry
            if "429" in lower or "resource_exhausted" in lower:
                delay = retry_delay(err, attempt)
                update_job(job_id, message=f"Page {page_number}: rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_attempts})")
            elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in lower or "timed out" in lower:
                delay = min(15.0, 2.0 * attempt + random.uniform(1.0, 3.0))
                update_job(job_id, message=f"Page {page_number}: API timed out, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_attempts})")
            else:
                delay = min(20.0, 2.5 * attempt + random.uniform(0.5, 2.0))
                update_job(job_id, message=f"Page {page_number}: retrying in {delay:.0f}s ({err_name})")

            await asyncio.sleep(delay)

    return PageAnnotation(page_number=page_number, highlights=[], margin_notes=[])


async def generate_annotations(api_key, job_id, pdf_hash, reading, stopped_at, total_pages, force_regenerate=False):
    if work_setting("demo"):
        update_job(job_id, status="running", total=len(reading), message="Rendering the labeled sample; no AI request is made")
        results = []
        for page_number, text, _ in reading:
            source_lines = [(int(n), value) for n, value in re.findall(r"\[L(\d+)\] ([^\n]+)", text)]
            highlights = [HighlightItem(quote=value, color=["pink", "yellow", "blue"][i % 3], line_number=n)
                          for i, (n, value) in enumerate(source_lines) if i in (0, 2, 4)]
            notes = [MarginNote(anchor_quote=value, note_text=note, line_number=n)
                     for (n, value), note in zip(source_lines[1::4], ["try explaining this idea in your own words", "what evidence would help test this claim?"])]
            results.append(PageAnnotation(page_number=page_number, highlights=highlights, margin_notes=notes))
            page_done(job_id, page_number)
        return results
    cache, cache_path = load_cache(pdf_hash)
    if force_regenerate:
        # Back up the previous successful pages before replacing the checkpoint.
        # New pages are still saved after each successful Gemini response.
        if cache.get("pages"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = cache_directory() / f"{pdf_hash}_before_regenerate_{stamp}_{uuid.uuid4().hex[:6]}.json"
            backup_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print("Previous annotation cache backed up:", backup_path, flush=True)
        cache["pages"] = {}
        save_cache(cache, cache_path)
    update_job(
        job_id, status="running", total=len(reading),
        skipped=total_pages - len(reading), notes_start=stopped_at,
        message="Checking saved annotations..."
    )
    results = []
    last_request = [0.0]
    sdk = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )
    try:
        async with sdk.aio as client:
            for page_number, text, _ in reading:
                saved = cache["pages"].get(str(page_number))
                if saved is not None:
                    try:
                        item = PageAnnotation.model_validate(saved)
                        if item.page_number == page_number:
                            results.append(item)
                            page_done(job_id, page_number, cached=True)
                            continue
                    except (ValueError, TypeError):
                        pass
                item = await annotate_page(client, page_number, text, job_id, last_request)
                cache["pages"][str(page_number)] = item.model_dump()
                save_cache(cache, cache_path)
                results.append(item)
                page_done(job_id, page_number)
    finally:
        sdk.close()
    return results


# ------------------------------------------------------------
# RENDER GENUINE NEURAL STROKE GEOMETRY IN SOLID BLACK
# ------------------------------------------------------------

def choose_note_area(page, prose_rects, planned_left=None):
    """Use existing blank margin and never move notes over actual source ink.

    The planned writing boundary comes from the original page before it was
    copied onto Letter paper. Checking the prose rectangles again protects
    against inaccurate blank-margin estimation on scanned documents.
    """
    page_width = page.rect.width
    detected_right = max((r.x1 for r in prose_rects), default=0.0)
    if planned_left is None:
        planned_left = max(0.0, detected_right) + NOTE_TEXT_GAP
    left = max(float(planned_left), detected_right + NOTE_TEXT_GAP)
    right = page_width - NOTE_LAYOUT_PADDING
    # Do NOT silently force the note back over printed text if the margin
    # is too narrow. Callers will avoid generation if < 35pt remains.
    return min(left, right), max(0.0, right - left)



def estimated_wrap_chars(width_pt):
    """Use the actual writing area; prefer 2-3 words to a single word per line."""
    return max(16, min(64, round(max(40.0, width_pt - 8) / 8.5)))


def strokes_to_black_png(generated, width_pt, height_pt, seed):
    """Render Handwriter neural strokes as a fine black pen.

    Retains neural path variation with gentle pressure fluctuations;
    does not draw the thick, filled strokes of a felt-tip marker.
    """
    W = max(10, round(width_pt * RENDER_RESOLUTION))
    H = max(10, round(height_pt * RENDER_RESOLUTION))
    s = RENDER_RESOLUTION
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    model_width = max(1.0, float(generated["width"]))
    model_height = max(1.0, float(generated["height"]))
    rng = random.Random(seed)
    target_scale = NOTE_TARGET_SCALE + rng.uniform(-NOTE_SCALE_JITTER, NOTE_SCALE_JITTER)
    max_fit = min((width_pt - 8) / model_width, (height_pt - 8) / model_height)
    # Keep handwriting large whenever possible; only shrink if it truly cannot fit.
    scale = target_scale if max_fit >= target_scale else max_fit
    if max_fit < target_scale * 0.72:
        scale = max_fit
    if scale <= 0:
        return None

    dx, dy = rng.uniform(1.5, 3.5), rng.uniform(2.5, 4.5)
    base_px = PEN_WIDTH_POINTS * s

    for stroke in generated["strokes"]:
        if len(stroke) < 2:
            continue
        points = [
            ((dx + x * scale) * s, (dy + y * scale) * s)
            for x, y in stroke
        ]

        # Pen pressure varies naturally, but stays thinner than a marker.
        weight = rng.uniform(0.85, 1.10)
        if rng.random() < 0.14:
            weight *= rng.uniform(0.73, 0.88)
        phase = rng.uniform(-math.pi, math.pi)
        n = len(points)

        for j in range(n - 1):
            fraction = (j + 0.5) / max(n - 1, 1)
            pressure = (
                0.79
                + 0.18 * math.sin(math.pi * fraction)
                + 0.045 * math.sin(2 * math.pi * fraction + phase)
            )
            thickness = max(2, round(base_px * weight * pressure))
            draw.line(
                (points[j], points[j + 1]),
                fill=255,
                width=thickness,
                joint="curve",
            )
            if j == 0 or j == n - 2:
                x, y = points[j] if j == 0 else points[j + 1]
                radius = thickness / 2
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=255,
                )

    # Tilt the entire handwritten note, never independent characters.
    mask = mask.rotate(
        rng.uniform(-1.6, 1.6),
        resample=Image.Resampling.BICUBIC,
        expand=False,
    )

    # Faint ink bleed around very dark cores, not a uniform gray font.
    if SOFTNESS > 0:
        blurred = mask.filter(
            ImageFilter.GaussianBlur(SOFTNESS * s * 0.33)
        )
        mask = ImageChops.lighter(
            mask,
            blurred.point(lambda a: round(a * 0.075)),
        )

    # Subtle absorption irregularity keeps most strokes nearly black.
    grain = Image.effect_noise((W, H), 14).point(
        lambda v: 243 + round(v * 12 / 255)
    )
    mask = ImageChops.multiply(mask, grain)

    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    image.putalpha(mask)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def estimate_printed_right_edge(page):
    """Estimate where actual content ends, ignoring the blank PDF background.

    Prefer text-block geometry (including imperfect searchable layers).
    For image-only scans, look for dark ink columns instead of treating the
    full-page scan bitmap as printed matter. No AI request is required.
    """
    width, height = page.rect.width, page.rect.height
    text_edges = []
    text_chars = 0
    for block in page.get_text('blocks'):
        if len(block) < 7 or block[6] != 0:
            continue
        left, top, right, bottom, body = block[:5]
        if (top < height * .08 or top > height * .90
                or right <= left or bottom <= top):
            continue
        if not isinstance(body, str) or len(body.strip()) < 55:
            continue
        lower = body.lower()
        if 'this content downloaded' in lower or 'all use subject to' in lower:
            continue
        letters = sum(ch.isalpha() for ch in body)
        if letters < 35 or letters / max(1, len(body)) < .32:
            continue
        text_edges.append(right)
        text_chars += len(body)

    if text_chars >= 160 and text_edges:
        return min(width, max(text_edges)), 'selectable text'

    # A visual scan may contain a full-page background image, but actual
    # printed letters occupy only part of it. Analyze the dark-pixel extent
    # at low resolution; avoid extracting strings from artwork alone.
    resolution = min(1.0, 520.0 / width, 680.0 / height)
    pix = page.get_pixmap(
        matrix=fitz.Matrix(resolution, resolution),
        colorspace=fitz.csGRAY, alpha=False,
    )
    px_width, px_height = pix.width, pix.height
    x0, x1 = int(px_width * .055), int(px_width * .985)
    y0, y1 = int(px_height * .085), int(px_height * .89)
    counts = [0] * max(0, x1 - x0)
    pixels = pix.samples
    for y in range(y0, y1, 2):
        row = pixels[y * pix.stride + x0:y * pix.stride + x1]
        for x, value in enumerate(row):
            if value < 130:
                counts[x] += 1
    threshold = max(2, int((y1 - y0) * .003))
    ink_cols = [i for i, count in enumerate(counts) if count >= threshold]
    if not ink_cols:
        return width * .8, 'blank/unreadable'
    # Robust to isolated dust near the page edge.
    index = min(len(ink_cols) - 1, int(.985 * len(ink_cols)))
    return min(width, (x0 + ink_cols[index] + 2) / resolution), 'scanned ink'


def fit_on_letter_portrait(source_doc):
    """FIRST STEP: output 8.5 x 11 portrait without shrinking unnecessarily.

    If the original page already fits, keep it at 100% size and in the same
    upper-left position. Use its existing right-hand white margin for notes.
    Only reduce the page a little more when the available right margin is too
    narrow. Never crop, rotate, upscale, stretch, or enlarge the sheet.

    Returns (formatted_doc, {1-based page_number: safe_note_left}).
    """
    result = fitz.open()
    positions = {}
    paper_w, paper_h = LETTER_PORTRAIT_WIDTH, LETTER_PORTRAIT_HEIGHT
    for index in range(len(source_doc)):
        source = source_doc[index]
        size = source.rect
        if size.width <= 0 or size.height <= 0:
            raise RuntimeError(f'Invalid dimensions on page {index + 1}')

        # A standard 612x792 page is not reduced, unless its own right
        # whitespace is actually inadequate for handwritten notes.
        fit = min(1.0, paper_w / size.width, paper_h / size.height)
        right_edge, measurement = estimate_printed_right_edge(source)
        desired_right_edge = paper_w - NOTE_LAYOUT_PADDING - NOTE_TEXT_GAP - MIN_NOTE_WIDTH_PT
        scale = fit
        if right_edge * fit > desired_right_edge:
            scale = max(fit * MAX_MARGIN_SHRINK, desired_right_edge / max(right_edge, 1))
            scale = min(scale, fit)
        # Hold the original sheet at top left, even if a slight reduction
        # leaves extra whitespace below and on the right.
        display_w, display_h = size.width * scale, size.height * scale
        page_out = result.new_page(width=paper_w, height=paper_h)
        page_out.show_pdf_page(
            fitz.Rect(0, 0, display_w, display_h), source_doc, index,
            keep_proportion=True, overlay=True,
        )
        content_right = min(paper_w, right_edge * scale)
        safe_left = content_right + NOTE_TEXT_GAP
        positions[index + 1] = safe_left
        print(
            f'Margin-first page {index + 1}/{len(source_doc)}: '
            f'paper 8.5x11 portrait, scale {scale:.1%} '
            f'({measurement}); writing begins at x={safe_left:.1f}pt',
            flush=True,
        )
    return result, positions


def apply_scan_finish(doc):
    """Warm slightly blurred scanned-paper appearance over the entire page.

    This rasterizes the PDF: the result is printable, but text is no longer
    selectable. Grain is a narrow, near-white MULTIPLIER, not mid-gray
    multiplication (which would turn a white page much too dark).
    """
    final = fitz.open()
    zoom = SCAN_DPI / 72
    for page_number, page in enumerate(doc):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        if SCAN_BLUR_PX:
            img = img.filter(ImageFilter.GaussianBlur(SCAN_BLUR_PX))

        # Maintain dark printed type and firmly pressed black ink.
        img = ImageEnhance.Contrast(img).enhance(SCAN_CONTRAST)

        # Warm paper WITHOUT turning truly black strokes gray.
        # A multiplicative tint changes whites but preserves pure blacks.
        tint_channels = tuple(round(
            (1.0 - SCAN_TINT_STRENGTH) * 255
            + SCAN_TINT_STRENGTH * channel
        ) for channel in SCAN_PAPER_COLOR)
        img = ImageChops.multiply(
            img, Image.new("RGB", img.size, tint_channels)
        )

        # Fine independent brightness fluctuations in the paper and the ink.
        # Typical multipliers here stay near 245-255, NEVER around 128.
        noise = Image.effect_noise(img.size, SCAN_GRAIN_SIGMA)
        grain = noise.point(lambda v: max(236, min(255, round(
            250 + (v - 128) * SCAN_GRAIN_STRENGTH
        ))))
        texture = Image.merge("RGB", (grain,) * 3)
        img = ImageChops.multiply(img, texture)

        jpeg = io.BytesIO()
        img.save(
            jpeg,
            format="JPEG",
            quality=SCAN_JPEG_QUALITY,
            subsampling=0,
            optimize=True,
        )
        out_page = final.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=jpeg.getvalue())
        print(f"Scan finish: page {page_number + 1}/{len(doc)}", flush=True)
    return final


# ------------------------------------------------------------
# PHYSICAL PAGE INK DETECTION (NOT JUST TEXT OR PROSE BOUNDS)
# ------------------------------------------------------------

def make_source_ink_mask(page):
    """Snapshot the page BEFORE adding any highlights or handwriting.

    Dark pixels include text, sidebars, photographs, diagram lines, image
    borders, headers and footers. They all count as occupied page content.
    """
    pix = page.get_pixmap(
        matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY, alpha=False,
    )
    gray = Image.frombytes('L', (pix.width, pix.height), pix.samples)
    return gray.point(lambda value: 255 if value < 210 else 0)


def is_actual_blank_space(ink_mask, rect, guard=4.0):
    """Only accept rectangles with almost no printed/drawn material.

    A tiny number of scan specks is allowed; lines, photographs, headings,
    vertical labels, or existing page text are not. A safety border protects
    against handwriting touching a neighboring printed element.
    """
    width, height = ink_mask.size
    x0 = max(0, int(math.floor(rect.x0 - guard)))
    y0 = max(0, int(math.floor(rect.y0 - guard)))
    x1 = min(width, int(math.ceil(rect.x1 + guard)))
    y1 = min(height, int(math.ceil(rect.y1 + guard)))
    if x1 <= x0 or y1 <= y0:
        return False
    if rect.x0 < 0 or rect.y0 < 0 or rect.x1 > width or rect.y1 > height:
        return False
    crop = ink_mask.crop((x0, y0, x1, y1))
    dark_count = crop.histogram()[255]
    coverage = dark_count / max(1, (x1 - x0) * (y1 - y0))
    return coverage < 0.0015


def get_blank_note_zones(page, prose_rects, planned_left=None):
    """Possible places; all candidates still undergo a real ink test."""
    if not prose_rects:
        return {}
    width, height = page.rect.width, page.rect.height
    prose_left = min(r.x0 for r in prose_rects)
    prose_bottom = max(r.y1 for r in prose_rects)
    right_start, _ = choose_note_area(page, prose_rects, planned_left)
    zones = {
        'right': (right_start, width - 9, 16.0, height - 14),
    }
    left_end = min(prose_left - 10, width * 0.38)
    if left_end - 9 >= 76:
        zones['left'] = (9.0, left_end, 16.0, height - 14)
    bottom_top = max(prose_bottom + 12, height * 0.52)
    if height - bottom_top >= 72:
        zones['bottom'] = (12.0, width - 12, bottom_top, height - 12)
    return zones


def find_ink_free_note_rect(ink_mask, bounds, target_height, anchor_y,
                            occupied, zone_name):
    """Search for a note location entirely on empty paper.

    Try x positions just beyond any vertical border or side label and y
    positions near the passage first. Do not reduce the source page or put
    writing on top of existing marks as a fallback.
    """
    zone_left, zone_right, zone_top, zone_bottom = bounds
    zone_left = max(8.0, float(zone_left))
    zone_right = min(float(ink_mask.width) - 8.0, float(zone_right))
    zone_top = max(8.0, float(zone_top))
    zone_bottom = min(float(ink_mask.height) - 8.0, float(zone_bottom))
    width_available = zone_right - zone_left
    height_available = zone_bottom - zone_top
    if width_available < 65 or height_available < 56:
        return None

    height = min(float(target_height), height_available - 2)
    if height < 56:
        return None

    # Allow the handwriting to move just to the right of a printed sidebar,
    # but do not make it a single-word column unnecessarily.
    minimum_width = 78 if zone_name != 'bottom' else 175
    if width_available < minimum_width:
        minimum_width = max(62, width_available - 4)
    if width_available < minimum_width:
        return None
    left_choices = []
    offset = 1.0
    while zone_left + offset + minimum_width <= zone_right:
        left_choices.append(zone_left + offset)
        offset += 8.0
    if not left_choices:
        left_choices = [zone_left + 1]

    y_limit = zone_bottom - height
    center = min(max(float(anchor_y) - height * 0.18, zone_top), y_limit)
    y_choices = [zone_top + 7.0 * i for i in range(int((y_limit-zone_top)/7.0) + 1)]
    y_choices.extend([zone_top, y_limit, center])
    # Choose the nearest usable position to the note's source passage.
    y_choices = sorted(set(round(y, 2) for y in y_choices), key=lambda y: abs(y - center))

    for y in y_choices:
        for left in left_choices:
            rect = fitz.Rect(left, y, zone_right-1, y+height)
            if rect.width < minimum_width or rect.height < 56:
                continue
            if any(rect.intersects(old + (-7, -7, 7, 7)) for old in occupied):
                continue
            if is_actual_blank_space(ink_mask, rect):
                return rect
    return None


def plan_note_zones(page, notes, prose_rects, planned_left=None, ocr_page=None):
    """Plan handwriting across REAL empty left/right/bottom whitespace.

    Original printed text stays exactly where it is. A bottom slot is only
    offered below the last prose line and above any detected printed footer.
    Notes never overlap prose; if there is no safe left/bottom region, use
    the right margin. The same plan is used before and after neural generation.
    """
    if not notes or not prose_rects:
        return {}
    width, height = page.rect.width, page.rect.height
    left_edge = min(r.x0 for r in prose_rects)
    bottom_edge = max(r.y1 for r in prose_rects)
    right_x, right_width = choose_note_area(page, prose_rects, planned_left)

    bottom_start = max(bottom_edge + 20, height * .71)
    bottom_end = height - 14
    # A JSTOR footer / page number is still printed content, not blank paper.
    try:
        options = {'textpage': ocr_page} if ocr_page is not None else {}
        footer_words = page.get_text('words', **options)
        footer_ys = [float(w[1]) for w in footer_words
                     if w[1] > bottom_edge + 13 and
                     w[1] > bottom_start and len(str(w[4]).strip()) >= 2]
        if len(footer_ys) >= 2:
            bottom_end = min(bottom_end, min(footer_ys) - 11)
    except Exception:
        pass
    has_bottom = (bottom_end - bottom_start >= 100 and len(notes) >= 2)
    # Reserve the bottom zone before placing other handwritten notes.
    side_bottom = bottom_start - 16 if has_bottom else height - 16
    left_x0 = 12.0
    left_x1 = min(left_edge - 12, width * .35)
    has_left = left_x1 - left_x0 >= 92
    zones = {
        'right': (right_x, max(right_x, width - NOTE_LAYOUT_PADDING), 24.0, side_bottom),
    }
    if has_left:
        zones['left'] = (left_x0, left_x1, 46.0, side_bottom)
    if has_bottom:
        zones['bottom'] = (16.0, width - 16, bottom_start, bottom_end)

    planned = {}
    last_index = len(notes) - 1 if has_bottom else -1
    side_count = 0
    for i, note in enumerate(notes):
        if i == last_index:
            name = 'bottom'
        elif has_left and side_count % 2 == 1:
            name = 'left'
            side_count += 1
        else:
            name = 'right'
            side_count += 1
        planned[i] = {'zone': name, 'bounds': zones[name]}
    return planned


def place_note_positions(anchored, page_height, top=16.0, bottom=None):
    """Give large handwriting space, without overlapping adjacent margin notes.

    Reserve room from the model's actual pen-stroke height rather than using
    an arbitrary fixed height per line. If all four notes cannot fit at full
    scale, shrink their reserved areas proportionally instead of clipping.
    """
    bottom = page_height - 16.0 if bottom is None else bottom
    gap = 17.0
    heights = [
        min(260.0, max(120.0, 18.0 + float(item[3].get("height", 60.0)) * (NOTE_TARGET_SCALE + NOTE_SCALE_JITTER)))
        for item in anchored
    ]
    capacity = max(1.0, bottom - top - gap * max(0, len(heights) - 1))
    if sum(heights) > capacity:
        ratio = capacity / sum(heights)
        heights = [h * ratio for h in heights]

    positions = []
    for i, (anchor, _, _, _) in enumerate(anchored):
        y = max(top, min(anchor, bottom - heights[i]))
        if positions:
            y = max(y, positions[-1] + heights[i - 1] + gap)
        positions.append(y)

    if positions and positions[-1] + heights[-1] > bottom:
        positions[-1] = bottom - heights[-1]
        for i in range(len(positions) - 2, -1, -1):
            positions[i] = min(positions[i], positions[i + 1] - gap - heights[i])
    return positions, heights


def draw_handheld_marker(page, quad, color, seed, over_image=True):
    """Render a wandering marker swipe, with uneven edges and doubled ends.

    Draw BEFORE the printed text so the text stays crisp and black.
    This is appearance-only PDF artwork (not a selectable PDF highlight widget).
    """
    rect = quad.rect if hasattr(quad, "rect") else fitz.Rect(quad)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return

    rng = random.Random(seed)
    opacity_scale = 0.68 if over_image else 1.0
    h = rect.height
    x0 = rect.x0 - rng.uniform(0.7, 2.7)
    x1 = rect.x1 + rng.uniform(0.8, 4.7)
    width = x1 - x0
    ymid = (rect.y0 + rect.y1) / 2 + rng.uniform(-0.3, 0.5)
    thickness = h * rng.uniform(0.77, 0.93)
    ytop, ybottom = ymid - thickness / 2, ymid + thickness / 2
    count = max(7, min(34, round(width / 12)))
    phase = rng.uniform(0, 2 * math.pi)
    drift = rng.uniform(-2.8, 2.8)

    top_edge, bottom_edge = [], []
    for i in range(count + 1):
        t = i / count
        x = x0 + width * t
        # The upper and lower edges respond to the same hand movement
        # but have slightly different texture as the felt tip drags.
        swing = MARKER_WOBBLE_PT * (
            0.67 * math.sin(2 * math.pi * t + phase)
            + 0.24 * math.sin(5 * math.pi * t + phase * 0.7)
        )
        y_base = drift * t + swing
        xx_upper = x + (rng.uniform(-1.5, 1.1) if i in (0, count) else 0)
        xx_lower = x + (rng.uniform(-2.0, 1.8) if i in (0, count) else 0)
        top_edge.append(fitz.Point(
            xx_upper, ytop + y_base + rng.uniform(-0.42, 0.42)))
        bottom_edge.append(fitz.Point(
            xx_lower, ybottom + y_base + rng.uniform(-0.46, 0.46)))

    page.draw_polyline(
        top_edge + list(reversed(bottom_edge)),
        color=None, fill=color, fill_opacity=MARKER_OPACITY * opacity_scale,
        closePath=True, overlay=over_image,
    )

    # Real felt-tip highlighting tends to concentrate pigment at the edges.
    # Make subtle irregular INSIDE bands, not a hard rectangular outline.
    # This keeps the center light and the original printed text legible.
    upper_inner = []
    lower_inner = []
    for i, (top_pt, bottom_pt) in enumerate(zip(top_edge, bottom_edge)):
        t = i / count
        waviness = 0.16 * math.sin(8 * math.pi * t + phase)
        top_depth = MARKER_EDGE_THICKNESS_PT * rng.uniform(0.65, 1.35) + waviness
        bottom_depth = MARKER_EDGE_THICKNESS_PT * rng.uniform(0.65, 1.35) - waviness
        upper_inner.append(fitz.Point(top_pt.x, top_pt.y + top_depth))
        lower_inner.append(fitz.Point(bottom_pt.x, bottom_pt.y - bottom_depth))

    page.draw_polyline(
        top_edge + list(reversed(upper_inner)),
        color=None, fill=color,
        fill_opacity=MARKER_EDGE_OPACITY * opacity_scale,
        closePath=True, overlay=over_image,
    )
    page.draw_polyline(
        bottom_edge + list(reversed(lower_inner)),
        color=None, fill=color,
        fill_opacity=MARKER_EDGE_OPACITY * 0.92 * opacity_scale,
        closePath=True, overlay=over_image,
    )

    # Small dark end caps where a real marker gets pressed down or lifted.
    # Deliberately vary how heavy each end is.
    for is_start in (True, False):
        strip_w = min(width * rng.uniform(0.022, 0.045), 5.0)
        if is_start:
            x_a, x_b = x0, x0 + strip_w
        else:
            x_a, x_b = x1 - strip_w, x1
        page.draw_polyline(
            [
                fitz.Point(x_a + rng.uniform(-0.4, 0.3), ytop + rng.uniform(0, 0.6)),
                fitz.Point(x_b + rng.uniform(-0.3, 0.5), ytop + rng.uniform(0, 0.7)),
                fitz.Point(x_b + rng.uniform(-0.4, 0.5), ybottom - rng.uniform(0, 0.7)),
                fitz.Point(x_a + rng.uniform(-0.3, 0.4), ybottom - rng.uniform(0, 0.6)),
            ],
            color=None, fill=color,
            fill_opacity=MARKER_TIP_OPACITY * rng.uniform(0.75, 1) * opacity_scale,
            closePath=True, overlay=over_image,
        )

    # Extra patches simulate the tip pressing down at the start and lifting
    # later, sometimes re-tracing the final few millimeters.
    for start_end in (True, False):
        if rng.random() < 0.07:
            continue
        span = min(width * rng.uniform(0.055, 0.14), 19)
        xa, xb = (x0, x0 + span) if start_end else (x1 - span, x1)
        center_swell = rng.uniform(-0.5, 0.5)
        corners = [
            fitz.Point(xa + rng.uniform(-2.2, 0.8), ytop + rng.uniform(-0.8, 0.5)),
            fitz.Point(xb + rng.uniform(-0.6, 1.1), ytop + center_swell),
            fitz.Point(xb + rng.uniform(-0.4, 1.0), ybottom + rng.uniform(-0.2, 0.8)),
            fitz.Point(xa + rng.uniform(-2.2, 1.2), ybottom + rng.uniform(-0.9, 0.9)),
        ]
        page.draw_polyline(
            corners, color=None, fill=color,
            fill_opacity=MARKER_END_OVERLAP * rng.uniform(0.70, 1.0) * opacity_scale,
            closePath=True, overlay=over_image,
        )




def render_pdf(doc, annotations, reading, strokes, ocr_textpages=None, note_positions=None, note_zones=None):
    ocr_textpages = ocr_textpages or {}
    note_positions = note_positions or {}
    note_zones = note_zones or {}
    rects_by_page = {page: rects for page, _, rects in reading}
    num_highlights = num_notes = unmatched = 0
    for data in annotations:
        previous_highlights, previous_notes = num_highlights, num_notes
        bound_ocr = ocr_textpages.get(data.page_number)
        page, ocr_page = bound_ocr if bound_ocr else (doc[data.page_number - 1], None)
        prose_rects = rects_by_page.get(data.page_number, [])
        source_ink_mask = make_source_ink_mask(page)  # before drawing highlights
        occupied_note_rects = []
        # Original pages were copied at natural scale (or gently reduced) onto portrait paper with
        # show_pdf_page() BEFORE rendering highlights. This makes the source
        # artwork an opaque Form XObject, even if the source was vector text.
        # Drawing with overlay=False places marker pigment BEHIND the whole
        # page -- completely invisible. After shrinking the source into the
        # left region, image-area heuristics can no longer detect this.
        # ALWAYS draw translucent marker strokes ABOVE the already-formatted
        # page, including automatic sparse-fill strokes.
        highlight_over_image = True
        seen = set()
        already_marked_lines = set()
        for item in data.highlights:
            quote = item.quote.strip()
            if not quote or quote.casefold() in seen:
                continue
            seen.add(quote.casefold())
            if item.line_number:
                # Do not fall back to a different paragraph if Gemini's line
                # index is wrong; a misplaced highlight is worse than none.
                hits = find_at_numbered_line(
                    page, quote, prose_rects, item.line_number, textpage=ocr_page
                )
            else:
                # Old notes have no numbered line. Keep a conservative exact
                # match route for compatibility, not loose whole-page guessing.
                opts = {'textpage': ocr_page} if ocr_page is not None else {}
                hits = [hit for hit in page.search_for(quote, quads=True, **opts)
                        if any(rect_area(hit.rect & r) >= 0.6 * rect_area(hit.rect)
                               for r in prose_rects)]
            if not hits:
                unmatched += 1
                if not work_setting("hosted"):
                    print(f'Page {data.page_number}: skipped unverified highlight '
                          f'at line {item.line_number}: {quote!r}', flush=True)
                continue
            chosen = [hits[0]]
            for next_quad in hits[1:]:
                rect0 = chosen[-1].rect if hasattr(chosen[-1], 'rect') else fitz.Rect(chosen[-1])
                rect1 = next_quad.rect if hasattr(next_quad, 'rect') else fitz.Rect(next_quad)
                if 0 < rect1.y0 - rect0.y0 < 21:
                    chosen.append(next_quad)
                else:
                    break
            for k, line_quad in enumerate(chosen):
                seed = int.from_bytes(hashlib.sha256(
                    f"{data.page_number}|{quote}|{k}".encode("utf-8")
                ).digest()[:8], "big")
                draw_handheld_marker(
                    page, line_quad, HIGHLIGHT_COLORS[item.color], seed,
                    over_image=highlight_over_image,
                )
            num_highlights += 1
            # Track actual line position, which may differ from model's index.
            best_center = chosen[0].rect.y0 if hasattr(chosen[0], 'rect') else fitz.Rect(chosen[0]).y0
            nearest = min(range(len(prose_rects)),
                          key=lambda k: abs(prose_rects[k].y0 - best_center)) if prose_rects else None
            if nearest is not None:
                already_marked_lines.add(nearest)

        # A page with readable paragraphs must not silently end up with
        # only margin notes because OCR dropped Gemini's highlight quotes.
        num_highlights += fill_sparse_highlights(
            page, prose_rects, already_marked_lines, data.page_number,
            textpage=ocr_page, over_image=highlight_over_image,
        )

        anchored = []
        for i, note in enumerate(data.margin_notes):
            note_id = f"{data.page_number}_{i}"
            writing = strokes.get(note_id)
            if not writing or not writing.get("strokes"):
                continue
            if note.line_number and 1 <= note.line_number <= len(prose_rects):
                # Numbered source line is reliable even with OCR typos.
                anchor_y = prose_rects[note.line_number - 1].y0
            else:
                found = find_in_prose(page, note.anchor_quote, prose_rects, textpage=ocr_page)
                if not found:
                    unmatched += 1
                    if ocr_page is None or not prose_rects:
                        continue
                    ordered = sorted(prose_rects, key=lambda r: (r.y0, r.x0))
                    relative = (i + 0.5) / max(1, len(data.margin_notes))
                    fallback_index = min(len(ordered) - 1, int(relative * len(ordered)))
                    anchor_y = ordered[fallback_index].y0
                else:
                    anchor_y = found[0].y0
            anchored.append((anchor_y, note.note_text, note_id, writing))
        if not anchored:
            continue

        # Distribute comments across the RIGHT margin, LEFT margin and
        # unused space BELOW the printed text, rather than squeezing all of
        # them into one very tall column.
        plan = (note_zones or {}).get(data.page_number) or plan_note_zones(
            page, data.margin_notes, prose_rects,
            planned_left=note_positions.get(data.page_number), ocr_page=ocr_page,
        )
        # Test the actual scanned/rendered page BEFORE inserting any note.
        # Zones that look blank based on prose geometry can contain photograph
        # borders, vertical black labels, or a press release's enclosing box.
        candidates = get_blank_note_zones(
            page, prose_rects, planned_left=note_positions.get(data.page_number),
        )
        for anchor, note_text, note_id, writing in sorted(anchored, key=lambda row: row[0]):
            try:
                ordinal = int(note_id.rsplit('_', 1)[1])
            except (ValueError, IndexError):
                continue
            preferred = plan.get(ordinal, {}).get('zone', 'right')
            order = [preferred] + [name for name in ('right', 'left', 'bottom') if name != preferred]
            desired_height = min(
                250.0,
                max(85.0, 15.0 + float(writing.get('height', 60)) * NOTE_TARGET_SCALE),
            )
            safe_rect = None
            for name in order:
                bounds = candidates.get(name)
                if bounds is None:
                    continue
                found = find_ink_free_note_rect(
                    source_ink_mask, bounds, desired_height, anchor,
                    occupied_note_rects, name,
                )
                if found is not None:
                    safe_rect = found
                    break
            if safe_rect is None:
                # Never lay ink over original content simply to meet note count.
                if not work_setting("hosted"):
                    print(f'Page {data.page_number}: skipped note; no truly blank space '
                          f'for {note_text[:45]!r}', flush=True)
                continue
            seed = int.from_bytes(hashlib.sha256(note_id.encode()).digest()[:4], 'big')
            png = strokes_to_black_png(
                writing, safe_rect.width, safe_rect.height, seed,
            )
            if png:
                page.insert_image(
                    safe_rect, stream=png, overlay=True, keep_proportion=False,
                )
                occupied_note_rects.append(safe_rect)
                num_notes += 1
        print(f"Page {data.page_number}: placed "
              f"{num_highlights - previous_highlights}/{len(data.highlights)} highlights, "
              f"{num_notes - previous_notes}/{len(data.margin_notes)} notes", flush=True)
    return num_highlights, num_notes, unmatched


# ------------------------------------------------------------
# BACKGROUND PROCESSOR
# ------------------------------------------------------------

def process_pdf(job_id, pdf_bytes, api_key, force_regenerate=False):
    if not processing_lock.acquire(blocking=False):
        update_job(job_id, status="failed", message="Another PDF is processing. Please retry after it finishes.")
        return
    try:
        update_job(job_id, message="Checking Handwriter engine and model...")
        node, bridge = ensure_handwriter()  # Fail early, BEFORE using Gemini quota.
        with fitz.open(stream=pdf_bytes, filetype="pdf") as original_doc:
            if original_doc.needs_pass:
                raise ValueError("Password-protected PDFs are unsupported")
            update_job(job_id, message="Preserving original page sizes and measuring real handwriting margins...")
            letter_doc, note_positions = fit_on_letter_portrait(original_doc)
            try:
                # IMPORTANT: all later work happens on the formatted pages.
                # OCR / extraction / highlights / notes all use the final paper size,
                # and the scan finish is applied last over the complete finished page.
                reading, notes_start, ocr_textpages, ocr_count = collect_prose_pages(letter_doc, job_id)
                if not reading:
                    raise ValueError(
                        "No readable prose found, even after OCR. The PDF may contain only images, "
                        "technical drawings, or scans too faint for OCR."
                    )
                pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
                annotated = asyncio.run(generate_annotations(
                    api_key, job_id, pdf_hash, reading, notes_start, len(letter_doc),
                    force_regenerate=force_regenerate,
                ))
                update_job(job_id, message="Handwriter is generating pen strokes...")
                tasks = []
                note_zones = {}
                prose_by_page = {n: rects for n, _, rects in reading}
                for item in annotated:
                    page = letter_doc[item.page_number - 1]
                    ocr_ref = ocr_textpages.get(item.page_number)
                    ocr_page = ocr_ref[1] if ocr_ref else None
                    plan = plan_note_zones(
                        page, item.margin_notes, prose_by_page.get(item.page_number, []),
                        planned_left=note_positions.get(item.page_number), ocr_page=ocr_page,
                    )
                    note_zones[item.page_number] = plan
                    for i, note in enumerate(item.margin_notes):
                        slot = plan.get(i)
                        if not slot:
                            continue
                        x0, x1, _, _ = slot['bounds']
                        wrap_chars = estimated_wrap_chars(x1 - x0)
                        note_id = f"{item.page_number}_{i}"
                        seed = int.from_bytes(hashlib.sha256(
                            (pdf_hash + "|" + note_id + "|" + note.note_text).encode()
                        ).digest()[:4], "big")
                        tasks.append({
                            "id": note_id,
                            "text": note.note_text,
                            "seed": seed,
                            "wrap_chars": wrap_chars,
                        })
                neural_strokes = generate_neural_strokes(tasks, node, bridge)
                update_job(job_id, message="Placing highlights and black handwriting across available margins...")
                h, n, missing = render_pdf(
                    letter_doc, annotated, reading, neural_strokes,
                    ocr_textpages=ocr_textpages, note_positions=note_positions,
                    note_zones=note_zones,
                )
                if h + n == 0:
                    raise RuntimeError(
                        "No annotations could be placed on this PDF. OCR may have found text, "
                        "but no exact quote anchors matched. Check the scan quality."
                    )
                if SCAN_FINISH:
                    update_job(job_id, message="Applying subtle full-page scanned finish last...")
                    final = apply_scan_finish(letter_doc)
                    try:
                        output = final.tobytes(garbage=4, deflate=True)
                    finally:
                        final.close()
                else:
                    output = letter_doc.tobytes(garbage=4, deflate=True)
            finally:
                letter_doc.close()
            # Fail loudly if any future change accidentally exports landscape.
            with fitz.open(stream=output, filetype="pdf") as checked:
                if any(abs(p.rect.width - 612) > 0.1 or abs(p.rect.height - 792) > 0.1
                       for p in checked):
                    raise RuntimeError("Export size error: expected 8.5 x 11 in portrait on every page")
        with jobs_lock:
            jobs[job_id].update(
                status="complete", pdf_bytes=output,
                message=f"Done (ink-safe margins v18): {h} highlights, {n} handwritten notes, {missing} unmatched quotes; OCR checked {ocr_count} scanned pages"
            )
    except Exception as exc:
        if work_setting("hosted"):
            # Upstream exceptions can contain credentials, document excerpts, or URLs.
            # Preserve only a bounded category for the browser and application logs.
            category = "provider" if any(word in str(exc).lower() for word in ("gemini", "api", "quota", "failed", "timeout", "timed out")) else "processing"
            print(f"Annotation failed: {category} ({type(exc).__name__})", flush=True)
            update_job(job_id, status="failed", message=("The AI request could not be completed. Check your Gemini key, model access, and quota." if category == "provider" else "This PDF could not be annotated. Try a text-based reading with clear margins."), error_category=category)
        else:
            traceback.print_exc()
            update_job(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")
    finally:
        processing_lock.release()


# ------------------------------------------------------------
# BROWSER UI
# ------------------------------------------------------------




@app.route("/")
def index():
    response = app.make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


# Hosted requests return the finished PDF in the same invocation. No hosted
# request depends on a background thread or a later invocation's memory.
@app.route("/api/config")
def api_config():
    response = jsonify(
        hosted=HOSTED, max_upload_bytes=HOSTED_MAX_UPLOAD,
        max_pages=HOSTED_MAX_PAGES, max_output_bytes=HOSTED_MAX_OUTPUT,
        requires_api_key=True, demo_available=True,
        model=MODEL_NAME, page_size="US Letter portrait", scanned_pdfs=False,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def validate_hosted_pdf(pdf_bytes):
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
        return "Select a valid PDF file", "invalid_pdf", 400
    if len(pdf_bytes) > HOSTED_MAX_UPLOAD:
        return f"The browser app accepts PDFs up to {HOSTED_MAX_UPLOAD // (1024 * 1024)} MiB", "file_too_large", 413
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                return "Password-protected PDFs are unsupported", "encrypted_pdf", 400
            if not 1 <= len(doc) <= HOSTED_MAX_PAGES:
                return f"The browser app accepts readings of 1 to {HOSTED_MAX_PAGES} pages", "page_limit", 400
            if any(len(page.get_text("text").strip()) < 180 or native_layer_has_broken_word_spacing(page.get_text("text")) for page in doc):
                return "Use a text-based PDF with selectable prose. Scanned PDFs need the local OCR workflow", "text_pdf_required", 400
    except Exception:
        return "This file could not be read as a PDF", "invalid_pdf", 400
    return None


def sample_pdf():
    """An original public sample, with no personal reading or cached annotations."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((42, 49), "Auto Annotator: sample reading", fontsize=18)
    page.insert_text((42, 68), "DEMO - preset annotations, not an AI-generated result", fontsize=9, color=(0.4, 0.2, 0.5))
    paragraphs = [
        "Learning works best when readers pause to ask questions about a text. A useful note connects a claim to evidence rather than copying every sentence. Choosing one important idea can make a difficult reading easier to discuss.",
        "Annotations also help readers notice uncertainty. When an author makes a broad claim, the reader can ask what information would support it. A short question in the margin can guide the next conversation or experiment.",
        "Good study tools should keep the original document readable. Highlights need to stay translucent, and notes need enough empty space. A printable page is useful when it supports reflection without hiding the source material.",
    ]
    top = 102
    for paragraph in paragraphs:
        page.insert_textbox(fitz.Rect(42, top, 405, top + 155), paragraph, fontsize=11, lineheight=1.5)
        top += 165
    doc.set_metadata({"title": "Auto Annotator demonstration", "author": "Auto Annotator", "subject": "Original sample with preset demo annotations; no AI request"})
    output = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return output


def direct_annotation(pdf_bytes, api_key, filename, *, demo=False, selected_model=None):
    problem = validate_hosted_pdf(pdf_bytes)
    if problem:
        message, code, status = problem
        return jsonify(error=message, code=code), status
    job_id = uuid.uuid4().hex
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(filename).stem).strip()[:100] or "annotated_reading"
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "done": 0, "total": 0, "skipped": 0,
            "notes_start": None, "message": "Starting...", "pdf_bytes": None,
        }
    try:
        with tempfile.TemporaryDirectory(prefix="autoannotator-") as workspace:
            token = request_work.set({
                "hosted": True, "demo": demo, "model": selected_model or MODEL_NAME,
                "cache_dir": Path(workspace) / "cache",
                "engine_dir": Path(workspace) / "engine",
            })
            try:
                process_pdf(job_id, pdf_bytes, api_key)
            finally:
                request_work.reset(token)
            with jobs_lock:
                state = jobs[job_id]
                output = state.get("pdf_bytes")
                error = state.get("message", "This PDF could not be annotated")
                category = state.get("error_category", "processing")
            if not output:
                if "Another PDF is processing" in error:
                    return jsonify(error="Another reading is being processed. Please try again shortly", code="busy"), 503
                return jsonify(error=error, code=category + "_failed"), 502 if category == "provider" else 422
            if len(output) > HOSTED_MAX_OUTPUT:
                return jsonify(error=f"The annotated PDF exceeds the {HOSTED_MAX_OUTPUT // (1024 * 1024)} MiB browser download limit. Try fewer pages or use the local workflow", code="output_too_large"), 413
            response = send_file(io.BytesIO(output), mimetype="application/pdf", as_attachment=True,
                                 download_name=f"{safe_stem}_Notes.pdf")
            response.headers["Cache-Control"] = "no-store, private"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-AutoAnnotator-Demo"] = "true" if demo else "false"
            return response
    finally:
        with jobs_lock:
            jobs.pop(job_id, None)


@app.route("/api/process", methods=["POST"])
def process_direct():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Select a PDF", code="file_required"), 400
    api_key = request.form.get("api_key", "").strip()
    if not api_key or len(api_key) > 512:
        return jsonify(error="Enter your own Gemini API key", code="api_key_required"), 400
    selected_model = request.form.get("model", MODEL_NAME).strip()
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,100}", selected_model):
        return jsonify(error="Enter a valid Gemini model name", code="invalid_model"), 400
    pdf_bytes = uploaded.read(HOSTED_MAX_UPLOAD + 1)
    return direct_annotation(pdf_bytes, api_key, uploaded.filename, selected_model=selected_model)


@app.route("/api/demo", methods=["POST"])
def process_demo():
    return direct_annotation(sample_pdf(), "", "AutoAnnotator_Demo", demo=True)


@app.route("/api/preview", methods=["POST"])
def preview_pdf():
    """Render the finished PDF's first page without AI or persistent storage."""
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Select an annotated PDF", code="file_required"), 400
    pdf_bytes = uploaded.read(HOSTED_MAX_UPLOAD + 1)
    if len(pdf_bytes) > HOSTED_MAX_UPLOAD:
        return jsonify(error=f"Preview accepts PDFs up to {HOSTED_MAX_UPLOAD // (1024 * 1024)} MiB", code="file_too_large"), 413
    if not pdf_bytes.startswith(b"%PDF-"):
        return jsonify(error="Select a valid PDF", code="invalid_pdf"), 400
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                return jsonify(error="Password-protected PDFs are unsupported", code="encrypted_pdf"), 400
            if not 1 <= len(doc) <= HOSTED_MAX_PAGES:
                return jsonify(error=f"Preview accepts PDFs of 1 to {HOSTED_MAX_PAGES} pages", code="page_limit"), 400
            page = doc[0]
            # Finished Auto Annotator documents use Letter portrait. Bounding the
            # page geometry also bounds the image allocation to 765 x 990 pixels.
            if abs(page.rect.width - 612) > 1 or abs(page.rect.height - 792) > 1:
                return jsonify(error="Preview expects an annotated US Letter portrait PDF", code="page_size"), 400
            image = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), colorspace=fitz.csRGB, alpha=False)
            png = image.tobytes("png")
            if len(png) > 2 * 1024 * 1024:
                return jsonify(error="The first-page preview is too large", code="preview_too_large"), 413
    except Exception:
        return jsonify(error="This PDF preview could not be rendered", code="invalid_pdf"), 400
    response = send_file(io.BytesIO(png), mimetype="image/png", download_name="annotation-preview.png")
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/api/version")
def app_version():
    return jsonify(
        version=15,
        page_size="8.5 x 11 in portrait",
        layout="preserve original-size page; use existing margins; shrink only if needed",
        highlights="overlay visible",
    )


@app.errorhandler(RequestEntityTooLarge)
def too_large(exc):
    limit = "4 MiB" if HOSTED else "500 MB"
    return jsonify(error=f"PDF exceeds the {limit} upload limit", code="file_too_large"), 413


@app.route("/api/annotate", methods=["POST"])
def start_annotation():
    if HOSTED:
        return jsonify(error="Use the direct PDF processing endpoint for hosted requests", code="use_process_endpoint"), 409
    uploaded = request.files.get("file")
    api_key = request.form.get("api_key", "").strip() or DEFAULT_API_KEY
    if not api_key:
        return jsonify(error="Enter your Gemini API key or set GEMINI_API_KEY locally", code="api_key_required"), 400
    if not uploaded or not uploaded.filename:
        return jsonify(error="Select a PDF"), 400
    pdf_bytes = uploaded.read()
    if not pdf_bytes.startswith(b"%PDF-"):
        return jsonify(error="The uploaded file is not a PDF"), 400
    original_stem = Path(uploaded.filename).stem.strip() or "annotated_reading"
    safe_stem = re.sub(r"[^A-Za-z0-9._ -]+", "", original_stem).strip() or "annotated_reading"
    download_name = f"{safe_stem}_Notes.pdf"
    job_id = uuid.uuid4().hex
    force_regenerate = request.form.get("mode", "resume") == "regenerate"
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "done": 0, "total": 0, "skipped": 0,
            "notes_start": None, "message": "Starting...", "pdf_bytes": None,
            "download_name": download_name,
        }
    threading.Thread(
        target=process_pdf, args=(job_id, pdf_bytes, api_key, force_regenerate), daemon=True
    ).start()
    return jsonify(job_id=job_id), 202


@app.route("/api/status/<job_id>")
def api_status(job_id):
    if HOSTED:
        return jsonify(error="Unknown job"), 404
    state = status_for(job_id)
    return jsonify(state) if state else (jsonify(error="Unknown job"), 404)


@app.route("/api/download/<job_id>")
def api_download(job_id):
    if HOSTED:
        return jsonify(error="Unknown job"), 404
    with jobs_lock:
        job = jobs.get(job_id)
        pdf_bytes = job.get("pdf_bytes") if job else None
        download_name = job.get("download_name", "annotated_reading.pdf") if job else "annotated_reading.pdf"
    if pdf_bytes is None:
        return jsonify(error="PDF is not ready"), 404
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
        download_name=download_name
    )


if __name__ == "__main__":
    print("AUTOANNOTATOR v14: PORTRAIT 8.5 x 11 -- visible highlighter overlay", flush=True)
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
