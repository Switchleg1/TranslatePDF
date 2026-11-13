"""Cut -> translate -> reconstruct a PDF (German -> English).

Phases (all resumable; translations cached in work/cache.json):
  1. cut:         split input into chunks of N pages  (work/chunks/)
  2. translate:   extract text blocks per chunk, translate, cache
  3. reconstruct: redact original text, write English text in same boxes
  4. merge:       join chunks into output/<name>_EN.pdf

Usage: python translate_pdf.py [--pages 1-20] [--chunk 100] [--backend google|llama]
"""
import argparse, glob, json, os, re, sys, time, hashlib
import pymupdf, requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "work"); CH = os.path.join(WORK, "chunks"); OUT_CH = os.path.join(WORK, "out")
CACHE_F = os.path.join(WORK, "cache.json")
LLAMA = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8082")  # overridden by --port/--host

def load_cache():
    return json.load(open(CACHE_F, encoding="utf-8")) if os.path.exists(CACHE_F) else {}
def save_cache(c):
    tmp = CACHE_F + ".tmp"; json.dump(c, open(tmp, "w", encoding="utf-8"), ensure_ascii=False); os.replace(tmp, CACHE_F)

def key(t): return hashlib.md5(t.encode()).hexdigest()

# ---------- translators ----------
def tr_google(text):
    from deep_translator import GoogleTranslator
    for a in range(5):
        try: return GoogleTranslator(source="de", target="en").translate(text)
        except Exception: time.sleep(5 * (a + 1))
    return None
SYSTEM = ("You are a professional German-to-English translator for automotive embedded-software documentation "
          "(Continental DQ500 transmission control unit, BIOS/boot loader, CAN, UDS). "
          "Translate the user's text to English. Rules: output ONLY the translation, no notes or quotes; "
          "keep function names, identifiers, signal names, hex values, numbers, units, file names and "
          "abbreviations (ECU, TCU, CAN, UDS, NVM, ...) unchanged; keep the same meaning and roughly the same length; "
          "if the text is already English or is only code/identifiers, return it unchanged.")
def tr_llama(text):
    for a in range(3):
        try:
            r = requests.post(LLAMA + "/v1/chat/completions", timeout=300, json={
                "temperature": 0.1, "top_p": 0.8, "max_tokens": min(4096, len(text) * 3 + 64),
                "chat_template_kwargs": {"enable_thinking": False},   # Qwen3: no reasoning, much faster
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": text}]})
            out = r.json()["choices"][0]["message"]["content"]
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            if out and len(out) < len(text) * 4 + 50: return out
        except Exception: time.sleep(2)
    return None

BATCH_SYSTEM = SYSTEM + (" The input is a list of numbered segments, one per line, formatted as [n] text. "
          "Translate each segment separately and output exactly one line per segment in the same format [n] translation, "
          "same numbers, same order. Never merge, split or skip segments.")
def tr_llama_batch(texts):
    """Translate many segments in one request. Returns list (None where a segment failed)."""
    src = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    res = [None] * len(texts)
    for a in range(2):
        try:
            r = requests.post(LLAMA + "/v1/chat/completions", timeout=900, json={
                "temperature": 0.1, "top_p": 0.8, "max_tokens": min(16384, len(src) * 2 + 256),
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": BATCH_SYSTEM}, {"role": "user", "content": src}]})
            out = r.json()["choices"][0]["message"]["content"] or ""
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
            for m in re.finditer(r"^\s*\[(\d+)\]\s?(.*)$", out, flags=re.M):
                i = int(m.group(1)) - 1
                if 0 <= i < len(texts) and m.group(2).strip() and res[i] is None:
                    t = m.group(2).strip()
                    if len(t) < len(texts[i]) * 4 + 50: res[i] = t
            if all(x is not None for x in res): break
        except Exception: time.sleep(2)
    return res

def _is_code(w):
    if re.match(r"^\.\w", w): return True                               # .member names (index)
    w = w.strip(".,;:()[]{}\"'")
    if not w or not re.search(r"[A-Za-zÄÖÜäöüß]", w): return True          # numbers / punctuation
    return ("_" in w or re.search(r"[a-zäöü][A-ZÄÖÜ]", w) is not None or re.search(r"\w\.\w", w) is not None
            or re.search(r"[A-Za-z]\d|\d[A-Za-z]", w) is not None or "()" in w)
def needs_tr(t):
    if not re.search(r"[A-Za-zÄÖÜäöüß]{2,}", t): return False
    return not all(_is_code(w) for w in t.split())                       # only identifiers -> keep as is

# ---------- blocks ----------
def _is_bullet(t):
    t = t.strip()
    return len(t) == 1 and (not t.isalnum() or t == "o")

def _rules(page):
    """Drawn table lines: returns (vertical [(x,y0,y1)], horizontal [(y,x0,x1)])."""
    vs, hs = [], []
    try:
        for dr in page.get_drawings():
            stroked = "s" in (dr.get("type") or "")
            fill = dr.get("fill")
            visible_fill = fill is not None and min(fill) < 0.95          # ignore white background boxes
            if not stroked and not visible_fill: continue
            for it in dr["items"]:
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    if abs(p1.x - p2.x) < 1 and abs(p1.y - p2.y) > 4: vs.append((p1.x, min(p1.y, p2.y), max(p1.y, p2.y)))
                    elif abs(p1.y - p2.y) < 1 and abs(p1.x - p2.x) > 4: hs.append((p1.y, min(p1.x, p2.x), max(p1.x, p2.x)))
                elif it[0] == "re":
                    r = it[1]
                    if r.width < 2 and r.height > 4 and (stroked or visible_fill): vs.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
                    elif r.height < 2 and r.width > 4: hs.append(((r.y0 + r.y1) / 2, r.x0, r.x1))
                    elif r.height > 4 and r.width > 4 and stroked:
                        vs += [(r.x0, r.y0, r.y1), (r.x1, r.y0, r.y1)]; hs += [(r.y0, r.x0, r.x1), (r.y1, r.x0, r.x1)]
    except Exception: pass
    return vs, hs

def _segments(page, rules=None):
    """Horizontal text pieces: one per run of spans, split at table rules / tab gaps; keeps justified lines whole."""
    segs = []; rules = (rules or _rules(page))[0]
    def ruled(x0, x1, y0, y1):
        return any(x0 - 0.5 <= x <= x1 + 0.5 and ry0 <= y1 and ry1 >= y0 for x, ry0, ry1 in rules)
    # collect all horizontal spans and rebuild visual rows by baseline (PyMuPDF splits justified lines)
    spans = []
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"] != 0: continue
        for l in b["lines"]:
            if abs(l["dir"][0] - 1) > 0.05: continue          # rotated text (side banners): leave untouched
            for x in l["spans"]:
                chars = x["chars"]; text = "".join(c["c"] for c in chars)
                if not text.strip(): continue
                # dot leaders (". . . . ." in TOC/index): cut them out, they and the page number stay untouched
                cuts = [(m.start(), m.end()) for m in re.finditer(r"(?:\s?\.){4,}\s*", text)]
                pos, brk = 0, False
                for a_, b_ in cuts + [(len(text), len(text))]:
                    piece = chars[pos:a_]
                    if "".join(c["c"] for c in piece).strip():
                        r = pymupdf.Rect(piece[0]["bbox"])
                        for c in piece[1:]: r |= pymupdf.Rect(c["bbox"])
                        y = {k: x[k] for k in ("size", "flags", "font", "color", "origin")}
                        y.update(text="".join(c["c"] for c in piece), bbox=tuple(r), origin=(r.x0, x["origin"][1]), brk=brk)
                        spans.append(y)
                    pos, brk = b_, True
    spans.sort(key=lambda x: (x["origin"][1], x["bbox"][0]))
    rows = []
    for x in spans:
        for r in rows:
            if abs(r[0]["origin"][1] - x["origin"][1]) < 0.25 * x["size"]: r.append(x); break
        else: rows.append([x])
    def emit(cur, bullet):
        r = pymupdf.Rect(cur[0]["bbox"])
        for y in cur[1:]: r |= pymupdf.Rect(y["bbox"])
        text = cur[0]["text"]
        for p_, y in zip(cur, cur[1:]):       # re-insert spaces that exist only as gaps
            gap = y["bbox"][0] - p_["bbox"][2]
            sep = " " if gap > 0.15 * y["size"] and not text.endswith(" ") and not y["text"].startswith(" ") else ""
            text += sep + y["text"]
        text = re.sub(r"\s+", " ", text).strip()
        w = sum(len(y["text"]) for y in cur) or 1
        bold = sum(len(y["text"]) for y in cur if y["flags"] & 16 or "bold" in y["font"].lower()) > w / 2
        ital = sum(len(y["text"]) for y in cur if y["flags"] & 2) > w / 2
        segs.append(dict(rect=r, text=text, size=max(y["size"] for y in cur), color=cur[0]["color"],
                         bold=bold, italic=bool(ital), serif=bool(cur[0]["flags"] & 4), bullet=bullet,
                         nojoin=any(y.get("nojoin") for y in cur)))
    for row in rows:
        row = sorted(row, key=lambda x: x["bbox"][0])
        if any(x.get("brk") for x in row): row = [dict(x, nojoin=True) for x in row]
        dots = [x for x in row if re.fullmatch(r"[.\s]+", x["text"])]
        if len(dots) >= 4:                     # leader drawn as separate "." spans: drop them, split there
            keep, brk = [], False
            for x in row:
                if re.fullmatch(r"[.\s]+", x["text"]): brk = True; continue
                if brk: x = dict(x, brk=True); brk = False
                keep.append(x)
            row = [dict(x, nojoin=True) for x in keep]
            if not row: continue
        cells, cur = [], []                    # 1) split at drawn table rules
        for x in row:
            if cur and ruled(cur[-1]["bbox"][2], x["bbox"][0], x["bbox"][1], x["bbox"][3]): cells.append(cur); cur = []
            cur.append(x)
        cells.append(cur)
        for sp in cells:                       # 2) inside a cell: bullets, tab gaps (but keep justified text)
            bullet = False
            if len(sp) > 1 and _is_bullet(sp[0]["text"]): sp = sp[1:]; bullet = True
            gaps = [sp[i + 1]["bbox"][0] - sp[i]["bbox"][2] for i in range(len(sp) - 1)]
            wg = [g for g in gaps if g > 0.1 * sp[0]["size"]]
            mingap = min(wg) if len(wg) >= 2 else 0
            cur = [sp[0]]
            for x, g in zip(sp[1:], gaps):
                sz = cur[-1]["size"]; left = "".join(y["text"] for y in cur).strip()
                label = re.fullmatch(r"[\d.]+|.*:|[A-Z]?\d+(\.\d+)*[a-z]?", left) is not None
                if mingap == 0: big = g > 4 * sz or (g > 1.0 * sz and label)     # single gap: tab after label only
                else: big = g > 1.0 * sz and g > 2.5 * mingap
                if big or x.get("brk") or cur[-1].get("brk_after"):
                    emit(cur, bullet); bullet = False; cur = []
                cur.append(x)
            emit(cur, bullet)
    return segs

def page_blocks(page):
    """Group segments into paragraphs using geometry (adjacent lines, same size, previous line full-width)."""
    vr, hr = _rules(page)
    def hruled(y0, y1, x0, x1): return any(y0 - 1 <= y <= y1 + 1 and rx0 < x1 and rx1 > x0 for y, rx0, rx1 in hr)
    segs = sorted(_segments(page, (vr, hr)), key=lambda g: (round(g["rect"].y0), g["rect"].x0))
    paras = []
    for g in segs:
        best = None
        for p in paras:
            last = p["segs"][-1]; sz = g["size"]
            if g["bullet"] or g["nojoin"] or last["nojoin"] or abs(last["size"] - sz) > 0.8: continue
            if re.match(r"^(\d+|0x[0-9A-Fa-f]+|[a-z])\s*[:)]\s", g["text"]): continue   # enumerated item "0: ...", "a) ..."
            gap = g["rect"].y0 - last["rect"].y1
            if not (-0.3 * sz <= gap <= 0.6 * sz) or g["rect"].y0 < last["rect"].y0 + 0.5 * sz: continue
            if abs(g["rect"].x0 - p["rect"].x0) > 1.5 * sz or g["rect"].x0 > last["rect"].x1: continue
            # previous line ended early (next line's first word would have fit on it) -> paragraph ended
            first = g["text"].split(" ")[0]
            room = max(p["maxx1"], g["rect"].x1) - last["rect"].x1
            if room > pymupdf.get_text_length(first + " ", "helv", sz) * 1.1 + 0.5 * sz: continue
            if g["rect"].x1 > p["maxx1"] + 1.5 * sz: continue
            if last["bold"] != g["bold"]: continue
            if hruled(last["rect"].y1 - 0.5, g["rect"].y0 + 0.5, g["rect"].x0, g["rect"].x1): continue   # table row line
            hyph = last["text"].endswith("-") and g["text"][:1].islower()
            if not hyph and " " not in last["text"]: continue   # previous line is a single token (identifier list)
            best = p
        if best:
            best["segs"].append(g); best["rect"] |= g["rect"]; best["maxx1"] = max(best["maxx1"], g["rect"].x1)
        else:
            paras.append(dict(segs=[g], rect=pymupdf.Rect(g["rect"]), maxx1=g["rect"].x1))
    out = []
    for p in paras:
        sg = p["segs"]
        text = sg[0]["text"]
        for g in sg[1:]:                                   # undo end-of-line hyphenation ("Vor-" + "befuellung")
            if text.endswith("-") and g["text"][:1].islower() and len(text) > 1 and text[-2].isalpha(): text = text[:-1] + g["text"]
            else: text += " " + g["text"]
        n = sum(len(g["text"]) for g in sg) or 1
        out.append(dict(bbox=tuple(p["rect"]), lines=[tuple(g["rect"]) for g in sg], text=text,
                        size=max(g["size"] for g in sg), color=sg[0]["color"],
                        bold=sum(len(g["text"]) for g in sg if g["bold"]) > n / 2,
                        italic=sg[0]["italic"], serif=sg[0]["serif"], nlines=len(sg)))
    return out

def parse_range(s, n):
    if not s: return 0, n
    a, _, b = s.partition("-"); return int(a) - 1, min(int(b or a), n)

def rebuild(f, of):
    cache = load_cache()
    d = pymupdf.open(f)
    for p in d:
        items = [(b, cache[key(b["text"])]) for b in page_blocks(p) if key(b["text"]) in cache]
        for b, _ in items:
            for l in b["lines"]:
                r = pymupdf.Rect(l); p.add_redact_annot(pymupdf.Rect(r.x0 + .3, r.y0 + .5, r.x1 - .3, r.y1 - .5), fill=False)
        p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        for b, t in items:
            rect = pymupdf.Rect(b["bbox"]); size = b["size"]
            if b["nlines"] == 1:      # single line: shrink font so it fits the width instead of wrapping
                w = pymupdf.get_text_length(t, "hebo" if b["bold"] else "helv", size)
                if w > rect.width: size = max(size * rect.width / w, size * 0.45)
                rect.x1 += 1.5 + 0.02 * rect.width; rect.y1 += 1; rect.y0 -= 0.5
            else:
                rect.x1 += 1.5
            tt = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if b["bold"]: tt = f"<b>{tt}</b>"
            if b["italic"]: tt = f"<i>{tt}</i>"
            fam = "sans-serif"
            html = f'<span style="color:#{b["color"]:06x}">{tt}</span>'
            css = f"* {{font-family:{fam}; font-size:{size:.2f}px; line-height:1.12; margin:0; padding:0}}"
            try: p.insert_htmlbox(rect, html, css=css, scale_low=0.4)
            except Exception as ex: print("  box fail", ex)
    d.subset_fonts()                                   # each text box embeds the font: dedupe + subset
    d.save(of + ".tmp", garbage=4, deflate=True); d.close(); os.replace(of + ".tmp", of)
    return os.path.basename(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages"); ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--backend", default="google"); ap.add_argument("--threads", type=int, default=None, help="translation threads (default: 1 for llama/strata, 16 for google)"); ap.add_argument("--procs", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=None, help="llama server port (default 8082)")
    ap.add_argument("-i", "--in", dest="inp", help="input PDF (default: first PDF in input/)")
    ap.add_argument("-o", "--out", dest="out", help="output PDF (default: output/<name>_EN.pdf)")
    ap.add_argument("--batch", type=int, default=40, help="segments per request (llama backend)")
    ap.add_argument("--batch-chars", type=int, default=4000, help="max characters per request")
    a = ap.parse_args()
    if a.threads is None: a.threads = 16 if a.backend == "google" else 1
    global LLAMA, CH, OUT_CH
    if a.port is not None: LLAMA = f"http://{a.host}:{a.port}"
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    src = a.inp or glob.glob(os.path.join(ROOT, "input", "*.pdf"))[0]
    if not os.path.isfile(src) and os.path.isfile(os.path.join(ROOT, "input", src)): src = os.path.join(ROOT, "input", src)
    name = os.path.splitext(os.path.basename(src))[0]
    sub = hashlib.md5(os.path.abspath(src).encode()).hexdigest()[:8]
    CH, OUT_CH = os.path.join(CH, sub), os.path.join(OUT_CH, sub)
    for d_ in (CH, OUT_CH): os.makedirs(d_, exist_ok=True)
    doc = pymupdf.open(src); s, e = parse_range(a.pages, len(doc))
    tr = {"google": tr_google, "llama": tr_llama, "dummy": lambda t: t + " " + t[: len(t) // 5]}[a.backend]  # dummy: layout testing
    cache = load_cache()
    tag = f"{s+1}-{e}"

    # 1. CUT
    chunks = []
    for i in range(s, e, a.chunk):
        j = min(i + a.chunk, e); f = os.path.join(CH, f"c_{i+1:05d}_{j:05d}.pdf")
        if not os.path.exists(f):
            c = pymupdf.open(); c.insert_pdf(doc, from_page=i, to_page=j - 1); c.save(f); c.close()
        chunks.append(f)
    print("cut:", len(chunks), "chunks", flush=True)

    # 2. TRANSLATE (all chunks, threaded pool, cached/resumable)
    pend = [f for f in chunks if not os.path.exists(os.path.join(OUT_CH, os.path.basename(f)))]
    todo = set()
    for f in pend:
        d = pymupdf.open(f)
        for p in d:
            for b in page_blocks(p):
                if needs_tr(b["text"]) and key(b["text"]) not in cache: todo.add(b["text"])
        d.close()
    todo = sorted(todo); print("to translate:", len(todo), "blocks", flush=True)
    def work(batch):
        if a.backend == "llama" and a.batch > 1:
            return list(zip(batch, tr_llama_batch(batch)))
        return [(t, tr(t)) for t in batch]
    for rnd in range(3):                       # failed blocks are retried (up to 3 passes)
        todo = [t for t in todo if key(t) not in cache]
        if not todo: break
        if rnd: print(f"retry pass {rnd}: {len(todo)} blocks", flush=True)
        bs = max(1, a.batch // (4 ** rnd))     # smaller batches on retry
        batches, cur, chars = [], [], 0        # cap by count and by characters
        for t in todo:
            if cur and (len(cur) >= bs or chars + len(t) > a.batch_chars): batches.append(cur); cur, chars = [], 0
            cur.append(t); chars += len(t)
        if cur: batches.append(cur)
        done, t0 = 0, time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            for res in ex.map(work, batches):
                for t, r in res:
                    if r: cache[key(t)] = r
                done += len(res); save_cache(cache)
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el/60:.1f} min, eta {el/done*(len(todo)-done)/60:.0f} min", flush=True)
        save_cache(cache)
    left = [t for t in todo if key(t) not in cache]
    if left: print("WARNING: untranslated blocks left:", len(left), flush=True)
    save_cache(cache)

    # 3. RECONSTRUCT (one process per chunk)
    outs = [os.path.join(OUT_CH, os.path.basename(f)) for f in chunks]
    jobs = [(f, o) for f, o in zip(chunks, outs) if not os.path.exists(o)]
    with ProcessPoolExecutor(a.procs) as ex:
        for name_ in ex.map(rebuild, [j[0] for j in jobs], [j[1] for j in jobs]):
            print("rebuilt", name_, flush=True)

    # 4. MERGE
    final = a.out or os.path.join(ROOT, "output", f"{name}_EN" + ("" if not a.pages else f"_p{tag}") + ".pdf")
    if a.out and not os.path.dirname(a.out): final = os.path.join(ROOT, "output", a.out)  # bare name -> output/
    os.makedirs(os.path.dirname(os.path.abspath(final)), exist_ok=True)
    m = pymupdf.open()
    for of in outs:
        x = pymupdf.open(of); m.insert_pdf(x); x.close()
    m.save(final, garbage=4, deflate=True); print("wrote", final)

if __name__ == "__main__": main()
