"""
Download FCA Handbook sourcebook PDFs and Consumer Duty materials,
extract text via PyMuPDF, and save to data/raw/ with YAML frontmatter.

Usage:
    python src/data_ingestion.py

Consumer Duty PDFs must be downloaded manually and placed in:
    data/raw/consumer_duty/*.pdf
"""

import re
import sys
from pathlib import Path
from datetime import date

import requests
import fitz  # pymupdf
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW_DIR, FCA_SOURCEBOOKS

SOURCEBOOK_TITLES = {
    "PRIN": "Principles for Businesses",
    "COBS": "Conduct of Business Sourcebook",
    "SYSC": "Senior Management Arrangements, Systems and Controls",
}

CONSUMER_DUTY_DIR = DATA_RAW_DIR / "consumer_duty"
PDF_CACHE_DIR = DATA_RAW_DIR / "pdfs"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def download_pdf(url: str, dest_path: Path) -> Path:
    """
    Stream-download a PDF to dest_path with a progress bar.
    Validates that the response is actually a PDF (checks %PDF magic bytes).
    Raises RuntimeError with diagnostic info if the server returns HTML instead.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120, headers=_HEADERS)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        snippet = resp.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Server returned HTML instead of a PDF for {url}\n"
            f"Content-Type: {content_type}\n"
            f"Response snippet:\n{snippet}\n\n"
            "The URL may be wrong or the site requires a browser session. "
            "Download the PDF manually and place it in data/raw/pdfs/"
        )

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as fh, tqdm(
        desc=dest_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))

    # Validate magic bytes — a real PDF starts with %PDF
    magic = dest_path.read_bytes()[:4]
    if magic != b"%PDF":
        dest_path.unlink()  # remove the bad file so re-runs don't skip it
        raise RuntimeError(
            f"Downloaded file is not a valid PDF (magic bytes: {magic!r}).\n"
            f"URL: {url}\n"
            "Download the PDF manually and place it in data/raw/pdfs/"
        )

    return dest_path


# FCA Handbook PDFs use a two-column layout where rule numbers (e.g.
# "COBS 4.2.1") and their type letters (R/G/D/E) are in separate PDF text
# blocks at different x-coordinates on the same visual line.  Standard text
# extraction loses the association.  The constants below identify the
# type-letter column so we can spatially re-join them.
_TYPE_LETTER_X_MIN = 110      # type letters appear at x ≈ 116
_TYPE_LETTER_X_MAX = 125
_TYPE_LETTER_SIZE = 7.2       # font size (within ± 0.5)
_TYPE_LETTER_SIZE_TOL = 0.5
_Y_MATCH_TOLERANCE = 5.0      # max vertical distance to pair rule + type

# Matches rule headings like "COBS 4.2.1", "PRIN 2A.1.3", "SYSC 4.1.1-AR"
_RE_RULE_HEADING = re.compile(
    r"^(COBS|PRIN|SYSC)\s+"
    r"-?\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]{0,2}"
    r"(-[A-Z]{1,2})?$"
)


def _extract_page_text_with_rule_types(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page, spatially joining rule-heading
    numbers with their separated type letters (R/G/D/E).

    The FCA Handbook PDFs place the type letter in a separate column at
    x ≈ 116.  This function detects those isolated letters, matches each
    to the nearest rule heading on the same visual line (within 5 pt),
    appends the letter to the rule number, and omits the orphaned letter
    from the output.
    """
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    # --- Step 1: identify isolated type-letter blocks and rule headings ---
    type_letters = []       # (block_idx, y_origin, letter)
    rule_headings = []      # (block_idx, line_idx, span_idx, y_origin, text)

    for bi, block in enumerate(blocks):
        if "lines" not in block:
            continue
        for li, line in enumerate(block["lines"]):
            spans = line["spans"]
            full_text = "".join(s["text"] for s in spans).strip()
            if not spans:
                continue
            s0 = spans[0]

            # Isolated type letter in the type-letter column
            if (full_text in ("R", "G", "D", "E")
                    and abs(s0["size"] - _TYPE_LETTER_SIZE) < _TYPE_LETTER_SIZE_TOL
                    and _TYPE_LETTER_X_MIN < s0["origin"][0] < _TYPE_LETTER_X_MAX):
                type_letters.append((bi, s0["origin"][1], full_text))
                continue

            # Bold rule heading at the standard heading font size
            if (_RE_RULE_HEADING.match(full_text)
                    and "Bold" in s0.get("font", "")
                    and abs(s0["size"] - _TYPE_LETTER_SIZE) < _TYPE_LETTER_SIZE_TOL):
                rule_headings.append((bi, li, 0, s0["origin"][1], full_text))

    # --- Step 2: spatial join — pair each type letter to nearest heading ---
    # Map (block_idx, line_idx) -> suffix letter to append
    heading_suffix: dict[tuple[int, int], str] = {}
    # Set of block indices to skip entirely (orphaned type-letter blocks)
    skip_blocks: set[int] = set()

    for tl_bi, tl_y, letter in type_letters:
        best = None
        best_dist = _Y_MATCH_TOLERANCE + 1
        for rh_bi, rh_li, _, rh_y, _ in rule_headings:
            dist = abs(tl_y - rh_y)
            if dist < best_dist:
                best_dist = dist
                best = (rh_bi, rh_li)
        if best is not None and best_dist <= _Y_MATCH_TOLERANCE:
            heading_suffix[best] = letter
        skip_blocks.add(tl_bi)

    # --- Step 3: rebuild page text with suffixes applied ---
    lines_out: list[str] = []
    for bi, block in enumerate(blocks):
        if bi in skip_blocks:
            continue
        if "lines" not in block:
            continue
        for li, line in enumerate(block["lines"]):
            spans = line["spans"]
            line_text = "".join(s["text"] for s in spans)
            # Append type suffix if this heading was matched
            suffix = heading_suffix.get((bi, li))
            if suffix:
                line_text = line_text.rstrip() + suffix
            lines_out.append(line_text)

    return "\n".join(lines_out)


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[dict]]:
    """
    Extract full text and table-of-contents from a PDF.

    Returns:
        text: page-delimited text with [PAGE N] markers
        toc:  list of {level, title, page} dicts from the PDF outline
    """
    doc = fitz.open(str(pdf_path))

    toc = [
        {"level": lvl, "title": title, "page": page}
        for lvl, title, page in doc.get_toc()
    ]

    pages = []
    for page_num, page in enumerate(doc, start=1):
        page_text = _extract_page_text_with_rule_types(page)
        if page_text.strip():
            pages.append(f"[PAGE {page_num}]\n{page_text}")

    doc.close()
    return "\n\n".join(pages), toc


def save_raw_file(text: str, metadata: dict, dest_path: Path) -> None:
    """Write text with YAML frontmatter to dest_path (UTF-8)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
    dest_path.write_text(f"---\n{frontmatter}---\n\n{text}", encoding="utf-8")


def ingest_sourcebooks() -> None:
    """Download each FCA sourcebook PDF and save extracted text to data/raw/."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for sourcebook, url in FCA_SOURCEBOOKS.items():
        txt_path = DATA_RAW_DIR / f"{sourcebook}.txt"
        if txt_path.exists():
            print(f"[skip] {sourcebook} already ingested at {txt_path}")
            continue

        print(f"\n[{sourcebook}] Downloading {url}")
        pdf_path = PDF_CACHE_DIR / f"{sourcebook}.pdf"
        if not pdf_path.exists():
            download_pdf(url, pdf_path)
        else:
            print(f"  Using cached PDF: {pdf_path}")

        print(f"  Extracting text …")
        text, toc = extract_text_from_pdf(pdf_path)

        metadata = {
            "sourcebook": sourcebook,
            "title": SOURCEBOOK_TITLES.get(sourcebook, sourcebook),
            "source_url": url,
            "source_type": "pdf",
            "page_count": text.count("[PAGE "),
            "toc_entries": len(toc),
            "ingested_at": date.today().isoformat(),
        }
        save_raw_file(text, metadata, txt_path)
        print(
            f"  Saved {txt_path.name} "
            f"({len(text):,} chars | {metadata['page_count']} pages | {len(toc)} TOC entries)"
        )


def ingest_consumer_duty_pdfs() -> None:
    """
    Ingest manually downloaded Consumer Duty PDFs.

    Place PDF files in:  data/raw/consumer_duty/
    before running this step.
    """
    if not CONSUMER_DUTY_DIR.exists():
        CONSUMER_DUTY_DIR.mkdir(parents=True)
        print(
            f"\nCreated {CONSUMER_DUTY_DIR}\n"
            "Place Consumer Duty PDFs there and re-run."
        )
        return

    pdfs = sorted(CONSUMER_DUTY_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"\n[Consumer Duty] No PDFs found in {CONSUMER_DUTY_DIR} — skipping.")
        return

    for pdf_path in pdfs:
        stem = pdf_path.stem.replace(" ", "_")
        txt_path = DATA_RAW_DIR / f"consumer_duty_{stem}.txt"
        if txt_path.exists():
            print(f"[skip] {txt_path.name} already exists")
            continue

        print(f"\n[Consumer Duty] Extracting: {pdf_path.name}")
        text, toc = extract_text_from_pdf(pdf_path)
        metadata = {
            "sourcebook": "CONSUMER_DUTY",
            "title": pdf_path.stem,
            "source_file": pdf_path.name,
            "source_type": "pdf",
            "page_count": text.count("[PAGE "),
            "toc_entries": len(toc),
            "ingested_at": date.today().isoformat(),
        }
        save_raw_file(text, metadata, txt_path)
        print(f"  Saved {txt_path.name} ({len(text):,} chars)")


if __name__ == "__main__":
    ingest_sourcebooks()
    ingest_consumer_duty_pdfs()
    print("\nData ingestion complete.")
