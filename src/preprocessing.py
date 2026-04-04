"""
Clean raw FCA text files: normalise unicode/whitespace, strip PDF artefacts,
and tag rule types (R/G/D/E). Output to data/processed/.

Usage:
    python src/preprocessing.py
"""

import re
import sys
import unicodedata
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# FCA PDF footer: "■ Release 52 ● www.handbook.fca.org.uk   Jan 2024"
_RE_RELEASE_LINE = re.compile(
    r"■\s*Release\s+\d+\s*●\s*www\.handbook\.fca\.org\.uk[^\n]*",
    re.IGNORECASE,
)

# Sourcebook page-ref footers: "COBS 4/1", "SYSC 10A/2"
_RE_PAGE_REF = re.compile(r"\b[A-Z]{2,6}\s+\d+[A-Z]?/\d+\b")

# Standalone bare page numbers on their own line
_RE_BARE_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# "FCA Handbook" on its own line (chapter-header echo)
_RE_FCA_HANDBOOK = re.compile(r"^\s*FCA Handbook\s*$", re.MULTILINE | re.IGNORECASE)

# Hyphenated soft-wrap artefacts (word- \n word → word word)
_RE_SOFT_HYPHEN = re.compile(r"(\w)-\n(\w)")

# Rule-type annotation: "1.2.3 R", "2A.1.4 G", "10.1.1 D", "7.3.2 E"
# Captures the rule number and type separately so we can wrap the type.
_RE_RULE_TYPE_INLINE = re.compile(
    r"(\b\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]?\s+)([RGDE])(?=\s|\n|$)"
)

# Bracketed / parenthesised markers at the start of a line: "(R)", "[G]", etc.
_RE_BRACKETED_RULE = re.compile(r"^\s*[\(\[](R|G|D|E)[\)\]]\s*", re.MULTILINE)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def parse_raw_file(filepath: Path) -> tuple[dict, str]:
    """Return (metadata_dict, body_text) parsed from a YAML-frontmatter file."""
    content = filepath.read_text(encoding="utf-8")
    if content.startswith("---"):
        end = content.find("\n---\n", 4)
        if end != -1:
            metadata = yaml.safe_load(content[4:end]) or {}
            body = content[end + 5:]
            return metadata, body
    return {}, content


def normalize_unicode(text: str) -> str:
    """NFKC-normalize and replace common PDF character substitutions."""
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # FCA bullet/circle markers (used in release-line footers)
    text = text.replace("\u25a0", "").replace("\u25cf", "")  # ■ ●
    # Non-breaking space → regular space
    text = text.replace("\u00a0", " ")
    return text


def strip_pdf_artefacts(text: str) -> str:
    """Remove headers, footers, and other PDF extraction noise."""
    text = _RE_RELEASE_LINE.sub("", text)
    text = _RE_PAGE_REF.sub("", text)
    text = _RE_BARE_PAGE_NUM.sub("", text)
    text = _RE_FCA_HANDBOOK.sub("", text)
    text = _RE_SOFT_HYPHEN.sub(r"\1\2", text)
    return text


def tag_rule_types(text: str) -> str:
    """
    Wrap FCA rule-type markers with [RULE:X] tags so downstream components
    can use them as metadata signals.

    Examples:
        "2.1.1 R Some rule text"   → "2.1.1 [RULE:R] Some rule text"
        "(G) Some guidance text"   → "[RULE:G] Some guidance text"
    """
    text = _RE_RULE_TYPE_INLINE.sub(r"\1[RULE:\2]", text)
    text = _RE_BRACKETED_RULE.sub(lambda m: f"[RULE:{m.group(1)}] ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preprocess_file(raw_path: Path, processed_path: Path) -> None:
    """Run the full preprocessing pipeline on a single raw file."""
    metadata, body = parse_raw_file(raw_path)

    body = normalize_unicode(body)
    body = strip_pdf_artefacts(body)
    body = tag_rule_types(body)
    body = normalize_whitespace(body)

    metadata["preprocessed"] = True
    frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
    processed_path.write_text(f"---\n{frontmatter}---\n\n{body}", encoding="utf-8")


def preprocess_all() -> None:
    """Preprocess every .txt file in data/raw/ and write to data/processed/."""
    raw_files = sorted(DATA_RAW_DIR.glob("*.txt"))
    if not raw_files:
        print(f"No raw .txt files found in {DATA_RAW_DIR}. Run data_ingestion.py first.")
        return

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for raw_path in tqdm(raw_files, desc="Preprocessing"):
        processed_path = DATA_PROCESSED_DIR / raw_path.name
        if processed_path.exists():
            print(f"[skip] {raw_path.name}")
            continue
        preprocess_file(raw_path, processed_path)
        size_kb = processed_path.stat().st_size // 1024
        print(f"  {raw_path.name} → {processed_path.name} ({size_kb} KB)")


if __name__ == "__main__":
    preprocess_all()
    print("\nPreprocessing complete.")
