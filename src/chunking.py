"""
Recursive chunking with section-awareness and metadata attachment.

Strategy:
- Primary split: by regulatory section boundaries (rule/guidance blocks).
- If a section exceeds CHUNK_MAX_TOKENS: recursively split by paragraph, then sentence.
- Each chunk carries metadata: sourcebook, chapter, section, rule_number, rule_type.

Usage:
    python src/chunking.py
"""

import re
import sys
from pathlib import Path

import tiktoken
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROCESSED_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

# Tokeniser for counting tokens (cl100k_base is used by text-embedding-3-small)
_ENC = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Regex patterns for section detection
# ---------------------------------------------------------------------------

# Matches section headers like "Section : COBS 4.2 Fair, clear and not misleading"
_RE_SECTION_HEADER = re.compile(
    r"^Section\s*:\s*(.+)$", re.MULTILINE
)

# Matches rule numbers like "COBS 4.2.1", "PRIN 2A.1.4", "SYSC 10.1.1"
_RE_RULE_NUMBER = re.compile(
    r"^([A-Z]{2,6}\s+\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]?\w*)\s*$", re.MULTILINE
)

# Matches [RULE:X] tags inserted by preprocessing
_RE_RULE_TYPE_TAG = re.compile(r"\[RULE:([RGDE])\]")

# Page markers from PDF extraction
_RE_PAGE_MARKER = re.compile(r"^\[PAGE \d+\]$", re.MULTILINE)

# Repeated sourcebook headers (e.g. "PRIN\nPRIN Principles for Businesses\nwww...")
_RE_PAGE_HEADER = re.compile(
    r"^[A-Z]{2,6}\n[A-Z]{2,6}\s+.+\nwww\.handbook\.fca\.org\.uk\n.+\d{4}$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(filepath: Path) -> tuple[dict, str]:
    """Return (metadata, body) from a YAML-frontmatter file."""
    content = filepath.read_text(encoding="utf-8")
    if content.startswith("---"):
        end = content.find("\n---\n", 4)
        if end != -1:
            metadata = yaml.safe_load(content[4:end]) or {}
            body = content[end + 5:]
            return metadata, body
    return {}, content


def strip_page_artefacts(text: str) -> str:
    """Remove [PAGE N] markers and repeated page headers."""
    text = _RE_PAGE_MARKER.sub("", text)
    text = _RE_PAGE_HEADER.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _extract_rule_info(text: str) -> tuple[str | None, str | None]:
    """Extract the first rule number and rule type from a text block."""
    rule_match = _RE_RULE_NUMBER.search(text)
    rule_number = rule_match.group(1).strip() if rule_match else None

    # Try [RULE:X] tag first (from preprocessing), then fall back to the
    # trailing letter on the rule number itself (e.g. "COBS 4.2.1R" → "R")
    type_match = _RE_RULE_TYPE_TAG.search(text)
    if type_match:
        rule_type = type_match.group(1)
    elif rule_number and rule_number[-1] in "RGDE":
        rule_type = rule_number[-1]
    else:
        rule_type = None

    return rule_number, rule_type


def _extract_chapter_section(section_title: str) -> tuple[str, str]:
    """
    From a section title like 'COBS 4.2 Fair, clear and not misleading',
    extract chapter='4' and section='4.2'.
    """
    m = re.match(r"[A-Z]{2,6}\s+(\d+[A-Z]?)(?:\.(\d+[A-Z]?))?", section_title)
    if m:
        chapter = m.group(1)
        section = f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1)
        return chapter, section
    return "", ""


def split_into_sections(text: str) -> list[dict]:
    """
    Split processed text into sections using 'Section :' headers.
    Returns list of dicts with 'title', 'text', 'chapter', 'section_num'.
    """
    text = strip_page_artefacts(text)

    # Find all section header positions
    headers = list(_RE_SECTION_HEADER.finditer(text))

    if not headers:
        # No section headers — treat the whole text as one section
        return [{"title": "", "text": text.strip(), "chapter": "", "section_num": ""}]

    sections = []
    for i, match in enumerate(headers):
        title = match.group(1).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        chapter, section_num = _extract_chapter_section(title)
        sections.append({
            "title": title,
            "text": body,
            "chapter": chapter,
            "section_num": section_num,
        })

    return sections


def split_into_rule_blocks(section_text: str) -> list[str]:
    """
    Within a section, split at individual rule number boundaries
    (e.g. 'COBS 4.2.1', 'COBS 4.2.2').
    """
    parts = _RE_RULE_NUMBER.split(section_text)

    # parts alternates: [text_before, rule_num_1, text_after_1, rule_num_2, ...]
    blocks = []
    if parts[0].strip():
        blocks.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        rule_num = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        blocks.append(f"{rule_num}\n{body}".strip())

    return [b for b in blocks if b]


# ---------------------------------------------------------------------------
# Recursive splitting for oversized blocks
# ---------------------------------------------------------------------------

def _split_by_paragraphs(text: str) -> list[str]:
    """Split on double newlines."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_by_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _merge_with_overlap(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Greedily merge text pieces into chunks up to max_tokens,
    with overlap_tokens of trailing context carried to the next chunk.
    """
    chunks = []
    current_pieces: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece)

        if current_tokens + piece_tokens > max_tokens and current_pieces:
            chunks.append("\n\n".join(current_pieces))
            # Build overlap from the tail of current_pieces
            overlap_pieces: list[str] = []
            overlap_count = 0
            for p in reversed(current_pieces):
                t = count_tokens(p)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_pieces.insert(0, p)
                overlap_count += t
            current_pieces = overlap_pieces
            current_tokens = overlap_count

        current_pieces.append(piece)
        current_tokens += piece_tokens

    if current_pieces:
        chunks.append("\n\n".join(current_pieces))

    return chunks


def recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    If text fits in max_tokens, return it as-is.
    Otherwise split by paragraphs; if any paragraph is still too large, split by sentences.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = _split_by_paragraphs(text)

    # Check if any paragraph itself is oversized — split those by sentence
    fine_pieces = []
    for para in paragraphs:
        if count_tokens(para) > max_tokens:
            sentences = _split_by_sentences(para)
            fine_pieces.extend(sentences)
        else:
            fine_pieces.append(para)

    return _merge_with_overlap(fine_pieces, max_tokens, overlap_tokens)


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

MIN_CHUNK_TOKENS = 10  # Skip chunks with fewer tokens than this

# Words that dominate application-table rows (Annex sections listing rule applicability)
_TABLE_FILLER_WORDS = frozenset({
    "rule", "guidance", "not", "applicable", "r", "g", "d", "e",
    "otherwise", "otherwise,", "to", "in", "relation", "activities",
    "insurance", "distribution", ".", ",",
})


def _is_application_table_entry(text: str, section_title: str) -> bool:
    """
    Detect low-quality application-table chunks from Annex sections.

    These chunks contain a rule reference followed by filler like
    'Rule Rule Not applicable Guidance' — useful as a lookup table but
    harmful for RAG retrieval because they match rule codes without
    providing substantive regulatory text.
    """
    if "annex" not in section_title.lower():
        return False
    # Strip rule numbers before checking word content
    stripped = _RE_RULE_NUMBER.sub("", text)
    words = stripped.split()
    if not words:
        return True
    filler_count = sum(1 for w in words if w.lower().strip(".,;:") in _TABLE_FILLER_WORDS)
    return filler_count / len(words) > 0.7


def chunk_file(filepath: Path) -> list[dict]:
    """
    Chunk a single processed file into a list of chunk dicts:
    {
        'chunk_id': str,
        'text': str,
        'metadata': {sourcebook, chapter, section, rule_number, rule_type, title}
    }
    """
    file_meta, body = parse_frontmatter(filepath)
    sourcebook = file_meta.get("sourcebook", filepath.stem)

    sections = split_into_sections(body)
    chunks = []

    for section in sections:
        rule_blocks = split_into_rule_blocks(section["text"])

        for block in rule_blocks:
            sub_chunks = recursive_split(block, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)

            for i, chunk_text in enumerate(sub_chunks):
                if count_tokens(chunk_text) < MIN_CHUNK_TOKENS:
                    continue
                if _is_application_table_entry(chunk_text, section["title"]):
                    continue

                rule_number, rule_type = _extract_rule_info(chunk_text)

                chunk_id = f"{sourcebook}_{len(chunks):04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "sourcebook": sourcebook,
                        "chapter": section["chapter"],
                        "section": section["section_num"],
                        "section_title": section["title"],
                        "rule_number": rule_number or "",
                        "rule_type": rule_type or "",
                        "part": i if len(sub_chunks) > 1 else None,
                    },
                })

    return chunks


def chunk_all() -> list[dict]:
    """Chunk all processed files and return the full list of chunks."""
    processed_files = sorted(DATA_PROCESSED_DIR.glob("*.txt"))
    if not processed_files:
        print(f"No processed files found in {DATA_PROCESSED_DIR}. Run preprocessing.py first.")
        return []

    all_chunks = []
    for filepath in tqdm(processed_files, desc="Chunking"):
        file_chunks = chunk_file(filepath)
        print(f"  {filepath.name}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    if chunks:
        # Print summary stats
        tokens = [count_tokens(c["text"]) for c in chunks]
        print(f"Token stats: min={min(tokens)}, max={max(tokens)}, "
              f"mean={sum(tokens)/len(tokens):.0f}")
