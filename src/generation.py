# LLM generation with baseline and enhanced prompt templates

"""
Generation module — prompt formatting and OpenAI chat completion calls.

Provides two generation modes:
    1. generate_baseline() — vanilla prompt, simple context formatting
    2. generate_enhanced() — expert prompt with citation instructions and
                             source-attributed context blocks
"""

import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import GENERATION_MODEL, OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """\
Answer the following question based on the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

ENHANCED_SYSTEM_PROMPT = """\
You are an expert UK financial regulation assistant specialising in the FCA Handbook.

Each source in the user's context is labelled with its chunk ID, sourcebook, section, \
and rule type:
- (R) = Rule — a legally binding obligation ("must")
- (G) = Guidance — the FCA's expectation but not mandatory ("should")
- (D) = Direction / (E) = Evidential provision — treat as binding unless noted otherwise

Follow these rules strictly:
1. Answer using ONLY the provided context. Do not introduce outside knowledge.
2. Start with a direct answer, then support it with specific rule citations in the \
format SOURCEBOOK X.Y.ZR/G (e.g. COBS 4.2.1R, PRIN 2.1.1R, SYSC 10.1.8G).
3. For each citation, state whether it is a Rule (R) or Guidance (G) and what this \
means for the firm's obligation (mandatory vs. expected).
4. If multiple rules from the same section appear, distinguish their different \
requirements — do not conflate them.
5. Preserve precise regulatory language: "must" (Rules) is not interchangeable with \
"should" (Guidance).
6. If the context partially answers the question, provide what you can and explicitly \
list what information is missing.
7. Only refuse to answer if NO retrieved source addresses the question at all. If any \
source partially relates to the question, extract and present whatever relevant \
information you can find. Err on the side of answering rather than refusing."""

ENHANCED_USER_PROMPT = """\
Retrieved regulatory context (ordered by relevance):
---
{context_with_sources}
---

Question: {question}

Think step-by-step, then provide a clear, grounded answer with rule references:"""

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_context(chunks: list[dict]) -> str:
    """Concatenate retrieved chunk texts separated by a divider."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("chunk_id", f"chunk_{i}")
        parts.append(f"[{source}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _format_context_with_sources(chunks: list[dict]) -> str:
    """Format chunks with source attribution for the enhanced prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        meta = chunk.get("metadata", {})
        sourcebook = meta.get("sourcebook", "Unknown")
        section = meta.get("section", "")
        rule_type = meta.get("rule_type", "")

        header = f"[Source {i}: {chunk_id} | {sourcebook}"
        if section:
            header += f" {section}"
        if rule_type:
            header += f" ({rule_type})"
        header += "]"

        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------


def generate_no_rag(question: str) -> str:
    """
    Ask GPT-4o the question directly with no retrieved context.

    This serves as the true baseline — the LLM answering from its
    training data alone, with no RAG pipeline involved.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def generate_baseline(question: str, chunks: list[dict]) -> str:
    """
    Build the baseline prompt from retrieved chunks and call GPT-4o.

    Returns the assistant's answer as a string.
    """
    context = _format_context(chunks)
    prompt = BASELINE_PROMPT.format(context=context, question=question)

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def generate_enhanced(question: str, chunks: list[dict]) -> str:
    """
    Build the enhanced prompt with source-attributed context and citation
    instructions, then call GPT-4o.

    Returns the assistant's answer as a string.
    """
    context = _format_context_with_sources(chunks)
    user_msg = ENHANCED_USER_PROMPT.format(
        context_with_sources=context, question=question
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()
