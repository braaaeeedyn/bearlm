"""Token-based recursive text splitter (local backend).

Phase 2 of the local migration: swap the old word-count chunker for a
`RecursiveCharacterTextSplitter` whose length function counts *tokens* (via
tiktoken) rather than characters. This keeps chunk sizes aligned with the model's
real context budget and splits on natural boundaries (paragraphs, then lines,
then sentences, then words) instead of a blind word window.

The old word-based `doc_processor.chunk_text` is left intact for the OpenAI
backend so the two ingest paths can be compared.
"""

from __future__ import annotations

import functools

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


@functools.lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def count_tokens(text: str) -> int:
    """Token count used as the splitter's length function."""
    return len(_encoder().encode(text))


@functools.lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    """A token-length RecursiveCharacterTextSplitter using the shared chunk knobs.

    chunk_size / chunk_overlap are interpreted in *tokens* because
    `length_function=count_tokens`.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_text(text: str) -> list[str]:
    """Split one long string into token-bounded chunks."""
    return [c for c in get_splitter().split_text(text) if c.strip()]
