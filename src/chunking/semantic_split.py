"""Sentence-level semantic chunking heuristics (no external embeddings)."""

from __future__ import annotations

import re
from typing import List, Tuple

from src.config.settings import SemanticChunkingParams
from src.enums.sentence_splitter_mode import SentenceSplitterMode


def _word_jaccard(left: str, right: str) -> float:
    """Return Jaccard similarity over whitespace token sets."""
    left_tokens = {token for token in re.split(r"\s+", left.lower()) if token}
    right_tokens = {token for token in re.split(r"\s+", right.lower()) if token}
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def _split_sentences_simple_regex(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences with inclusive-exclusive spans in ``text``."""
    sentences: List[Tuple[str, int, int]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return sentences
        start = index
        while index < length and text[index] not in ".!?\n":
            index += 1
        while index < length and text[index] in ".!?":
            index += 1
        end = index
        segment = text[start:end]
        if segment.strip():
            lead = len(segment) - len(segment.lstrip())
            trail = len(segment) - len(segment.rstrip())
            span_start = start + lead
            span_end = end - trail
            if span_end > span_start:
                sentences.append((text[span_start:span_end], span_start, span_end))
        if index < length and text[index] == "\n":
            index += 1
    return sentences


def _split_long_sentence_spans(  # pylint: disable=too-many-locals
    sentence: str,
    span_start: int,
    max_chars: int,
) -> List[Tuple[str, int, int]]:
    """Word-wrap a single long sentence into spans capped by ``max_chars``."""
    if max_chars < 1:
        return []
    stripped = sentence.strip()
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        lead = sentence.find(stripped[0])
        abs_start = span_start + lead
        abs_end = abs_start + len(stripped)
        return [(stripped, abs_start, abs_end)]
    words = list(re.finditer(r"\S+", sentence))
    chunks: List[Tuple[str, int, int]] = []
    current_words: List[re.Match[str]] = []
    current_len = 0

    def flush_word_chunk() -> None:
        nonlocal current_words, current_len
        first = current_words[0]
        last = current_words[-1]
        piece = sentence[first.start() : last.end()]
        abs_start = span_start + first.start()
        abs_end = span_start + last.end()
        chunks.append((piece.strip(), abs_start, abs_end))
        current_words = []
        current_len = 0

    for match in words:
        word_len = len(match.group(0))
        add_len = word_len if not current_words else current_len + 1 + word_len
        if current_words and add_len > max_chars:
            flush_word_chunk()
            add_len = word_len
        current_words.append(match)
        current_len = add_len
    flush_word_chunk()
    return chunks


def _normalize_sentence_spans(
    sentences: List[Tuple[str, int, int]],
    max_chars: int,
) -> List[Tuple[str, int, int]]:
    """Expand sentences that exceed ``max_chars`` at word boundaries."""
    normalized: List[Tuple[str, int, int]] = []
    for sent_text, span_start, _span_end in sentences:
        normalized.extend(
            _split_long_sentence_spans(sent_text, span_start, max_chars)
        )
    return normalized


def _merge_sentence_spans(
    sentences: List[Tuple[str, int, int]],
    params: SemanticChunkingParams,
) -> List[Tuple[int, int]]:
    """Return list of (chunk_start, chunk_end_exclusive) in original text."""
    if not sentences:
        return []
    chunks: List[Tuple[int, int]] = []
    group: List[Tuple[str, int, int]] = []

    def flush_group() -> None:
        nonlocal group
        chunk_start = group[0][1]
        chunk_end = group[-1][2]
        chunks.append((chunk_start, chunk_end))
        group = []

    for sent_text, span_start, span_end in sentences:
        if not group:
            group.append((sent_text, span_start, span_end))
            continue
        combined_text = " ".join(item[0] for item in group) + " " + sent_text
        current_len = len(combined_text)
        if current_len > params.max_chars:
            flush_group()
            group.append((sent_text, span_start, span_end))
            continue
        prior_text = " ".join(item[0] for item in group)
        prior_len = len(prior_text)
        if prior_len < params.min_chars:
            group.append((sent_text, span_start, span_end))
            continue
        if _word_jaccard(prior_text, sent_text) < params.similarity_threshold:
            flush_group()
            group.append((sent_text, span_start, span_end))
            continue
        group.append((sent_text, span_start, span_end))
    flush_group()
    return chunks


def _apply_overlap_spans(
    chunk_spans: List[Tuple[int, int]],
    full_text: str,
    overlap_chars: int,
) -> List[Tuple[str, int, int]]:
    """Expand chunk starts backward by ``overlap_chars`` for context overlap."""
    if overlap_chars <= 0:
        return [
            (full_text[start:end], start, end) for start, end in chunk_spans
        ]
    result: List[Tuple[str, int, int]] = []
    for index, (start, end) in enumerate(chunk_spans):
        if index == 0:
            result.append((full_text[start:end], start, end))
            continue
        new_start = max(0, start - overlap_chars)
        result.append((full_text[new_start:end], new_start, end))
    return result


def semantic_sentence_chunks(
    full_text: str,
    params: SemanticChunkingParams,
) -> List[Tuple[str, int, int]]:
    """Split ``full_text`` into semantic chunks with (text, start, end) spans.

    ``end`` is exclusive. Chunk text matches ``full_text[start:end]`` for each
    returned row before any column-level transforms.
    """
    if params.sentence_splitter != SentenceSplitterMode.SIMPLE_REGEX:
        splitter = params.sentence_splitter
        label = getattr(splitter, "value", str(splitter))
        raise ValueError(f"Unsupported sentence splitter: {label}")
    if not full_text or not str(full_text).strip():
        return []
    text = str(full_text)
    sentences = _split_sentences_simple_regex(text)
    sentences = _normalize_sentence_spans(sentences, params.max_chars)
    chunk_spans = _merge_sentence_spans(sentences, params)
    return _apply_overlap_spans(chunk_spans, text, params.overlap_chars)
