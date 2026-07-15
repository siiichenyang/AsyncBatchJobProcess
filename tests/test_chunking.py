import pytest

from batch_processor.chunking import (
    TextChunk,
    chunk_text,
)


def test_chunking_text_normal():
    text = "one two three four five six"

    chunks = chunk_text(
        text,
        document_id="1",
        chunk_size=4,
        overlap=2,
    )

    assert chunks == [
        TextChunk(
            document_id="1",
            chunk_index=0,
            text="one two three four",
            start_word=0,
            end_word=4,
        ),
        TextChunk(
            document_id="1",
            chunk_index=1,
            text="three four five six",
            start_word=2,
            end_word=6,
        ),
    ]


def test_chunking_text_empty_input():
    chunks = chunk_text(
        "",
        document_id="1",
        chunk_size=4,
        overlap=2,
    )

    assert not chunks


def test_chunking_text_wrong_overlap_chunk_size():
    text = "one two three four five six"

    with pytest.raises(ValueError, match="overlap"):
        chunk_text(
            text,
            document_id="1",
            chunk_size=4,
            overlap=6,
        )
