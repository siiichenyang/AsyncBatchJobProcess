from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    chunk_index: int
    text: str
    start_word: int
    end_word: int


def chunk_text(
    text: str,
    *,
    document_id: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must satisfy 0 <= overlap < chunk_size"
        )

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []

    for chunk_index, start_word in enumerate(
        range(0, len(words), step)
    ):
        end_word = min(start_word + chunk_size, len(words))

        chunks.append(
            TextChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=" ".join(words[start_word:end_word]),
                start_word=start_word,
                end_word=end_word,
            )
        )

        if end_word == len(words):
            break

    return chunks
