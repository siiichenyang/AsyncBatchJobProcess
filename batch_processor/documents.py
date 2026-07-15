from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    source_path: str
    text: str


def load_text_document(
    path: str | Path,
    *,
    document_id: str,
) -> Document:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(
            "document_id must be a non-empty string"
        )

    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")

    return Document(
        document_id=document_id,
        source_path=str(source_path),
        text=text,
    )
