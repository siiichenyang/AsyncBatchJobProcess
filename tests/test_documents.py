from pathlib import Path

from batch_processor.documents import load_text_document
from batch_processor.chunking import chunk_text


def test_load_document(tmp_path):
    doc_path = tmp_path / "document.txt"

    text = "This is file content."

    doc_path.write_text(
        text,
        encoding="utf-8",
    )

    document = load_text_document(doc_path, document_id="001")

    assert document.document_id == "001"
    assert Path(document.source_path) == Path(doc_path)
    assert document.text == text

    chunks = chunk_text(
        document.text,
        document_id=document.document_id,
        chunk_size=4,
        overlap=2,
    )

    assert chunks
    assert all(
        chunk.document_id == "001"
        for chunk in chunks
    )
