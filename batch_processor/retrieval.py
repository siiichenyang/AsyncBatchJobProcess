from batch_processor.chunking import chunk_text
from batch_processor.documents import Document
from batch_processor.embeddings import EmbeddingClient
from batch_processor.vector_store import (
    InMemoryVectorStore,
    SearchResult,
)


class Retriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._embedding_client = embedding_client
        self.vector_store = vector_store

    async def index_document(
        self,
        document: Document,
        *,
        chunk_size: int,
        overlap: int = 0,
    ) -> None:
        chunks = chunk_text(
            document.text,
            document_id=document.document_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in chunks:
            embedding = await self._embedding_client.embed(chunk.text)
            self.vector_store.add(chunk, embedding)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        query_embedding = await self._embedding_client.embed(query)

        return self.vector_store.search(query_embedding, top_k=top_k)
