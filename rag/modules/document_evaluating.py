import numpy as np
from rag.drivers.embeddings.embedding import EmbeddingWrapper
from rag.modules.logs.logger import LoggerWrapper
from rag.entities.document import DocumentCollection, Document
from rag.modules.splitters.simple_text_splitter import SimplTextSplitters
from rag.services.vector_evaluation_service import VectorEvaluationService

class DocumentRelevanceEvaluator:
    def __init__(
            self,
            embedding: EmbeddingWrapper,
            vector_evaluation_service: VectorEvaluationService,
            split_service: SimplTextSplitters,
    ) -> None:
        self.embedding = embedding
        self.vector_evaluation_service = vector_evaluation_service
        self.split_service = split_service
    def search_chunks(
            self,
            query: str,
            documents: DocumentCollection,
            max_results: int
    ) -> DocumentCollection:
        chunks, metadata = self._split_documents(documents)
        if not chunks:
            return DocumentCollection()

        chunk_embeddings, query_embedding = self._embed_chunks_and_query(chunks, query)
        similarities = self.vector_evaluation_service.cosine([query_embedding], chunk_embeddings)
        top_indices = self._find_top_relevant_chunks(similarities, max_results)

        result = DocumentCollection()
        for i in top_indices:
            payload = metadata[i]
            payload['score'] = similarities[i]
            result.push(Document(text=chunks[i], metadata=payload))

        return result

    def _split_documents(self, documents: DocumentCollection) -> tuple[list[str], list]:
        chunks = []
        metadata = []

        for document in documents.all():
            chunked = self.split_service.split_text(document.text)
            chunks.extend(chunked)
            metadata.extend([document.metadata] * len(chunked))

        return chunks, metadata

    def _embed_chunks_and_query(self, chunks: list[str], query: str) -> tuple[np.ndarray, np.ndarray]:
        chunk_embeddings = self.embedding.embed_documents(chunks)
        query_embedding = self.embedding.embed_query(query)
        return chunk_embeddings, query_embedding

    def _find_top_relevant_chunks(self, similarities: np.ndarray, max_results: int) -> np.ndarray:
        top_indices = np.argpartition(similarities, -max_results)[-max_results:]
        return top_indices[np.argsort(similarities[top_indices])[::-1]]