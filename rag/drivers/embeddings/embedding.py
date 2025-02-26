from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingWrapper(Embeddings):
    def __init__(self, model_name: str, driver: str) -> None:
        self.embedding = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs={'device': driver}
        )

    def embed_documents(self, documents: list[str]) -> list:
        return self.embedding.embed_documents(documents)

    def embed_query(self, query: str) -> list:
        return self.embedding.embed_query(query)

    def get_embedding(self) -> HuggingFaceEmbeddings:
        return self.embedding