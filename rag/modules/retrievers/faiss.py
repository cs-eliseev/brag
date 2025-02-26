from rag.contracts.retriever import RetrieverContract
from rag.drivers.databases.faiss_db import FaissDB
from rag.entities.document import DocumentCollection, Document
from rag.entities.vector_store import VectorStoreQueryParams
from rag.utils.logger import logger

class FAISSRetriever(RetrieverContract):
    def __init__(self, client: FaissDB) -> None:
        self.client = client

    def search(self, params: VectorStoreQueryParams) -> DocumentCollection:
        documents = DocumentCollection()
        for document in self.client.search(params):
            logger().debug(f"text: {document.page_content}")
            documents.push(Document(text=document.page_content, metadata={}))
        return documents
