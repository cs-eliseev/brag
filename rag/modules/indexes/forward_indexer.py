from pathlib import Path
from langchain_core.documents import Document
from rag.contracts.file_loader import FileLoaderContract
from rag.contracts.index_db import IndexDBContract
from rag.contracts.indexer import IndexerContract
from rag.entities.document import DocumentCollection

class ForwardIndexer(IndexerContract):
    def __init__(self,
                 db_client: IndexDBContract,
                 file_loader: FileLoaderContract,
    ) -> None:
        self.db_client = db_client
        self.file_loader = file_loader

    def index_by_path(self, dataset_path: Path) -> None:
        self.index(self.file_loader.load(dataset_path))

    def index(self, documents: DocumentCollection) -> None:
        self.db_client.create_db([
            Document(
                page_content=document.text,
                metadata=document.metadata
            ) for document in documents.all()
        ])