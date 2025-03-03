from pathlib import Path
from langchain_core.documents import Document
from rag.contracts.file_loader import FileLoaderContract
from rag.contracts.index_db import IndexDBContract
from rag.contracts.indexer import IndexerContract
from rag.contracts.splitter import SplitterContract
from rag.entities.document import DocumentCollection

class SimpleIndexer(IndexerContract):
    def __init__(self,
                 splitter: SplitterContract,
                 db_client: IndexDBContract,
                 file_loader: FileLoaderContract,
    ) -> None:
        self.splitter = splitter
        self.db_client = db_client
        self.file_loader = file_loader

    def index_by_path(self, dataset_path: Path) -> None:
        self.index(self.file_loader.load(dataset_path))

    def index(self, documents: DocumentCollection) -> None:
        dataset = []
        for document in documents.all():
            chunks = self.splitter.split_text(document.text)
            for chunk in chunks:
                dataset.append(Document(page_content=chunk, metadata=document.metadata))

        self.db_client.create_db(dataset)