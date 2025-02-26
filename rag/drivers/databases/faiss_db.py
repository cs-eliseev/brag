from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from rag.contracts.index_db import IndexDBContract
from rag.contracts.vector_store import VectorStoreContract
from rag.entities.vector_store import VectorStoreQueryParams
from rag.utils.path import absolute_path

class FaissDBNotInitError(Exception):
    def __init__(self) -> None:
        super().__init__(f"Faiss DB not init")

class FaissDBExistError(Exception):
    def __init__(self, db_file: str) -> None:
        super().__init__(f"DB file {db_file} exist")

class FaissDatasetNotExistError(Exception):
    def __init__(self, file_dataset: str) -> None:
        super().__init__(f"Dataset file {file_dataset} does not exist")

class FaissDB(VectorStoreContract, IndexDBContract):
    def __init__(
            self,
            db_path: Path|str,
            embeddings: Embeddings,
    ) -> None:
        self.embeddings = embeddings
        self.db_path = absolute_path(db_path)
        self.db_file = absolute_path(f"{db_path}/index.faiss")
        self.db = None
        if self.db_file.exists():
            self.db = FAISS.load_local(str(self.db_path), self.embeddings, allow_dangerous_deserialization=True)

    def create_db(self, documents: list[Document|str]) -> FAISS:
        if self.db_file.exists():
            raise FaissDBExistError(str(self.db_file))

        self.db = FAISS.from_documents(documents, self.embeddings)
        self.db.save_local(str(self.db_path))
        return self.db

    def delete_db(self) -> None:
        path = str(self.db_path)
        if not self.db_file.exists():
            raise FaissDatasetNotExistError(path)
        self.db_file.unlink()
        self.db = None

    def search(self, dto: VectorStoreQueryParams) -> list[Document]:
        if self.db is None:
            raise FaissDBNotInitError()
        return self.db.similarity_search(query=dto.query, k=dto.max_results)

    def get_db_path(self) -> Path:
        return self.db_path