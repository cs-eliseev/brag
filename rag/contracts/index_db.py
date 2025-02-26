from abc import ABC, abstractmethod
from langchain.schema import Document

class IndexDBContract(ABC):
    @abstractmethod
    def create_db(self, documents: list[Document]):
        pass

    @abstractmethod
    def delete_db(self) -> None:
        pass