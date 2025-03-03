from typing import List, Any
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    metadata: dict[str, Any]

class DocumentCollection:
    def __init__(self, items: List[Document] = None):
        if items is None:
            items = []
        self.items = items

    def push(self, document: Document) -> None:
        self.items.append(document)

    def pull(self) -> list[Document]:
        items = self.items
        self.items = []
        return items

    def all(self) -> list[Document]:
        return self.items

    def pop(self) -> Document:
        return self.items.pop()

    def shift(self) -> Document:
        return self.items.pop(0)

    def last(self) -> Document:
        return self.items[-1]

    def first(self) -> Document:
        return self.items[0]

    def iterator(self):
        for item in self.items:
            yield item
        return []

    def not_empty(self) -> bool:
        return bool(self.items)

    def empty(self) -> bool:
        return not self.not_empty()