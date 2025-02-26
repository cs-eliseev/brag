from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.contracts.splitter import SplitterContract


class RecursiveCharacterTextSplitterWrapper(SplitterContract):
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)