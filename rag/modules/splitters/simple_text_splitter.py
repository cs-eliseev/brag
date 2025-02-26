from typing import List
from rag.contracts.splitter import SplitterContract

class SimplTextSplitters(SplitterContract):
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def split_text(self, text: str) -> List[str]:
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]