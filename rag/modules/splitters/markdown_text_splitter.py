from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter, MarkdownHeaderTextSplitter

from rag.contracts.splitter import SplitterContract


class MarkdownHeaderTextSplitterWrapper(SplitterContract):
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        return [doc.page_content for doc in self.text_splitter.split_documents(self.md_splitter.split_text(text))]