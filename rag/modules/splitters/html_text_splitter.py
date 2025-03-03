from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter

from rag.contracts.splitter import SplitterContract


class HTMLSectionSplitterWrapper(SplitterContract):
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.html_splitter = HTMLSectionSplitter(headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> List[str]:
        return [doc.page_content for doc in self.text_splitter.split_documents(self.html_splitter.split_text(text))]