import json
from json.decoder import JSONDecodeError
from pathlib import Path
from rag.contracts.file_loader import FileLoaderContract
from rag.entities.document import DocumentCollection, Document

class JsonFileNotFound(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(f"Json file not found: {path}")

class JsonDecodeFailed(JSONDecodeError):
    def __init__(self, path: Path):
        super().__init__(f"Failed to decode json file: {path}")

class JsonFileLoader(FileLoaderContract):
    def load(self, file_path: Path) -> DocumentCollection:
        if not file_path.is_file():
            raise JsonFileNotFound(file_path)

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                documents = DocumentCollection()
                for document in data:
                    documents.push(Document(
                        text=document.get('text'),
                        metadata=document.get('metadata', {}),
                    ))
                return documents
        except(JSONDecodeError):
            raise JsonDecodeFailed(file_path)