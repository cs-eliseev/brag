import json
from pathlib import Path
from typing import Dict, List, Iterator, Any


class JsonDocumentServiceError(Exception):
    pass

class FileReadError(JsonDocumentServiceError):
    def __init__(self, path: Path, error: Exception):
        super().__init__(f"Failed to read file {path}: {str(error)}")

class FileWriteError(JsonDocumentServiceError):
    def __init__(self, path: Path, error: Exception):
        super().__init__(f"Failed to write to file {path}: {str(error)}")

class JsonParseError(JsonDocumentServiceError):
    def __init__(self, path: Path, error: Exception):
        super().__init__(f"Failed to parse JSON from file {path}: {str(error)}")

class JsonFileService:
    def read(self, file_path: Path) -> Iterator[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not isinstance(data, list):
                data = [data]

            for doc in data:
                yield doc

        except json.JSONDecodeError as e:
            raise JsonParseError(file_path, e)
        except Exception as e:
            raise FileReadError(file_path, e)


    def write(self, file_path: Path, questions: List[Dict]) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(questions, file, ensure_ascii=False, indent=2)
        except Exception as e:
            raise FileWriteError(file_path, e)

    def stream_write(self, path: Path, documnets: Iterator[Any]) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            for questions in documnets:
                for question in questions:
                    json.dump(question, file, ensure_ascii=False, separators=(",", ":"))
                    file.write('\n')