import json
import re
from typing import Dict

class JSONParserServiceError(Exception):
    pass

class JSONParsingError(JSONParserServiceError):
    def __init__(self, response: str):
        super().__init__(f"Failed to parse JSON from response:\n{response}")

class JSONDataNotFound(JSONParserServiceError):
    def __init__(self, response: str):
        super().__init__(f"Could not find JSON from response:\n{response}")

class JSONService:
    def __init__(self):
        self._json_pattern = r'(\{.*?\})'

    def parse(self, text: str) -> Dict:
        try:
            match = re.search(self._json_pattern, text, re.DOTALL)
            if not match:
                raise JSONDataNotFound(text)

            return json.loads(match.group(1))
        except json.JSONDecodeError:
            raise JSONParsingError(text)

    def dump(self, data: Dict, file, ensure_ascii: bool = False, separators=(",", ":")) -> None:
        json.dump(data, file, ensure_ascii=ensure_ascii, separators=separators)