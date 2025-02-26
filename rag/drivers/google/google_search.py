import requests
from pydantic import BaseModel

class GoogleSearchQueryParamsDTO(BaseModel):
    query: str
    max_results: int = 10
    result_lang: str = 'ru'

class GoogleSearchResponseError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Google search request error! Code: {code}, message: {message}")

class GoogleSearch:
    def __init__(
            self,
            app_key: str,
            app_secret: str,
            api_url: str,
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.api_url = api_url

    def search(self, dto: GoogleSearchQueryParamsDTO):
        response = requests.get(self.api_url, params={
            "q": dto.query,
            "key": self.app_secret,
            "cx": self.app_key,
            "num": dto.max_results,
            "hl": dto.result_lang,
        })

        if response.status_code == 200:
            return response.json().get("items", [])

        raise GoogleSearchResponseError(response.status_code, response.text)