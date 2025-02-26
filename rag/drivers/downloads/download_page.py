import requests

class DownloadPageResponseError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"Download page request error! Code: {code}, message: {message}")

class DownloadPage:
    def download(self, url: str, headers: dict[str, str] = {}) -> str:

        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.text

        raise DownloadPageResponseError(response.status_code, response.text)