class UrlService:
    @staticmethod
    def get_last_segment(url: str) -> str:
        return url.split('/')[-1]