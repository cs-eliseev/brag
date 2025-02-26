from bs4 import BeautifulSoup

class TextCleanerService:
    @staticmethod
    def clean_html_text(text: str) -> str:
        text = "\n".join(p.get_text() for p in BeautifulSoup(text, "html.parser").find_all("p"))
        return text.strip()