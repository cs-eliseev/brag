from rag.drivers.downloads.download_page import DownloadPage
from rag.drivers.google.google_search import GoogleSearch, GoogleSearchQueryParamsDTO
from rag.entities.document import Document, DocumentCollection
from rag.entities.vector_store import VectorStoreQueryParams
from rag.modules.document_evaluating import DocumentRelevanceEvaluator
from rag.contracts.retriever import RetrieverContract
from rag.services.text_clean import TextCleanerService
from rag.services.url import UrlService


class GoogleSearchFullRetriever(RetrieverContract):
    def __init__(
            self,
            text_cleaner: TextCleanerService,
            download_page: DownloadPage,
            url_service: UrlService,
            client: GoogleSearch,
    ) -> None:
        self.text_cleaner = text_cleaner
        self.download_page = download_page
        self.url_service = url_service
        self.client = client

    def search(self, params: VectorStoreQueryParams) -> DocumentCollection:
        return self._prepare_results(
            self.client.search(GoogleSearchQueryParamsDTO(query=params.query, max_results=params.max_results))
        )

    def _prepare_results(self, search_results) -> DocumentCollection:
        documents = DocumentCollection()
        for result in search_results:
            url = result.get('link', '')
            if url:
                title = result.get('title', '')
                document = self._prepare_result_document(result, url, title)
                if document:
                    documents.push(document)

        return documents

    def _prepare_result_document(self, result, url, title) -> Document|None:
        page_text = self.text_cleaner.clean_html_text(self.download_page.download(url))
        if page_text:
            id = self.url_service.get_last_segment(url)
            return Document(text=page_text, metadata={'title': title, 'url': url, 'id': id})
        return None

class GoogleSearchSnippetRetriever(GoogleSearchFullRetriever):
    def __init__(
            self,
            text_cleaner: TextCleanerService,
            download_page: DownloadPage,
            url_service: UrlService,
            client: GoogleSearch,
    ) -> None:
        super().__init__(text_cleaner, download_page, url_service, client)

    def _prepare_result_document(self, result, url, title) -> Document|None:
            id = self.url_service.get_last_segment(url)
            return Document(text=result.get('snippet', ''), metadata={'title': title, 'url': url, 'id': id})

class GoogleSearchChunkRetriever(GoogleSearchFullRetriever):
    def __init__(
            self,
            text_cleaner: TextCleanerService,
            download_page: DownloadPage,
            url_service: UrlService,
            client: GoogleSearch,
            document_evaluating: DocumentRelevanceEvaluator
    ) -> None:
        super().__init__(text_cleaner, download_page, url_service, client)
        self.document_evaluating = document_evaluating

    def search(self, params: VectorStoreQueryParams) -> DocumentCollection:
        return self.document_evaluating.search_chunks(
            params.query,
            self._prepare_results(
                self.client.search(GoogleSearchQueryParamsDTO(query=params.query))
            ),
            params.max_results
        )