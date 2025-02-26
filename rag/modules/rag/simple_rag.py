from rag.contracts.llm import LLMContract
from rag.contracts.prompt import PromptContract
from rag.contracts.rag import RagContract
from rag.contracts.retriever import RetrieverContract
from rag.entities.vector_store import VectorStoreQueryParams

class SimpleRAG(RagContract):
    def __init__(
            self,
            retriever: RetrieverContract,
            prompt: PromptContract,
            llm: LLMContract,
            max_search_results: int,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.max_search_results = max_search_results

    def query(self, question: str) -> str:
        formatted_content = self.prompt.render(
            question=question,
            context=self.retriever.search(VectorStoreQueryParams(query=question, max_results=self.max_search_results))
        )
        return self.llm.generate(formatted_content)