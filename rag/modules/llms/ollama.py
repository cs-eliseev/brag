from rag.contracts.llm import LLMContract
from rag.drivers.llms.ollama import OllamaClient

class OllamaLLM(LLMContract):
    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(self, context: str) -> str:
        return self.client.send_request(context)