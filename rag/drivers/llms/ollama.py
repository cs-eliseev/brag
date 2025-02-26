from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama

class OllamaClient:
    def __init__(self, model: str, temperature: float) -> None:
        self.model = ChatOllama(model=model, temperature=temperature)

    def send_request(self, context: str) -> str:
        """Генерирует ответ модели."""
        return self.model.invoke([HumanMessage(content=context)]).content