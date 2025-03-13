import re
from rag.contracts.prompt import PromptContract
from rag.entities.document import DocumentCollection
from string import Template

REG_MORE_NEXT_LINE: str = r'\n{2,}'

class Prompt(PromptContract):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def render(self, **kwargs) -> str:
        if 'context' in kwargs:
            kwargs['context'] = self._format_documents(kwargs['context'])

        return Template(self.prompt).safe_substitute(kwargs)

    @staticmethod
    def _format_documents(documents: DocumentCollection) -> str:
        formatted_chunks = [
            f"\n#### {i + 1} Relevant chunk ####\n{document.metadata}\n{document.text}\n"
            for i, document in enumerate(documents.all())
        ]
        return re.sub(REG_MORE_NEXT_LINE, ' ', '\n ' . join(formatted_chunks))

