from typing import Dict, List
from rag.contracts.llm import LLMContract
from rag.contracts.prompt import PromptContract
from rag.services.json_service import JSONService, JSONParserServiceError

class QuestionGeneratorServiceError(Exception):
    pass

class LLMGenerationError(QuestionGeneratorServiceError):
    def __init__(self, error: Exception):
        super().__init__(f"LLM generation failed: {str(error)}")

class EmptyResponseError(QuestionGeneratorServiceError):
    def __init__(self):
        super().__init__("LLM returned empty response")

class EmptyQuestionsError(QuestionGeneratorServiceError):
    def __init__(self):
        super().__init__("LLM returned empty questions")

class QuestionGeneratorService:
    def __init__(
        self,
        llm: LLMContract,
        prompt: PromptContract,
        json_service: JSONService
    ):
        self.llm = llm
        self.prompt = prompt
        self.json_service = json_service

    def generate_questions(self, document: Dict) -> List[Dict]:
        try:
            response = self._generate_llm_response(document)
            if not response:
                raise EmptyResponseError()

            questions_data = self.json_service.parse(response)
            questions = self._prepare_questions(questions_data.get('questions', []), document)
            
            if not questions:
                raise EmptyQuestionsError()

            return questions
        except JSONParserServiceError:
            raise
        except QuestionGeneratorServiceError:
            raise
        except Exception as e:
            raise LLMGenerationError(e)

    def _generate_llm_response(self, document: Dict) -> str:
        prompt = self.prompt.render(text=document.get('text', ''))
        return self.llm.generate(prompt)

    @staticmethod
    def _prepare_questions(questions: List, document: Dict) -> List[Dict]:
        result = []
        for question in questions:
            item = document.copy()
            item['question'] = question
            result.append(item)
        return result