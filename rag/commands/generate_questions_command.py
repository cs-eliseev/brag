import traceback
from typing import List, Dict
from rag.services.question_generator_service import QuestionGeneratorService
from rag.utils.logger import logger

class QuestionGenerationError(Exception):
    pass

class DocumentProcessingError(QuestionGenerationError):
    def __init__(self, document_id: str, original_error: Exception):
        super().__init__(f"Error processing document {document_id}: {str(original_error)}")

class QuestionGenerateFailed(ValueError):
    def __init__(self):
        super().__init__('No questions generated')

class NoDocumentsProvidedForProcessingError(QuestionGenerationError):
    def __init__(self):
        super().__init__('No documents provided for processing')

class GenerateQuestionsCommand:
    def __init__(self, question_generator_service: QuestionGeneratorService):
        self.question_generator_service = question_generator_service

    def execute(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            raise NoDocumentsProvidedForProcessingError()
            
        failed_documents = []
        total_documents = len(documents)

        for i, document in enumerate(documents, 1):
            logger().info(f"Processing document {i}/{total_documents}")
            try:
                questions = self.question_generator_service.generate_questions(document)
                if not questions:
                    raise QuestionGenerateFailed()
                yield questions
                logger().debug(f"Successfully processed document {document.get('id')}, questions count: {len(questions)}")
            except Exception as e:
                logger().error(f"Failed to process document {document.get('id')}: {str(e)}\n{traceback.format_exc()}")
                failed_documents.append(document.get('id'))

        if failed_documents:
            raise QuestionGenerationError(
                f"Failed to process {len(failed_documents)} documents. "
                f"Successfully processed {total_documents - len(failed_documents)} documents."
            )