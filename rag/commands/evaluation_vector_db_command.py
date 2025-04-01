from typing import Dict, List, Tuple, Generator
from tqdm import tqdm
from rag.drivers.databases.faiss_db import FaissDB
from rag.drivers.embeddings.embedding import EmbeddingWrapper
from rag.entities.vector_store import VectorStoreQueryParams
from rag.modules.metrics.quality import QualityAnalyzer
from rag.modules.metrics.metrics import MetricsCollection
from rag.services.vector_evaluation_service import VectorEvaluationService
from rag.utils.logger import logger

class EvaluationVectorDBCommand:
    def __init__(
            self,
            quality_analyzer: QualityAnalyzer,
            embedding: EmbeddingWrapper,
            vector_eval: VectorEvaluationService,
            metrics_collector: MetricsCollection,
            max_results: int = 5
    ):
        self.quality_analyzer = quality_analyzer
        self.embedding = embedding
        self.vector_eval = vector_eval
        self.metrics_collector = metrics_collector
        self.max_results = max_results

    def execute(
            self,
            vector_dbs: List[Tuple[str, FaissDB]],
            questions: List[Dict]
    ) -> Generator[Dict, None, None]:
        if not vector_dbs:
            raise ValueError("No vector databases found for evaluation")

        if not questions:
            raise ValueError("No questions found for evaluation")

        failed_dbs = []
        total_dbs = len(vector_dbs)

        for i, (db_name, db) in enumerate(vector_dbs, 1):
            logger().info(f"Evaluating database {i}/{total_dbs}: {db_name}")
            try:
                db_result = self._evaluate_database(db_name, db, questions)
                yield db_result
            except Exception as e:
                logger().error(f"Error evaluating database {db_name}: {str(e)}")
                failed_dbs.append(db_name)

        if failed_dbs:
            logger().warning(
                f"Failed to evaluate {len(failed_dbs)} databases. "
                f"Successfully evaluated {total_dbs - len(failed_dbs)} databases."
            )

    def _evaluate_database(
            self,
            db_name: str,
            db: FaissDB,
            questions: List[Dict]
    ) -> Dict:
        processed_count = 0
        question_results = []

        print(f"\nEvaluating database: {db_name}")

        for i, question_data in enumerate(tqdm(questions[:100], desc=f"Processing questions for {db_name}")):
            question = question_data["question"]

            try:
                documents = db.search(VectorStoreQueryParams(query=question, max_results=self.max_results))

                if len(documents) == 0:
                    continue

                query_embedding = self.embedding.embed_query(question)
                document_texts = [doc.page_content for doc in documents]
                document_embeddings = self.embedding.embed_documents(document_texts)

                similarities = self.vector_eval.cosine([query_embedding], document_embeddings)

                doc_data = []
                for j, doc in enumerate(documents):
                    doc_data.append({
                        "content": doc.page_content,
                        "similarity": float(similarities[j]),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    })

                self.metrics_collector.start_operation("vector_search")
                
                search_metrics = self.quality_analyzer.analyze_search_quality(
                    query=question,
                    similarity_scores=similarities,
                    documents=doc_data,
                    db_name=db_name
                )

                result = {
                    "question": question,
                    "db_name": db_name,
                    "metrics": {
                        "avg_similarity_score": search_metrics.avg_similarity_score,
                        "max_similarity_score": search_metrics.max_similarity_score,
                        "min_similarity_score": search_metrics.min_similarity_score,
                        "similarity_std": search_metrics.similarity_std,
                        "retrieved_count": search_metrics.retrieved_count
                    },
                    "documents": doc_data
                }

                question_results.append(result)
                processed_count += 1

            except Exception as e:
                logger().error(f"Error processing question '{question}' for DB '{db_name}': {str(e)}")

        return {
            "database": db_name,
            "total_processed": processed_count,
            "results": question_results
        } 