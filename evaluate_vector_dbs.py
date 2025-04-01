import json
import os
import sys
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict, List, Tuple, Generator
from datetime import datetime

from rag.bootstrap.bootstrap import container
from rag.drivers.databases.faiss_db import FaissDB
from rag.services.vector_evaluation_service import VectorEvaluationService
from rag.entities.vector_store import VectorStoreQueryParams
from rag.drivers.embeddings.embedding import EmbeddingWrapper
from rag.modules.monitoring.quality import QualityAnalyzer, SearchQualityMetrics
from rag.modules.monitoring.metrics import MetricsCollector
from rag.utils.logger import logger


# Define custom exceptions
class ValidationError(Exception):
    pass


class FileNotFound(ValidationError):
    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")


class DatabasesNotFound(ValidationError):
    def __init__(self):
        super().__init__(f"No vector databases found for evaluation")


class QuestionsNotFound(ValidationError):
    def __init__(self):
        super().__init__(f"No questions found for evaluation")


class EvaluationError(Exception):
    pass


class DatabaseEvaluationError(EvaluationError):
    def __init__(self, db_name: str, original_error: Exception):
        super().__init__(f"Error evaluating database {db_name}: {str(original_error)}")


# Command class for evaluating vector databases
class EvaluateVectorDBsCommand:
    def __init__(
            self,
            quality_analyzer: QualityAnalyzer,
            embedding: EmbeddingWrapper,
            vector_eval: VectorEvaluationService,
            metrics_collector: MetricsCollector,
            max_results: int = 5
    ):
        self.quality_analyzer = quality_analyzer
        self.embedding = embedding
        self.vector_eval = vector_eval
        self.metrics_collector = metrics_collector
        self.max_results = max_results

    def execute(self, vector_dbs: List[Tuple[str, FaissDB]], questions: List[Dict],
                temp_dir: Path) -> Generator[Dict, None, None]:
        """Executes evaluation for each database."""
        if not vector_dbs:
            raise DatabasesNotFound()

        if not questions:
            raise QuestionsNotFound()

        failed_dbs = []
        total_dbs = len(vector_dbs)

        for i, (db_name, db) in enumerate(vector_dbs, 1):
            logger().info(f"Evaluating database {i}/{total_dbs}: {db_name}")
            try:
                db_result = self._evaluate_database(db_name, db, questions, temp_dir)
                yield db_result
            except Exception as e:
                logger().error(f"Error evaluating database {db_name}: {str(e)}\n{traceback.format_exc()}")
                failed_dbs.append(db_name)

        if failed_dbs:
            logger().warning(
                f"Failed to evaluate {len(failed_dbs)} databases. "
                f"Successfully evaluated {total_dbs - len(failed_dbs)} databases."
            )

    def _evaluate_database(self, db_name: str, db: FaissDB, questions: List[Dict],
                           temp_dir: Path) -> Dict:
        """Evaluates the quality of documents retrieved from a vector database."""
        processed_count = 0

        print(f"\nEvaluating database: {db_name}")

        for i, question_data in enumerate(tqdm(questions[:100], desc=f"Processing questions for {db_name}")):
            question = question_data["question"]

            try:
                # Get documents from vector database
                documents = db.search(VectorStoreQueryParams(query=question, max_results=self.max_results))

                # If no documents received, continue
                if len(documents) == 0:
                    continue

                # Get embeddings for question and documents
                query_embedding = self.embedding.embed_query(question)
                document_texts = [doc.page_content for doc in documents]
                document_embeddings = self.embedding.embed_documents(document_texts)

                # Calculate similarity
                similarities = self.vector_eval.cosine([query_embedding], document_embeddings)

                # Analyze search quality
                doc_data = []
                for j, doc in enumerate(documents):
                    doc_data.append({
                        "content": doc.page_content,
                        "similarity": float(similarities[j]),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    })

                # Начинаем операцию vector_search
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

                # Save result to file
                self._save_question_result(db_name, result, i, temp_dir)
                processed_count += 1

            except Exception as e:
                logger().error(f"Error processing question '{question}' for DB '{db_name}': {str(e)}")

        return {
            "database": db_name,
            "total_processed": processed_count
        }

    def _save_question_result(self, db_name: str, result: Dict, question_idx: int, temp_dir: Path):
        """Saves the result of processing a question to a separate file."""
        file_path = temp_dir / f"{db_name}_question_{question_idx}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return file_path


# Get list of all vector databases
def get_vector_databases() -> List[Tuple[str, FaissDB]]:
    """Gets a list of all vector databases from the databases directory."""
    database_dir = Path("databases")
    vector_dbs = []

    # Get embedding driver instance
    embedding = container.embedding_driver()

    # Check all subdirectories in databases
    for db_path in database_dir.iterdir():
        if db_path.is_dir() and (db_path / "index.faiss").exists():
            db_name = db_path.name
            db = FaissDB(db_path=db_path, embeddings=embedding)
            vector_dbs.append((db_name, db))

    return vector_dbs


# Load questions from file
def load_questions(file_path: Path) -> List[Dict]:
    """Loads questions from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f.readlines()]
    return questions


# Function to load all results for a database
def load_db_results(db_name: str, temp_dir: Path) -> List[Dict]:
    """Loads all results for the specified database."""
    results = []
    pattern = f"{db_name}_question_*.json"

    for file_path in Path(temp_dir).glob(pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            results.append(json.load(f))

    return results


# Function to calculate aggregated metrics
def calculate_db_metrics(db_name: str, temp_dir: Path) -> Tuple[int, float, float, float, float]:
    """Calculates aggregated metrics for a database."""
    results = load_db_results(db_name, temp_dir)

    if not results:
        return 0, 0.0, 0.0, 0.0, 0.0

    total = len(results)
    avg_sim = np.mean([r["metrics"]["avg_similarity_score"] for r in results])
    max_sim = np.mean([r["metrics"]["max_similarity_score"] for r in results])
    min_sim = np.mean([r["metrics"]["min_similarity_score"] for r in results])
    std_sim = np.mean([r["metrics"]["similarity_std"] for r in results])

    return total, avg_sim, max_sim, min_sim, std_sim


def validate_args(args: argparse.Namespace) -> None:
    """Validates command line arguments."""
    questions_path = Path(args.questions_file)
    if not questions_path.is_file():
        raise FileNotFound(questions_path)


# Default constants for convenience when imported
QUESTIONS_FILE = "datasets/questions.json"


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Vector Database Evaluation')
    parser.add_argument(
        '--questions_file',
        type=str,
        default=QUESTIONS_FILE,
        help='Path to the questions file (default: datasets/questions.json)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="results",
        help='Directory for saving results (default: results)'
    )
    parser.add_argument(
        '--max_results',
        type=int,
        default=5,
        help='Maximum number of documents to retrieve (default: 5)'
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    try:
        # Parse command line arguments
        args = parse_args()

        # Set configuration
        questions_file = Path(args.questions_file)
        output_dir = Path(args.output_dir)
        max_results = args.max_results

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_json = output_dir / f"vector_db_evaluation_{timestamp}.json"
        output_csv = output_dir / f"vector_db_evaluation_{timestamp}.csv"
        temp_results_dir = output_dir / f"temp_results_{timestamp}"

        # Create directories for results
        output_dir.mkdir(exist_ok=True)
        temp_results_dir.mkdir(exist_ok=True)

        # Initialize necessary components
        embedding = container.embedding_driver()
        vector_eval = container.vector_evaluation_service()
        metrics_collector = MetricsCollector()
        quality_analyzer = QualityAnalyzer(metrics_collector=metrics_collector)

        # Create evaluation command
        evaluate_command = EvaluateVectorDBsCommand(
            quality_analyzer=quality_analyzer,
            embedding=embedding,
            vector_eval=vector_eval,
            metrics_collector=metrics_collector,
            max_results=max_results
        )

        # Get all vector databases
        vector_dbs = get_vector_databases()
        logger().info(f"Found {len(vector_dbs)} vector databases")

        # Load questions
        questions = load_questions(questions_file)
        logger().info(f"Loaded {len(questions)} questions")

        # Record start time
        start_time = datetime.now()
        logger().info(f"Starting vector database evaluation", start_time=start_time.strftime("%H:%M:%S"))

        # Results for table and JSON
        table_data = []
        all_results = []

        # Perform evaluation for each database
        for db_summary in evaluate_command.execute(vector_dbs, questions, temp_results_dir):
            db_name = db_summary["database"]

            # Calculate aggregated metrics
            total, avg_sim, max_sim, min_sim, std_sim = calculate_db_metrics(db_name, temp_results_dir)

            # Add to final table
            if total > 0:
                table_data.append([
                    db_name,
                    total,
                    f"{avg_sim:.4f}",
                    f"{max_sim:.4f}",
                    f"{min_sim:.4f}",
                    f"{std_sim:.4f}"
                ])

            # Collect full results for saving to JSON
            db_results = load_db_results(db_name, temp_results_dir)
            all_results.append({
                "database": db_name,
                "total_processed": total,
                "results": db_results
            })

        # Save all results to JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json_out = {
                "timestamp": datetime.now().isoformat(),
                "databases": all_results
            }
            json.dump(json_out, f, ensure_ascii=False, indent=2)

        # Create and display results table
        headers = ["Database", "Questions Processed", "Average Similarity", "Max Similarity", "Min Similarity",
                   "Standard Deviation"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        # Create DataFrame and save to CSV
        df = pd.DataFrame(table_data, columns=headers)
        df.to_csv(output_csv, index=False)

        # Completion and logging
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger().info(
            "Vector database evaluation completed",
            start_time=start_time.strftime("%H:%M:%S"),
            end_time=end_time.strftime("%H:%M:%S"),
            duration_seconds=duration
        )

        print("\nVector Database Evaluation Results:")
        print(table)
        print(f"\nDetailed results saved to {output_json}")
        print(f"Results table saved to {output_csv}")

    except ValidationError as e:
        logger().error(f"Validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger().error(f"Error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 