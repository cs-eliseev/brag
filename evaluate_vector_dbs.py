import json
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from rag.bootstrap.bootstrap import container
from rag.modules.metrics.quality import QualityAnalyzer
from rag.modules.metrics.metrics import MetricsCollection
from rag.utils.logger import logger
from rag.services.vector_database_service import VectorDatabaseService
from rag.commands.evaluation_vector_db_command import EvaluationVectorDBCommand
from rag.services.results_service import ResultsService

DEFAULT_INPUT = "datasets/questions.json"

class ValidationError(Exception):
    pass

class FileNotFound(ValidationError):
    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")

def load_questions(file_path: Path) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f.readlines()]
    return questions

def validate_args(args: argparse.Namespace) -> None:
    questions_path = Path(args.input_file)
    if not questions_path.is_file():
        raise FileNotFound(questions_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Vector Database Evaluation')
    parser.add_argument(
        '--input_file',
        type=str,
        default=DEFAULT_INPUT,
        help='Path to the questions file'
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        help='Output file name'
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
        args = parse_args()

        questions_file = Path(args.input_file)
        output_dir = Path(args.output_dir)
        max_results = args.max_results

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_name = args.output_file_name if (hasattr(args, 'output_file_name')
                                                     and args.output_file_name) else f"vector_db_evaluation_{timestamp}"
        output_json = output_dir / f"{output_file_name}.json"
        output_csv = output_dir / f"{output_file_name}.csv"

        output_dir.mkdir(exist_ok=True)

        embedding = container.embedding_driver()
        vector_eval = container.vector_evaluation_service()
        metrics_collector = MetricsCollection()
        quality_analyzer = QualityAnalyzer(metrics_collector=metrics_collector)

        vector_db_service = VectorDatabaseService(embedding=embedding)
        evaluation_command = EvaluationVectorDBCommand(
            quality_analyzer=quality_analyzer,
            embedding=embedding,
            vector_eval=vector_eval,
            metrics_collector=metrics_collector,
            max_results=max_results
        )
        results_service = ResultsService()

        vector_dbs = vector_db_service.get_vector_databases()
        logger().info(f"Found {len(vector_dbs)} vector databases")

        questions = load_questions(questions_file)
        logger().info(f"Loaded {len(questions)} questions")

        start_time = datetime.now()
        logger().info(f"Starting vector database evaluation", start_time=start_time.strftime("%H:%M:%S"))

        table_data = []
        all_results = []

        for db_summary in evaluation_command.execute(vector_dbs, questions):
            db_name = db_summary["database"]
            db_results = db_summary["results"]
            
            total, avg_sim, max_sim, min_sim, std_sim = results_service.calculate_metrics_from_results(db_results)

            if total > 0:
                table_data.append([
                    db_name,
                    total,
                    f"{avg_sim:.4f}",
                    f"{max_sim:.4f}",
                    f"{min_sim:.4f}",
                    f"{std_sim:.4f}"
                ])

            all_results.append({
                "database": db_name,
                "total_processed": total,
                "results": db_results
            })

        results_service.save_results(
            all_results=all_results,
            table_data=table_data,
            output_json=output_json,
            output_csv=output_csv,
            timestamp=datetime.now().isoformat()
        )

        # Completion and logging
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger().info(
            "Vector database evaluation completed",
            start_time=start_time.strftime("%H:%M:%S"),
            end_time=end_time.strftime("%H:%M:%S"),
            duration_seconds=duration
        )
    except ValidationError as e:
        logger().error(f"Validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger().error(f"Error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 