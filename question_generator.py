import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from rag.bootstrap.bootstrap import container
from rag.utils.path import dataset_path
from rag.utils.logger import logger
from rag.commands.generate_questions_command import GenerateQuestionsCommand

class ValidationError(Exception):
    pass

class FileNotFound(ValidationError):
    def __init__(self, path: Path):
        super().__init__(f"File not found: {path}")

class DocumentsNotFound(ValidationError):
    def __init__(self):
        super().__init__(f"Documents not found")

def validate_args(args: argparse.Namespace) -> None:
    file_path = dataset_path(file=args.dataset)
    if not file_path.is_file():
        raise FileNotFound(file_path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate questions from documents')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Input JSON file name (relative to datasets directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='questions.json',
        help='Output file name for generated questions'
    )

    args = parser.parse_args()
    validate_args(args)
    return args

def main() -> None:
    try:
        args = parse_args()

        json_file_service = container.json_file_service()
        json_service = container.json_service()
        backup_service = container.backup_file_service()

        input_path = dataset_path(file=args.dataset)
        output_path = Path(args.output)

        backup_service.backup(output_path)
        documents = list(json_file_service.read(input_path))
        if not documents:
            raise DocumentsNotFound()

        command = GenerateQuestionsCommand(question_generator_service=container.question_generator_service())

        start_time = datetime.now()
        logger().info("Starting question generation", start_time=start_time.strftime("%H:%M:%S"))

        with open(output_path, 'w', encoding='utf-8') as file:
            for questions in command.execute(documents):
                for question in questions:
                    json_service.dump(question, file)
                    file.write('\n')

        end_time = datetime.now()
        logger().info(
            "Question generation completed",
            start_time=start_time.strftime("%H:%M:%S"),
            end_time=end_time.strftime("%H:%M:%S"),
            duration_seconds=(end_time - start_time).total_seconds()
        )
            
    except Exception as e:
        logger().error(f"{str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()