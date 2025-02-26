import argparse
from pathlib import Path
from config_loader.config import ConfigFactory
from config_loader.yaml_service import YamlReaderService

from rag.utils.path import dataset_path, absolute_path

class UndefinedIndexer(Exception):
    def __init__(self, indexer_type: str):
        super().__init__(f"Undefined indexer: '{indexer_type}'")

class IndexerNotFound(ValueError):
    def __init__(self, indexer: str):
        super().__init__(f"Indexer '{indexer}' not found in container")

class UndefinedIndexerHandler(Exception):
    def __init__(self, index_type: str, index_handler_type: str):
        super().__init__(f"Undefined '{index_type}' handler: '{index_handler_type}'")

class DatasetFileNotFound(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(f"Dataset file not found: '{path}'")

def validate_args(args) -> None:
    if args.indexer_type not in ['forward_indexer', 'splitter_indexer']:
        raise UndefinedIndexer(args.index_type)

    configs = ConfigFactory.create(YamlReaderService.load(absolute_path('configuration.yaml')))
    if configs.get(f"container.{args.indexer_type}.kwargs_factory.{args.handler_type}") is None:
        raise UndefinedIndexerHandler(args.indexer_type, args.handler_type)

    path = dataset_path(args.dataset_filename)
    if not path.is_file():
        raise DatasetFileNotFound(path)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_filename", type=str, required=True, help="Имя файла в дирректории dataset")
parser.add_argument("--indexer_type", type=str, required=True, help="Тип индексатора")
parser.add_argument("--handler_type", type=str, required=True, help="Тип обработчика")

args = parser.parse_args()
validate_args(args)

# optimization init
from rag.bootstrap.bootstrap import container
indexer = f"{args.indexer_type}__{args.handler_type}"
indexer_provider = container.providers.get(indexer)
if indexer_provider is None:
    raise IndexerNotFound(indexer)
indexer_instance = indexer_provider()
print(f"Use indexer: {indexer}")
indexer_instance.index(dataset_path(args.dataset_filename))
print('success')