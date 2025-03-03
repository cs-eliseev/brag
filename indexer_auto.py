import argparse
from pathlib import Path
import ijson
from config_loader.utils import yaml_load_config
from rag.entities.document import DocumentCollection, Document
from rag.utils.path import dataset_path, config_path

class IndexerNotFound(ValueError):
    def __init__(self, indexer_name: str):
        super().__init__(f"Indexer '{indexer_name}' not found in container")

class DatasetFileNotFound(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(f"Dataset file not found: '{path}'")

def read_json_file(filepath: Path) -> dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        for obj in ijson.items(file, 'item'):
            yield obj
    return {}

def get_indexer_type(indexer_name: str) -> str:
    return indexer_name.split('__')[-1]

def indexer(container, indexer_name: str, documents: DocumentCollection) -> None:
    if documents.empty():
        return

    try:
        indexer_provider = container.providers.get(indexer_name)
        if indexer_provider is None:
            raise IndexerNotFound(indexer_name)
        container.log().info(f"Use indexer: {indexer_name}")
        indexer_provider().index(documents)
        container.log().info('success')
    except Exception as e:
        container.log().error(e)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_filename", type=str, required=True, help="Имя файла в дирректории dataset")
args = parser.parse_args()
path = dataset_path(args.dataset_filename)
if not path.is_file():
    raise DatasetFileNotFound(path)

simple_text = DocumentCollection()
markdown_text = DocumentCollection()
html_text = DocumentCollection()

excluded_keys = {"age", "city"}
for item in read_json_file(path):
    metadata = {key: value for key, value in item.items() if key not in excluded_keys}
    if 'text' in item:
        simple_text.push(Document(text=item['text'], metadata=metadata))
    if 'text_html' in item:
        html_text.push(Document(text=item['text_html'], metadata=metadata))
    if 'text_markdown' in item:
        markdown_text.push(Document(text=item['text_markdown'], metadata=metadata))

indexer_configs = yaml_load_config(config_path('indexer_factory.yaml'))

# optimization init
from rag.bootstrap.bootstrap import container

if simple_text.not_empty():
    indexer(container, 'forward_indexer__documents', simple_text)
if html_text.not_empty():
    indexer(container, 'forward_indexer__documents__html', html_text)
if markdown_text.not_empty():
    indexer(container, 'forward_indexer__documents__md', markdown_text)

for splitter_name, value in indexer_configs.get('splitter_indexer').items():
    indexer_name = f"splitter_indexer__{splitter_name}"
    type = get_indexer_type(str(splitter_name))
    if type == 'html':
        indexer(container, indexer_name, html_text)
    elif type == 'md':
        indexer(container, indexer_name, markdown_text)
    else:
        indexer(container, indexer_name, simple_text)
