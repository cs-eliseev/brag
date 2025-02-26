from pathlib import Path

def base_path() -> Path:
    return (Path(__file__).parent / '..' / '..').resolve()

def absolute_path(relative_path: str) -> Path:
    return base_path() / relative_path

def file_path(dir: str, file: str = '') -> Path:
    relative_path = dir
    if file != '':
        relative_path += f"/{file}"
    return absolute_path(relative_path)

def data_path(file: str = '') -> Path:
    return file_path('data', file)

def config_path(file: str = '') -> Path:
    return file_path('configs', file)

def dataset_path(file: str = '') -> Path:
    return file_path('datasets', file)