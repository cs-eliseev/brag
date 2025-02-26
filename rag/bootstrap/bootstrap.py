from pathlib import Path
from di_autoloader import ContainerFactory, ContainerWrapper
from config_loader.utils import yaml_load_configs
from config_loader.yaml_service import YamlReaderService
from rag.utils.path import absolute_path, config_path

class ConfigurationFileNotFound(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(f'Configuration file not found: {path}')

class AppContainer:
    _instance = None

    @classmethod
    def get_instance(
            cls,
            configuration_file: Path = None,
            config_dir: Path = None,
            env_path: Path|None = None
    ) -> ContainerWrapper:
        if cls._instance is None:
            cls._instance = cls._init(configuration_file, config_dir, env_path)
        return cls._instance

    @staticmethod
    def _init(configuration_file: Path, config_dir: Path, env_path: Path | None) -> ContainerWrapper:
        if not configuration_file.is_file():
            raise ConfigurationFileNotFound(configuration_file)

        container = ContainerFactory.make_container(
            YamlReaderService().load(configuration_file).get('container'),
            yaml_load_configs(config_dir, env_path)
        )
        container.init_resources()
        return container

container = AppContainer.get_instance(
    configuration_file=absolute_path('configuration.yaml'),
    config_dir=config_path(),
    env_path=config_path('.env')
)
