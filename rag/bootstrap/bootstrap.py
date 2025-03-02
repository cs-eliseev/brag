from config_loader.utils import yaml_load_configs, yaml_load_config
from di_autoloader.container_autoloader import ContainerAutoloader
from rag.utils.path import absolute_path, config_path

container = ContainerAutoloader.get_instance_by_callable(
    configuration_function=lambda: yaml_load_config(
        yaml_file=absolute_path(relative_path='configuration.yaml')
    ).get('container'),
    configs_function=lambda: yaml_load_configs(yaml_dir=config_path(), env_path=config_path(file='.env')),
)