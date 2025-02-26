from configs.console import ConsoleConfig

class Command:
    def __init__(self, exit_code: set[str] = ConsoleConfig.EXIT_CODE) -> None:
        self.exit_code = exit_code

    def is_exit_command(self, command: str) -> bool:
        return command in self.exit_code