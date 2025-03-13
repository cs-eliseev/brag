from rag.bootstrap.bootstrap import container
from rag.contracts.rag import RagContract
from rag.modules.logs.logger import LoggerWrapper
from rag.entities.console import Command
from rag.services.console import ConsoleService
from configs.console import ConsoleConfig

class ConsoleHandler:
    def __init__(
            self,
            commands: Command,
            console_service: ConsoleService,
            rag: RagContract,
            log: LoggerWrapper,
    ) -> None:
        self.commands = commands
        self.console_service = console_service
        self.rag = rag
        self.log = log

    def run(self):
        while True:
            query = self.console_service.send_question(ConsoleConfig.USER_INPUT_QUESTION)
            if not query:
                continue
            if self.commands.is_exit_command(query):
                break

            response = self.rag.query(query)
            self.log.debug(f"Query: {query}")
            self.log.info(f"Response: {response}")

if __name__ == "__main__":
    ConsoleHandler(
        commands=Command(),
        console_service=ConsoleService(),
        rag=container.rag__other_chunk(),
        log=container.log(),
    ).run()