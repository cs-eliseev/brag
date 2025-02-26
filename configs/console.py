from os import environ
from dotenv import load_dotenv

load_dotenv()

class ConsoleConfig:
    EXIT_CODE: set[str] = {'/q', '/exit'}
    USER_INPUT_QUESTION: str = environ.get('USER_INPUT_QUESTION', 'Введите вопрос: ')