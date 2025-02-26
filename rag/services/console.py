import chardet

class ConsoleService:
    @staticmethod
    def send_question(question: str) -> str:
        user_input = input(question).encode(errors='replace')
        detected_encoding = chardet.detect(user_input)['encoding']

        try:
            return user_input.decode(detected_encoding or 'utf-8').strip()
        except UnicodeDecodeError:
            print('!!! Ошибка парсинга вопрос')
            return ConsoleService.send_question(question)