from players.base_player import Player
import google.generativeai as genai


class GeminiPlayer(Player):
    def __init__(self, public_name, delta=1, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)
        self.api_key = open("google.key", "r").read().strip()
        genai.configure(api_key=self.api_key)
        self.model_name = kwargs.get('model_name', 'gemini-1.5-flash')
        self.model = genai.GenerativeModel(self.model_name)
        self.messages = list()
        self.user_name = "user"

    def add_message(self, message, role='user'):
        if role == "system":
            self.buffer += message
        self.messages.append({'role': role, 'parts': [message]})

    def clean_conv(self):
        self.messages = list()

    def get_text_answer(self, format_checker, decision=False):
        count = 0
        while count < 10:
            self.model_response = self.model.generate_content(self.messages)
            self.text_response = self.model_response.text
            if format_checker(self.text_response):
                self.messages.append({'role': 'model', 'parts': [self.text_response]})
                break
        return self.text_response
