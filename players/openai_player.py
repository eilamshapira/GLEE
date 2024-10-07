from players.base_player import Player
import os
from openai import OpenAI


class OpenAIPlayer(Player):
    def __init__(self, public_name, delta, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = kwargs.get('model', "gpt-4o")
        self.text_response = ""
        self.model_response = ""
        self.user_name = "user"
        self.messages = list()
        self.temperature = kwargs.get('temperature', 0.0)

    def add_message(self, message, role='user'):
        if role == 'system':
            self.buffer += message
        self.messages.append({'role': role, 'content': message})

    def clean_conv(self):
        self.messages = list()

    def get_text_answer(self, format_checker, decision=False):

        count = 0
        while count < 10:
            self.model_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                n=1
            )
            self.text_response = self.model_response.choices[0].message.content
            if format_checker(self.text_response):
                self.messages.append(self.model_response.choices[0].message)
                break
        return self.text_response
