from players.base_player import Player
import anthropic
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
from fastchat.model import load_model, get_conversation_template


class ClaudePlayer(Player):   # FIXME currently have some problems - not recommended to use
    def __init__(self, public_name, delta=1, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)
        self.client = anthropic.AnthropicVertex(region="europe-west1", project_id="generative-bot")
        
        assert "model" in kwargs, "Model name is required"
        self.model = kwargs["model"]

        self.conv = get_conversation_template("claude-3-5-sonnet-20240620")

        self.user_name = "user"

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def add_message(self, message, role='user'):
        if role == 'system':
            print("System message: ", message)
            self.conv.system_message = message
        else:
            self.conv.append_message(self.conv.roles[0], message)

    def clean_conv(self):
        self.conv.messages = list()

    def get_text_answer(self, format_checker, decision=False):
        count = 0
        while count < 5:
            response = self.client.messages.create(model=self.model_name,
                                                   messages=[{"role": role, "content": content} for role, content in
                                                             self.conv.messages],
                                                   max_tokens=200,
                                                   system=self.conv.system_message)
            self.text_response = response.content[0].text.strip()
            if format_checker(self.text_response):
                print(self.text_response)
                self.conv.append_message(self.conv.roles[1], self.text_response)
                break
            count += 1
        return self.text_response

    def set_system_message(self, message):
        self.conv.system_message = message
