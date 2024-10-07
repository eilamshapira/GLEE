from players.base_player import Player
import os
import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
from utils.conversation import get_conv_template


class VertexAIPlayer(Player):
    def __init__(self, public_name, delta=1, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)

        # set google credentials if not set
        if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

        self.project_id = kwargs.get('project_id', 'generative-bot')
        self.location = kwargs.get('location', 'us-central1')

        vertexai.init(project=self.project_id, location=self.location)
        self.model_name = kwargs.get('model_name', 'gemini-1.5-flash')

        self.conv = get_conv_template("gemini") if "gemini" in self.model_name.lower() else (
            get_conv_template("llama-3") if "llama3" in self.model_name.lower().replace("-", "")
            else None)
        assert self.conv is not None, f"Conversation template not found for model {self.model_name}"

        self.user_name = "user"

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def add_message(self, message, role='user'):
        if role == 'system':
            self.conv.system_message = message
        else:
            self.conv.append_message(self.conv.roles[0], message)

    def clean_conv(self):
        self.conv.messages = list()

    def get_text_answer(self, format_checker, decision=False):
        model = GenerativeModel(model_name=self.model_name)
        count = 0
        while count < 7:
            try:
                response = model.generate_content(self.conv.get_prompt(), safety_settings=self.safety_settings)
                self.text_response = response.candidates[0].content.text.strip()
                if format_checker(self.text_response):
                    self.conv.append_message(self.conv.roles[1], self.text_response)
                    break
            except Exception as e:
                print("Error generating content", e)
            count += 1
        return self.text_response

    def set_system_message(self, message):
        self.conv.system_message = message
