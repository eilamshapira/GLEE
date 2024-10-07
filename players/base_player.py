import re
import json
from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, public_name, delta=1.0, player_id=0):
        self.public_name = public_name
        self.delta = delta
        self.buffer = ""
        self.response = ""
        self.text_response = ""
        self.user_name = "user"
        self.messages = list()
        self.history = []
        self.req_offer_text = ""
        self.req_decision_text = ""
        self.last_action_json = None
        self.player_id = player_id
        self.rules = ""

        print(f"Prompt to {self.public_name} will \033[{91 + self.player_id}mbe colored\033[0m.")

    def act(self, format_checker, decision=False):
        prompt = self.buffer.strip()
        assert prompt, "Prompt is empty."
        self.add_message(prompt, self.user_name)
        # print the prompt in red
        print(f"\033[{91 + self.player_id}m{prompt}\033[0m")
        self.response = self.get_text_answer(format_checker, decision)
        assert format_checker(self.response), "Invalid format."

        action = re.search(r'\{.*?\}', self.response, re.DOTALL)
        action = action.group() if action else ""
        action = re.sub(r"(?<=\d),(?=\d{3})", "", action)

        for i in range(2):
            try:
                self.last_action_json = json.loads(action)
                # for each value in the JSON, check if it is a number save as string. if it is, change it float
                for key, value in self.last_action_json.items():
                    if isinstance(value, str) and value.replace("$", "").replace(".", "", 1).isdigit():
                        self.last_action_json[key] = float(value.replace("$", ""))
                break
            except json.JSONDecodeError:
                if "$" in action:
                    action = action.replace("$", "")
                else:
                    raise ValueError("Invalid JSON format.")
        else:
            raise ValueError("Invalid JSON format.")
        self.history.append((prompt, self.response))
        self.buffer = ""

    @abstractmethod
    def add_message(self, message, role):
        pass

    def clean_history(self):
        assert self.buffer == "", "Buffer is not empty."
        self.history = []

    def end_chat(self):
        self.buffer = ""
        self.clean_conv()
        print(f"The chat with {self.public_name} has ended.")

    def add_to_buffer(self, text):
        self.buffer += text

    def new_chat(self):
        print(f"New chat started with {self.public_name}.")

    @abstractmethod
    def get_text_answer(self, format_checker, decision=False):
        pass

    @abstractmethod
    def clean_conv(self):
        pass

    def set_system_message(self, message):
        raise NotImplementedError

    def edit_system_message(self, format_requested):
        new_prompt = self.rules.strip() + "\n" + format_requested.strip()
        new_prompt = new_prompt.strip()
        self.set_system_message(new_prompt)
        print("System:", new_prompt)
