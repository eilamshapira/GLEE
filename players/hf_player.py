from players.base_player import Player
import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import os
from huggingface_hub import login


class HFPlayer(Player):
    def __init__(self, public_name, delta=1.0, load_hf_model=True, player_id=0, **kwargs):
        super().__init__(public_name, delta, player_id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = kwargs['model_name']

        if load_hf_model:
            self.model, self.tokenizer = load_model(kwargs['model_name'], self.device, **kwargs['model_kwargs'])
        else:
            self.model, self.tokenizer = None, None
        self.conv = get_conversation_template(
            kwargs['model_name'] if "microsoft/phi" not in kwargs['model_name'].lower() else "TinyLlama")
        self.add_openai_wrap = kwargs.get('openai_wrap', False)
        if self.add_openai_wrap:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.oai_model = kwargs.get('oai_model', "gpt-4o")
        self.kwargs = kwargs

    def add_message(self, message, role='user'):
        if role == 'system':
            self.conv.system_message = message
        else:
            self.conv.append_message(self.conv.roles[0], message)

    def clean_conv(self):
        self.conv.messages = list()

    def openai_parsing(self, message, decision: bool = False):
        oai_messages = [{'role': 'system', 'content': "You are a helpful message parser. You receive messages that don'"
                                                      "t follow some instruction and you reformat them (while keeping"
                                                      "the same semantic) to follow the instruction."}]
        instruction = self.req_offer_text if not decision else self.req_decision_text
        oai_messages.append({'role': 'system', 'content': f"Here are the instructions:\n{instruction}"})
        oai_messages.append({'role': 'user', 'content': f'Please reformat this message {message}'})
        response = self.client.chat.completions.create(
            model=self.oai_model,
            messages=oai_messages,
            temperature=0.0,
            n=1
        )
        return response.choices[0].message.content

    def set_system_message(self, message):
        self.conv.system_message = message

    def get_text_answer(self, format_checker, decision=False):
        self.text_response = ""
        count = 0
        self.conv.append_message(self.conv.roles[1], None)
        while count < 10:
            count += 1
            prompt = self.conv.get_prompt()
            # Run inference
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            self.model_response = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.kwargs.get('temperature', 0.5),
                max_new_tokens=300,
                pad_token_id=self.tokenizer.eos_token_id
            )
            if self.model.config.is_encoder_decoder:
                self.model_response = self.model_response[0]
            else:
                self.model_response = self.model_response[0][len(inputs["input_ids"][0]):]
            self.text_response = self.tokenizer.decode(
                self.model_response, skip_special_tokens=True, spaces_between_special_tokens=False
            ).strip()
            print("Response from model:", self.text_response)
            if format_checker(self.text_response):
                self.conv.messages[-1][-1] = self.text_response
                break
            else:
                print("Failed format check. The response is:\n", self.text_response)
                if self.add_openai_wrap:
                    reformatted = self.openai_parsing(self.text_response, decision)
                    if format_checker(reformatted):
                        self.text_response = reformatted
                        self.conv.messages[-1][-1] = self.text_response
                        break
        return self.text_response
