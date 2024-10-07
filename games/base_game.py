import json
from abc import ABC, abstractmethod
from utils.utils import create_prompts_from_templates
from players.base_player import Player


class Game(ABC):
    def __init__(self, player_1: Player, player_2: Player, data_logger):
        self.player_1 = player_1
        self.player_2 = player_2
        self.player_1_rules = ''
        self.player_2_rules = ''
        self.data_logger = data_logger

    def set_rules(self, prompt_path, infos, player='p1'):
        if '.txt' in prompt_path:
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
        elif '.json' in prompt_path:
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
            prompt = prompts['seller'] if player == 'p1' else prompts['buyer']

        prompt = create_prompts_from_templates(prompt, infos)
        if player == 'p1':
            self.player_1_rules = prompt
        else:
            self.player_2_rules = prompt
        player = self.player_1 if player == 'p1' else self.player_2
        player.rules = prompt
        player.add_message(prompt, 'system')

    def add_format(self, prompt_path, infos, player='p1'):
        if '.txt' in prompt_path:
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
        elif '.json' in prompt_path:
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
            prompt = prompts['seller'] if player == 'p1' else prompts['buyer']

        prompt = create_prompts_from_templates(prompt, infos)
        player = self.player_1 if player == 'p1' else self.player_2
        player.req_offer_text = prompt

    def get_other_player(self, base='p1'):
        base_player = self.player_1 if base == 'p1' else self.player_2
        other_player = self.player_2 if base == 'p1' else self.player_1
        return base_player, other_player

    @abstractmethod
    def game(self):
        pass
