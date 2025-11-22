from players.terminal_player import TerminalPlayer
from players.litellm_player import LiteLLMPlayer
from players.otree_player import OtreePlayer
from players.demo_player import DemoPlayer
from players.otree_litellm_player import OtreeLiteLLMPlayer


PLAYERS = {
    'terminal': TerminalPlayer,
    'otree': OtreePlayer,
    'demo': DemoPlayer,
    'otree_LLM': OtreeLiteLLMPlayer,
    'litellm': LiteLLMPlayer
}


def player_factory(player_type, player_args):
    return PLAYERS[player_type](**player_args)
