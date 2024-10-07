from players.terminal_player import TerminalPlayer
from players.openai_player import OpenAIPlayer
from players.gemini_player import GeminiPlayer
from players.anthropic_player import ClaudePlayer
from players.hf_player import HFPlayer
from players.vertexai_player import VertexAIPlayer

from players.otree_player import OtreePlayer
from players.demo_player import DemoPlayer
from players.otree_vertexai_player import VertexAIPlayer as OtreeVertexAIPlayer


PLAYERS = {
    'terminal': TerminalPlayer,
    'openai': OpenAIPlayer,
    'gemini': GeminiPlayer,
    'claude': ClaudePlayer,
    'hf': HFPlayer,
    'vertexai': VertexAIPlayer,
    'otree': OtreePlayer,
    'demo': DemoPlayer,
    'otree_LLM': OtreeVertexAIPlayer
}


def player_factory(player_type, player_args):
    return PLAYERS[player_type](**player_args)
