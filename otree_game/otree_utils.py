import json
import os
import re
import hashlib
from games.bot_factory import bot_factory
from games.base_bot import Bot
from players.player_factory import player_factory

from .models import Constants, Player


# ------------- UTILS FUNCTIONS -------------


def load_bot_name(path):
    full_path = os.path.abspath(fr"{Constants.base_config_path}/{path}.json")
    try:
        args = json.load(open(full_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found in path: {full_path}")
    player_1_args = args["player_1_args"]
    player_2_args = args["player_2_args"]
    player_1_type = args["player_1_type"]
    player_2_type = args["player_2_type"]

    if "otree" not in (player_1_type, player_2_type):
        raise ValueError("One of the players must be an oTree player")
    if player_1_type == "otree" and player_2_type == "otree":
        raise ValueError("At least one player must not be an oTree player")

    if player_1_type == "otree":
        return player_2_args["public_name"]
    else:
        return player_1_args["public_name"]


def create_bot(path, human_name=None) -> Bot:
    # Attempt to load the configuration file
    full_path = os.path.abspath(fr"{Constants.base_config_path}/{path}.json")
    try:
        args = json.load(open(full_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found in path: {full_path}")

    game_type = args["game_type"]
    game_params = args["game_args"]
    player_1_args = args["player_1_args"]
    player_2_args = args["player_2_args"]
    player_1_type = args["player_1_type"]
    player_2_type = args["player_2_type"]

    if "otree" not in (player_1_type, player_2_type):
        raise ValueError("One of the players must be an oTree player")
    if player_1_type == "otree" and player_2_type == "otree":
        raise ValueError("At least one player must not be an oTree player")
    human_start = "otree" == player_1_type
    if human_name is not None:
        if human_start:
            player_1_args["public_name"] = human_name
            if player_2_type == "demo":
                player_2_args["other_player_name"] = human_name
        else:
            player_2_args["public_name"] = human_name
            if player_1_type == "demo":
                player_1_args["other_player_name"] = human_name
    first_player = player_factory(player_1_type, player_1_args)
    second_player = player_factory(player_2_type, player_2_args)
    new_bot = bot_factory(game_type, first_player, second_player, human_start, game_params)
    new_bot.initialize()
    return new_bot


def load_bot(player, human_name=None):
    if 'bot' not in player.participant.vars:
        player.participant.vars['bot'] = create_bot(player.session.config['path'], human_name=human_name)
        print("Bot created in load_bot")
    return player.participant.vars['bot']


def save_round(player: Player):
    bot = load_bot(player)
    fields = bot.get_otree_player_fields()
    player.player_name = bot.human_player.public_name
    player.real_turn = bot.turn - 1
    player.config_path = player.session.config['path']
    player.who_propose = bot.who_propose(minus_one=True)  # we already updated the turn
    player.offer = player.offer if 'offer' in fields else None
    player.proposer_message = player.proposer_message if 'proposer_message' in fields else None
    player.proposer_recommendation = player.proposer_recommendation if 'proposer_recommendation' in fields else None
    player.receiver_message = player.receiver_message if 'receiver_message' in fields else None
    player.accepted = player.accepted
    player.additional_info = player.additional_info if 'additional_info' in fields else None


def complete_code_hash(player: Player):
    participant = player.participant.id
    bot = load_bot(player)
    fields = bot.get_otree_player_fields()
    config_path = player.session.config['path']
    player_name = bot.human_player.public_name
    real_turn = bot.turn
    who_propose = bot.who_propose()
    offer = player.offer if 'offer' in fields else 'None'
    proposer_message = player.proposer_message if 'proposer_message' in fields else 'None'
    proposer_recommendation = player.proposer_recommendation if 'proposer_recommendation' in fields else 'None'
    receiver_message = player.receiver_message if 'receiver_message' in fields else 'None'
    accepted = player.accepted

    text = f"{participant}{config_path}{player_name}{real_turn}{who_propose}{offer}{proposer_message}" + \
           f"{proposer_recommendation}{receiver_message}{accepted}"
    text_bytes = text.encode('utf-8')
    hash_object = hashlib.sha256()
    hash_object.update(text_bytes)
    hash_hex = hash_object.hexdigest()
    return str(hash_hex)[:12]


def add_round_tags(text):
    sentences = text.split('\n')
    for i, sentence in enumerate(sentences):
        if 'Round' in sentence:
            sentences[i] = '<u>' + sentence + '</u>'
            break
    return '\n'.join(sentences)


def pretty_html_text(text):
    text = text.strip()
    pattern = r'\$\d*|%\d*|\d+'  # check
    text = re.sub(pattern, lambda match: f'<strong>{match.group()}</strong>', text)  # bold numbers
    text = add_round_tags(text)  # add special tags for the round number
    text = text.replace("\n", " <br> ")  # new lines
    text = text.replace("#", '•')  # bullet points
    return text


def pretty_rules_text(text):
    text = pretty_html_text(text)
    texts = text.split('<br>')
    new_texts = []
    for i, t in enumerate(texts):
        if i == len(texts) - 1:
            new_texts.append(f'<p style="margin-top: 10px;margin-bottom: 10px;">{Constants.instructions_quiz}</p>')
        new_texts.append(f'<p style="margin-top: 10px;margin-bottom: 10px;">{t}</p>')
    text = ''.join(new_texts)
    return text
