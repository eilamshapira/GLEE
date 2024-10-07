import argparse
from collections import defaultdict
import json
from copy import deepcopy
from consts import *
import string
import random
import pandas as pd
import subprocess
from datetime import datetime


def create_prompts_from_templates(template, info):
    new_prompt = deepcopy(template)
    for key in info:
        new_prompt = new_prompt.replace(key, f"{info[key]}")
    return new_prompt


def read_as_defaultdict(path, default_value=None):
    """
    Reads a json file as a default dict
    """
    with open(path, 'r') as f:
        data = json.load(f)

    def_dic = defaultdict(lambda: default_value)
    for key in data:
        def_dic[key] = data[key]
    return def_dic


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pretty_number(x):
    x = f"{x:,.2f}"
    x = x.rstrip('0').rstrip('.')
    return x


class DataLogger:
    def __init__(self, player_1, player_2, **args):
        self.player_1 = player_1
        self.player_2 = player_2
        self.actions = []
        self.game_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        self.args = args

    def add_action(self, **kwargs):
        player_name, data, round_number = kwargs['player_name'], kwargs['data'], kwargs['round_number']
        data["player"] = player_name
        data["round"] = round_number
        self.actions.append(data)

    def save(self):
        sub_folders = "/".join(self.game_id[:3])
        output_path = f"{OUTPUT_DIR}/{self.args['game_type']}/{sub_folders}/{self.game_id}"
        os.umask(0o002)
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/config.json", 'w') as f:
            # In addition to all arguments, we also save the date and time of the game
            self.args['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # In addition to all arguments, we also save the date and commit hash if exist
            self.args['commit'] = get_commit_hash()
            json.dump(self.args, f)

        player_1_log = pd.DataFrame(self.player_1.history, columns=['prompt', 'response'])
        player_2_log = pd.DataFrame(self.player_2.history, columns=['prompt', 'response'])
        player_1_log.to_csv(f"{output_path}/log_player_1.csv", index=False)
        player_2_log.to_csv(f"{output_path}/log_player_2.csv", index=False)
        actions_df = pd.DataFrame(self.actions)
        actions_df.to_csv(f"{output_path}/game.csv", index=False)


def get_commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return None


def add_ordinal_suffix(n):
    # Special case for 11, 12, 13
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        # Determine the suffix based on the last digit
        last_digit = n % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
        else:
            suffix = "th"
    return f"{n}{suffix}"
