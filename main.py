from games.game_factory import game_factory
from players.player_factory import player_factory
from utils.utils import read_as_defaultdict, DataLogger
import argparse


def create_players(config):
    """
    Create the players that will play the game
    """
    public_name_player_2 = "the buyer" if (config['game'] == "negotiation" and config.get('is_myopic', False)) else \
        config.get('player_2_name', 'Bob')
    p1_args = config.get('player_1_args', {'public_name': config.get('player_1_name', 'Alice'),
                                           'delta': config.get('player_1_delta', 0.99)})
    p2_args = config.get('player_2_args', {'delta': config.get('player_2_delta', 0.99),
                                           'public_name': public_name_player_2,
                                           'player_id': config.get('player_2_id', 3)})
    p1 = player_factory(config.get('player_1_type', 'terminal'), p1_args)
    if config.get('player_2_type', 'terminal') == 'hf' and config.get('player_1_type', 'terminal') == 'hf':
        if (p1_args.get('model_name', None) == p2_args.get('model_name', None) and
                p1_args.get('model_name', None) is not None):
            p2_args['load_hf_model'] = False
    p2 = player_factory(config.get('player_2_type', 'terminal'), p2_args)
    if not p2_args.get('load_hf_model', True):
        p2.model = p1.model
        p2.tokenizer = p1.tokenizer
    return p1, p2


def create_game(config, player_1, player_2):
    data_logger = DataLogger(player_1, player_2, **config)
    game_factory(config.get('game_type'), player_1, player_2, data_logger,
                 config.get('game_args', {}))


def main(config_path: str = "", n_games: int = 1):
    # Read config from a file
    config = read_as_defaultdict(config_path)

    player_1, player_2 = create_players(config)
    for i in range(n_games):
        create_game(config, player_1, player_2)

    print("Done.")


if __name__ == "__main__":
    print("Running main")
    parser = argparse.ArgumentParser("Game launching")
    parser.add_argument('-c', '--config_path', type=str, help="Path to the configuration of the game we want to run")
    parser.add_argument('-n', '--n_games', type=int, help="Number of games to run with this configuration", default=1)
    args = parser.parse_args()
    main(**vars(args))
