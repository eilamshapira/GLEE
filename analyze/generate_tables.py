import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import traceback
from multiprocessing import Pool
from consts import *

ERROR_CODE = -1


def get_paths(game_id, game_type):
    if isinstance(game_id, str):
        main_dir = DataStore
    else:
        assert isinstance(game_id, tuple)
        assert len(game_id) == 2
        main_dir = HUMAN_DIR if game_id[1] else DataStore
        game_id = game_id[0]

    base_path = os.path.join(game_type, *game_id[:3], f"{game_id}")
    game = os.path.join(main_dir, base_path, "game.csv")
    config = os.path.join(main_dir, base_path, "config.json")
    vectors = os.path.join(VECTORS_DIR, base_path, "vectors.npy")
    return {"game": game, "vectors": vectors, "config": config}


def count_games_as_factor_of_llms(configs):
    q = configs.groupby(["player_1_args_model_name", "player_2_args_model_name"]).size().unstack(fill_value=0)
    q.to_csv("tables/count_games_as_factor_of_llms.csv")


def correlation_between_first_offer_and_final_gain(configs):
    q = configs.groupby(["player_1_args_model_name", "player_2_args_model_name"]).apply(
        lambda x: x["first_round_offer_alice_gain"].corr(x["alice_final_share"])
    ).unstack(fill_value="-")
    q.to_csv("tables/correlation_between_first_offer_and_final_gain.csv")


def process_game(game, family):
    return count_messages_and_characters(game, family)


def count_messages_and_characters(game_id, game_type):
    try:
        paths = get_paths(game_id, game_type)
        data = pd.read_csv(paths["game"])
        config_file = os.path.join(paths["config"])
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        messages = data["message"].dropna() if "message" in data.columns else pd.Series()
        decisions = data["decision"].dropna() if "decision" in data.columns else pd.Series()
        num_messages = len(messages)
        try:
            total_characters = messages.str.len().sum() if num_messages > 0 else 0
            total_words = messages.str.split().str.len().sum() if num_messages > 0 else 0
        except Exception:
            total_characters = 0
            total_words = 0
        stats = {"total": {"num_messages": num_messages,
                           "total_words": total_words,
                           "total_characters": total_characters,
                           "decisions": len(decisions),
                           "games": 1}}

        for player in data["player"].unique():
            if player == "Nature":
                continue
            player_messages = data[data["player"] == player][
                "message"].dropna() if "message" in data.columns else pd.Series()
            player_decisions = data[data["player"] == player][
                "decision"].dropna() if "decision" in data.columns else pd.Series()
            num_messages = len(player_messages)
            try:
                total_characters = player_messages.str.len().sum() if num_messages > 0 else 0
                total_words = player_messages.str.split().str.len().sum() if num_messages > 0 else 0
            except Exception:
                total_characters = 0
                total_words = 0
            player_model = None
            if player == config_data["player_1_args"]["public_name"]:
                player_model = config_data["player_1_args"]["model_name"]
            elif player == config_data["player_2_args"]["public_name"] or player == "the buyer":
                player_model = config_data["player_2_args"]["model_name"]
            stats[player_model] = {"num_messages": num_messages,
                                   "total_words": total_words,
                                   "total_characters": total_characters,
                                   "decisions": len(player_decisions),
                                   "games": 1}
        return stats

    except Exception as e:
        # print information about the error
        print(f"Error in game {game_id}: {e}")
        traceback.print_exc()
        return ERROR_CODE


def create_basic_statistics_tables():
    families = ["negotiation", "persuasion", 'bargaining']

    def get_all_games(game_type):
        all_configs = pd.read_csv(f"configs/{family}_with_stats.csv")
        # games = set((all_configs["game_id"], all_configs["human_game"]))
        games = list(set(zip(all_configs["game_id"], all_configs["human_game"])))
        return games

    for family in families:
        games = get_all_games(family)
        tick = pd.Timestamp.now()
        print(f"Family: {family}, Number of games: {len(games)}")

        num_workers = os.cpu_count() - 8
        print("num_workers: ", num_workers)
        # m = list([c for c in games if c[1]])[0]
        # count_messages_and_characters(m, family)

        # results = Parallel(n_jobs=num_workers, backend='loky')(
        #     delayed(count_messages_and_characters)(game, family) for game in tqdm(games, desc="Processing games")
        # )
        games_with_family = [(game, family) for game in games]

        with Pool(num_workers) as pool:
            results = []
            for result in tqdm(pool.starmap(process_game, games_with_family), total=len(games),
                               desc="Processing games"):
                results.append(result)

        statistics = defaultdict(lambda: defaultdict(int))
        problems = 0
        for r in tqdm(results):
            if r == ERROR_CODE:
                problems += 1
                continue
            for player in r:
                for k in r[player]:
                    statistics[player][k] += r[player][k]
        print(statistics)
        # Convert the statistics dictionary to a DataFrame
        df = pd.DataFrame(statistics).transpose()

        # Save the DataFrame to a CSV file
        os.makedirs("basic_statistics", exist_ok=True)
        df.to_csv(f'basic_statistics/{family}.csv')
        tock = pd.Timestamp.now()
        print(f"Time to process {family}: {tock - tick}. Problems: {problems}")
        print()


if __name__ == "__main__":
    create_basic_statistics_tables()
