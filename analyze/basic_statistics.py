import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import traceback
from multiprocessing import Pool
from consts import *
import os
import argparse

ERROR_CODE = -1


def get_paths(main_dir, game_id, game_type):
    base_path = os.path.join(game_type, *game_id[:3], f"{game_id}")
    game = os.path.join(main_dir, base_path, "game.csv")
    config = os.path.join(main_dir, base_path, "config.json")
    print(game)
    return {"game": game, "config": config}


def count_games_as_factor_of_llms(configs):
    q = configs.groupby(["player_1_args_model_name", "player_2_args_model_name"]).size().unstack(fill_value=0)
    q.to_csv("output/tables/count_games_as_factor_of_llms.csv")


def correlation_between_first_offer_and_final_gain(configs):
    q = configs.groupby(["player_1_args_model_name", "player_2_args_model_name"]).apply(
        lambda x: x["first_round_offer_alice_gain"].corr(x["alice_final_share"])
    ).unstack(fill_value="-")
    q.to_csv("output/tables/correlation_between_first_offer_and_final_gain.csv")


def process_game(exp_name, game, family):
    return count_messages_and_characters(exp_name, game, family)


def count_messages_and_characters(exp_name, game_id, game_type):
    try:
        paths = get_paths(exp_name, game_id, game_type)
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

def make_saving_path(family, config_files):
    sorted_files = sorted(config_files)
    file_name = "_".join([os.path.basename(f).split(".")[0] for f in sorted_files]) 
    name = f"{family}_{file_name}"
    path = os.path.join("output/basic_statistics", name)
    os.makedirs(path, exist_ok=True)
    return path

def get_data_path_for_family(config_path):
    path = config_path.replace("output/configs/", "Data/")
    path = path.split("_")[:-3]
    path = "_".join(path)
    return path
    


def create_basic_statistics_tables(family, config_files):
    families = ["negotiation", "persuasion", 'bargaining']
    assert family in families, f"Family {family} not in {families}"
    
    saving_path = make_saving_path(family, config_files)
    print(f"Saving path: {saving_path}")
    if os.path.exists(saving_path + "/statistics.csv"):
        print(f"Path {saving_path} already exists. Skipping...")
        return
    
    def get_all_games(config_path):
        # Convert path to use output directory if it doesn't already
        if not config_path.startswith("output/"):
            config_path = f"output/configs/{os.path.basename(config_path)}"
        all_configs = pd.read_csv(config_path)
        games = all_configs["game_id"]
        return games

    games = []
    games_with_family_and_config = []
    for config_file in config_files:
        games = get_all_games(config_file)
        data_dir = get_data_path_for_family(config_file)
        print(f"Family: {family}, Number of games: {len(games)}")
        games_with_family_and_config += [(data_dir, game, family) for game in games]

    num_workers = os.cpu_count() // 4 
    print("num_workers: ", num_workers)
    tick = pd.Timestamp.now()

    with Pool(num_workers) as pool:
        results = []
        for result in tqdm(pool.starmap(process_game, games_with_family_and_config), total=len(games),
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
    df.to_csv(saving_path + "/statistics.csv", index=True)
    tock = pd.Timestamp.now()
    print(f"Time to process {family}: {tock - tick}. Problems: {problems}")
    print()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze game data")
    parser.add_argument("--bargaining", type=str, nargs="+")
    parser.add_argument("--persuasion", type=str, nargs="+")
    parser.add_argument("--negotiation", type=str, nargs="+")
    
    args = parser.parse_args()
    bargaining_files = args.bargaining
    persuasion_files = args.persuasion
    negotiation_files = args.negotiation
    
    if bargaining_files:
        create_basic_statistics_tables("bargaining", bargaining_files)
    if persuasion_files:
        create_basic_statistics_tables("persuasion", persuasion_files)
    if negotiation_files:
        create_basic_statistics_tables("negotiation", negotiation_files)
    

if __name__ == "__main__":
    main()