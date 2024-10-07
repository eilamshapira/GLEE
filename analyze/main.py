import pandas as pd
import os
from tqdm import tqdm
import json
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib
import generate_tables
import subprocess
from metrics import calc_metrics
from calc_adj_r2 import calc_adj_r2
from parameters import create_figures
from consts import *

os.makedirs("plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)

matplotlib.use('Agg')  # Use the 'Agg' backend


def load_data(game_type, game_id):
    if isinstance(game_id, str):
        is_human = False
    else:
        assert isinstance(game_id, tuple)
        game_id, is_human = game_id
    if is_human:
        main_folder = HUMAN_DIR
    else:
        main_folder = DataStore
    game_dir = os.path.join(main_folder, game_type, *game_id[:3], game_id)
    game_csv_path = os.path.join(game_dir, "game.csv")
    df = pd.read_csv(game_csv_path)
    return df


def get_stats_bargaining(game_type, game_id):
    df = load_data(game_type, game_id)
    stats = dict()
    stats["rounds_played"] = df["round"].iloc[-1]
    result = df["decision"].iloc[-1]
    stats["result"] = result
    if result == "accept":
        stats["alice_gain"] = df.iloc[-2]["alice_gain"]
        stats["bob_gain"] = df.iloc[-2]["bob_gain"]
    else:
        stats["alice_gain"], stats["bob_gain"] = 0, 0
    stats["first_round_offer_alice_gain"] = df.iloc[0]["alice_gain"]
    return stats


def get_stats_negotiation(game_type, game_id):
    df = load_data(game_type, game_id)
    stats = dict()
    stats["rounds_played"] = df["round"].iloc[-1]
    result = df["decision"].iloc[-1]
    stats["result"] = result
    if result == "AcceptOffer":
        stats["final_price"] = df["product_price"].iloc[-2]
    else:
        stats["final_price"] = np.nan
    stats["first_round_offer"] = df.iloc[0]["product_price"]
    stats["n_decisions"] = len(df["decision"].dropna())
    if "message" in df.columns:
        stats["n_messages"] = len(df["message"].dropna())
    else:
        stats["n_messages"] = np.nan
    return stats


def get_stats_persuasion(game_type, game_id):
    df = load_data(game_type, game_id)
    stats = dict()
    stats["rounds_played"] = df["round"].iloc[-1]
    qualities = df.drop_duplicates("round", keep="first")["round_quality"]
    recommendations = df[df["player"] == "Alice"]["decision"]
    decisions = df.drop_duplicates("round", keep="last")["decision"]
    assert len(qualities) == len(decisions) == stats["rounds_played"]
    for i in range(stats["rounds_played"]):
        stats[f"round_{i + 1}_quality"] = "high" in qualities.iloc[i]
        stats[f"round_{i + 1}_recommendation"] = "yes" in recommendations.iloc[i] if isinstance(recommendations.iloc[i],
                                                                                                str) else np.nan
        stats[f"round_{i + 1}_deal"] = "yes" in decisions.iloc[i]
    # remove pairs where delta in key
    return stats


def is_commit_before(commit1, commit2, repo_path=None):
    def is_repo_up_to_date():
        # Fetch latest changes from the remote
        subprocess.run(['git', 'fetch', 'origin'], check=True)

        # Check the status between local and remote branches
        status = subprocess.run(['git', 'status', '-uno'], capture_output=True, text=True)

        # Check if the output indicates the local branch is ahead/behind
        if "Your branch is up to date" in status.stdout:
            return True
        elif "Your branch is behind" in status.stdout:
            return False
        elif "Your branch is ahead" in status.stdout:
            return False
        else:
            # Handle other cases, like diverged branches
            return None

    # Save the current directory
    original_dir = os.getcwd()

    try:
        # Change to the repository directory
        if repo_path:
            os.chdir(repo_path)

        # check if repo is updated
        if not is_repo_up_to_date():
            # print in red color!
            print(f"\033[91mRepository is not up to date. Please pull the latest changes.\033[0m")

        # Execute the git command to check if commit1 is an ancestor of commit2
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', commit1, commit2],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # If the return code is 0, commit1 is an ancestor of commit2 (so it's before)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return False
    except FileNotFoundError:
        print(f"Repository path {repo_path} not found.")
        return False
    finally:
        # Change back to the original directory
        os.chdir(original_dir)


def create_configs_file(game_type, first_eligible_commit=None):
    data = []

    def process_directory(root):
        for file in os.listdir(root):
            if file == "config.json":
                game_id = root.split("/")[-1]
                config_file = os.path.join(root, "config.json")
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                config_data["game_id"] = game_id
                config_data["human_game"] = HUMAN in root
                return config_data
        return None

    # Collect all directories first
    all_directories = []
    for main_dir in [DataStore, HUMAN_DIR]:
        for root, dirs, files in tqdm(os.walk(os.path.join(main_dir, game_type))):
            all_directories.append(root)

    # Process directories in parallel
    num_cores = multiprocessing.cpu_count() - 8

    results = Parallel(n_jobs=num_cores)(
        delayed(process_directory)(root) for root in tqdm(all_directories)
    )

    # Filter out any None results (in case there were directories without config.json)
    data = [res for res in results if res is not None]

    # Convert the collected data to a DataFrame
    df = pd.DataFrame(data)

    # Function to process each column
    def process_column(col, df):
        if df[col].dtype == "object":
            if df[col].apply(lambda x: isinstance(x, str)).all():
                return None, df[col]  # No changes needed
            else:
                new_df = df[col].apply(pd.Series).add_prefix(col + "_")
                return col, new_df
        return None, df[col]  # Return unchanged if dtype is not "object"

    # Parallel execution of the process_column function
    def parallel_column_processing(df):
        processed_columns = Parallel(n_jobs=-1)(
            delayed(process_column)(col, df) for col in tqdm(df.columns)
        )

        # Reconstruct the DataFrame with the processed columns
        for col, new_df in processed_columns:
            if col:  # If column was split into new columns
                df = pd.concat([df.drop(col, axis=1), new_df], axis=1)

        return df

    # Example of usage:
    df = parallel_column_processing(df)

    commit_col = [c for c in df.columns if "commit" in c][0]
    df[commit_col] = np.where(df["human_game"], "human", df[commit_col])
    if first_eligible_commit:
        print(df[commit_col].unique())
        eligible_commits = {commit: is_commit_before(first_eligible_commit, commit)
                            for commit in df[commit_col].unique() if str(commit) != "nan" and commit != "human"}
        eligible_commits[np.nan] = False
        eligible_commits["human"] = True
        df = df[(df[commit_col].map(eligible_commits))]

    if game_type == "bargaining":
        df["delta_diff"] = (df["player_1_args_delta"] - df["player_2_args_delta"]).apply(lambda x: f"{x:.2f}")

    os.makedirs("configs", exist_ok=True)
    df.to_csv(f"configs/{game_type}.csv", index=False)
    print(f"Saved {len(df)} configs to configs/{game_type}.csv" + (
        f" with first commit {first_eligible_commit}" if first_eligible_commit else ""))
    print(f"That include human data from {len(df[df['human_game']])} games.")


def create_config_with_stats(game_type, configs=None):
    # Define the number of CPUs to use (all available CPUs)
    if configs is None:
        configs = pd.read_csv(f"configs/{game_type}.csv")

    get_stats_func = None
    if game_type == "bargaining":
        get_stats_func = get_stats_bargaining
    elif game_type == "persuasion":
        get_stats_func = get_stats_persuasion
    elif game_type == "negotiation":
        get_stats_func = get_stats_negotiation
    else:
        ValueError(f"Game type {game_type} not supported")

    num_cores = multiprocessing.cpu_count() - 8

    # Wrap the function to show the progress bar
    def process_game_id(game_id):
        return get_stats_func(game_type, game_id)

    # Test the function
    print("Testing the function")
    r = process_game_id((configs["game_id"].iloc[-1], configs["human_game"].iloc[-1]))
    print("Function works")

    # Parallel processing with a progress bar
    results = Parallel(n_jobs=num_cores)(
        delayed(process_game_id)((game_id, human_game)) for game_id, human_game in
        tqdm(zip(configs["game_id"], configs["human_game"]))
    )
    # results is a list of dictionaries, with 4 keys. convert it to columns
    results = pd.DataFrame(results)
    configs = pd.concat([configs, results], axis=1)

    # remove columns where "delta" in column name
    if game_type != "bargaining":
        configs = configs[[col for col in configs.columns if "delta" not in col]]
    if game_type == "negotiation":
        configs.to_csv(f"configs/tmp_{game_type}_with_stats.csv")
        bad_games = (configs["n_messages"] > configs["n_decisions"]) & (configs["n_messages"] > 1)
        print(f"Removing {bad_games.sum()} games with more messages than decisions")
        bad_games = bad_games | ((configs["n_decisions"] > configs["rounds_played"]) & ~configs["human_game"])
        print(f"Removing {bad_games.sum()} games with the last condition and more decisions than rounds played")
        configs = configs[~bad_games]

    print(f"Saved {len(configs)} configs to configs/{game_type}_with_stats.csv")

    def change_otree_names(x):
        if x == "otree":
            return "human"
        if x == "otree_LLM":
            return "gemini-1.5-flash"
        return x

    for player_col in ["player_1_args_model_name", "player_2_args_model_name"]:
        configs[player_col] = configs[player_col].apply(lambda x: change_otree_names(x))
    print(f"Saved {len(configs)} configs to configs/{game_type}_with_stats.csv")

    file_path = f"configs/{game_type}_with_stats.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
    configs.to_csv(file_path)
    return configs


def main():
    first_eligible_commits = {
        # "negotiation": "d7c3790702a2b084383522778da9053e3855f47b",
        # "persuasion": "7b691eb1fa55596b7673add01922dfe0eb0bb3fb",
        # "bargaining": "bee35cfee0d0e9aa299dc64dbe5167020b8db2a2"
    }  # Set them if you want to filter out data from commits before a certain commit
    for game_type in [
        "negotiation",
        "bargaining",
        "persuasion"
    ]:
        create_configs_file(game_type, first_eligible_commit=first_eligible_commits[game_type] if
        game_type in first_eligible_commits else None)  # to create a config file with the data
        create_config_with_stats(game_type)  # calculate statistics for the data, base on configs file created above

    generate_tables.create_basic_statistics_tables()  # Create basic statistics tables
    calc_metrics()  # to train regression models used to analyze the data, and calculate metrics for the data
    calc_adj_r2()  # to calculate adjusted R2 for the regression models
    create_figures()  # to create figures measuring the impact of the different parameters on the game metrics


if __name__ == "__main__":
    main()
