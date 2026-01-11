"""Utilities for discovering experiment runs and materializing config CSVs."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .consts import HUMAN, HUMAN_DIR, DataStore


def slugify_name(value: str) -> str:
    """Return a filesystem-friendly slug while preserving capitalization."""

    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return sanitized

CONFIGS_OUTPUT_DIR = Path("output") / "configs"
CONFIGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _full_data_path(data_path: str = DataStore, exp_name: Optional[str] = None) -> str:
    base = Path(data_path or DataStore)
    if not exp_name:
        return str(base)

    direct = base / exp_name
    slug = slugify_name(exp_name)
    sanitized = base / slug if slug else None

    for candidate in [direct, sanitized]:
        if candidate and candidate.exists():
            return str(candidate)

    # Fall back to direct path for clearer error messages
    return str(direct)


def _output_prefix(data_path: str = DataStore, exp_name: Optional[str] = None) -> str:
    if exp_name:
        slug = slugify_name(exp_name)
        if slug:
            return slug
    full_path = Path(_full_data_path(data_path, exp_name))
    slug = slugify_name(full_path.name)
    return slug or "Data"


def get_configs_csv_path(game_type: str, data_path: str = DataStore, exp_name: Optional[str] = None) -> str:
    prefix = _output_prefix(data_path, exp_name)
    return str(CONFIGS_OUTPUT_DIR / f"{prefix}_{game_type}.csv")


def get_configs_with_stats_path(game_type: str, data_path: str = DataStore, exp_name: Optional[str] = None) -> str:
    prefix = _output_prefix(data_path, exp_name)
    return str(CONFIGS_OUTPUT_DIR / f"{prefix}_{game_type}_with_stats.csv")


def load_data(game_type: str, game_id, main_data_folder: str) -> pd.DataFrame:
    if isinstance(game_id, str):
        is_human = False
    else:
        assert isinstance(game_id, tuple)
        game_id, is_human = game_id
    main_folder = main_data_folder
    game_dir = os.path.join(main_folder, game_type, *game_id[:3], game_id)
    game_csv_path = os.path.join(game_dir, "game.csv")
    return pd.read_csv(game_csv_path)


def get_stats_bargaining(game_type: str, game_id, main_data_folder: str) -> Dict[str, float]:
    df = load_data(game_type, game_id, main_data_folder)
    stats: Dict[str, float] = {}
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


def get_stats_negotiation(game_type: str, game_id, main_data_folder: str) -> Dict[str, float]:
    df = load_data(game_type, game_id, main_data_folder)
    stats: Dict[str, float] = {}
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


def get_stats_persuasion(game_type: str, game_id, main_data_folder: str) -> Dict[str, float]:
    df = load_data(game_type, game_id, main_data_folder)
    stats: Dict[str, float] = {}
    stats["rounds_played"] = df["round"].iloc[-1]
    qualities = df.drop_duplicates("round", keep="first")["round_quality"]
    recommendations = df[df["player"] == "Alice"]["decision"]
    decisions = df.drop_duplicates("round", keep="last")["decision"]
    assert len(qualities) == len(decisions) == stats["rounds_played"]
    for i in range(stats["rounds_played"]):
        stats[f"round_{i + 1}_quality"] = "high" in qualities.iloc[i]
        stats[f"round_{i + 1}_recommendation"] = (
            "yes" in recommendations.iloc[i] if isinstance(recommendations.iloc[i], str) else np.nan
        )
        stats[f"round_{i + 1}_deal"] = "yes" in decisions.iloc[i]
    return stats


def is_commit_before(commit1: str, commit2: str, repo_path: Optional[str] = None) -> bool:
    def is_repo_up_to_date() -> Optional[bool]:
        subprocess.run(['git', 'fetch', 'origin'], check=True)
        status = subprocess.run(['git', 'status', '-uno'], capture_output=True, text=True)
        if "Your branch is up to date" in status.stdout:
            return True
        if "Your branch is behind" in status.stdout:
            return False
        if "Your branch is ahead" in status.stdout:
            return False
        return None

    original_dir = os.getcwd()
    try:
        if repo_path:
            os.chdir(repo_path)
        if not is_repo_up_to_date():
            print("\033[91mRepository is not up to date. Please pull the latest changes.\033[0m")
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', commit1, commit2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as exc:
        print(f"Error occurred: {exc}")
        return False
    except FileNotFoundError:
        print(f"Repository path {repo_path} not found.")
        return False
    finally:
        os.chdir(original_dir)


def _list_game_directories(root: str) -> List[str]:
    if not os.path.exists(root):
        return []
    directories: List[str] = []
    for dirpath, _, _ in os.walk(root):
        directories.append(dirpath)
    return directories


def _process_directory(root: str) -> Optional[Dict]:
    for file in os.listdir(root):
        if file == "config.json":
            game_id = root.split("/")[-1]
            config_file = os.path.join(root, "config.json")
            with open(config_file, "r", encoding="utf-8") as handle:
                config_data = json.load(handle)
            config_data["game_id"] = game_id
            config_data["human_game"] = HUMAN in root
            return config_data
    return None


def _parallel_map_directories(directories: List[str]) -> List[Dict]:
    if not directories:
        return []
    num_cores = max(1, multiprocessing.cpu_count() - 8)
    results = Parallel(n_jobs=num_cores)(delayed(_process_directory)(root) for root in tqdm(directories))
    return [res for res in results if res is not None]


def _expand_dataframe_objects(df: pd.DataFrame) -> pd.DataFrame:
    def process_column(col: str, frame: pd.DataFrame) -> Tuple[Optional[str], pd.DataFrame]:
        if frame[col].dtype == "object":
            if frame[col].apply(lambda x: isinstance(x, str)).all():
                return None, frame[col]
            new_df = frame[col].apply(pd.Series).add_prefix(col + "_")
            return col, new_df
        return None, frame[col]

    processed_columns = Parallel(n_jobs=-1)(delayed(process_column)(col, df) for col in tqdm(df.columns))
    for col, new_df in processed_columns:
        if col:
            df = pd.concat([df.drop(col, axis=1), new_df], axis=1)
    return df


def create_configs_file(
    game_type: str,
    first_eligible_commit: Optional[str] = None,
    data_path: str = DataStore,
    exp_name: Optional[str] = None,
    include_human: bool = True,
) -> str:
    full_data_path = _full_data_path(data_path, exp_name)
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"Data path '{full_data_path}' does not exist.")

    data_roots = [os.path.join(full_data_path, game_type)]
    if include_human:
        human_root = os.path.join(HUMAN_DIR, game_type)
        if os.path.exists(human_root):
            data_roots.append(human_root)

    all_directories: List[str] = []
    for main_dir in data_roots:
        all_directories.extend(_list_game_directories(main_dir))

    data = _parallel_map_directories(all_directories)
    if not data:
        raise FileNotFoundError(
            f"No completed {game_type} games were found in '{full_data_path}'. "
            "Confirm that the experiment finished and data files exist."
        )

    df = pd.DataFrame(data)
    df = _expand_dataframe_objects(df)

    commit_candidates = [c for c in df.columns if "commit" in c]
    if commit_candidates:
        commit_col = commit_candidates[0]
        df[commit_col] = np.where(df["human_game"], "human", df[commit_col])
        if first_eligible_commit:
            eligible_commits = {
                commit: is_commit_before(first_eligible_commit, commit)
                for commit in df[commit_col].unique()
                if str(commit) != "nan" and commit != "human"
            }
            eligible_commits[np.nan] = False
            eligible_commits["human"] = True
            df = df[df[commit_col].map(eligible_commits)]

    if game_type == "bargaining":
        # Handle both old format (player_1_args_delta) and new format (game_args_delta_1)
        if "game_args_delta_1" in df.columns and "game_args_delta_2" in df.columns:
            df["delta_diff"] = (df["game_args_delta_1"] - df["game_args_delta_2"]).apply(lambda x: f"{x:.2f}")
        elif "player_1_args_delta" in df.columns and "player_2_args_delta" in df.columns:
            df["delta_diff"] = (df["player_1_args_delta"] - df["player_2_args_delta"]).apply(lambda x: f"{x:.2f}")

    config_path = Path(get_configs_csv_path(game_type, data_path=data_path, exp_name=exp_name))
    df.to_csv(config_path, index=False)
    print(f"Saved {len(df)} configs to {config_path}")
    return str(config_path)


def create_config_with_stats(
    game_type: str,
    data_path: str = DataStore,
    exp_name: Optional[str] = None,
) -> str:
    config_path = Path(get_configs_csv_path(game_type, data_path=data_path, exp_name=exp_name))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' does not exist. Run create_configs_file first.")

    configs = pd.read_csv(config_path)
    main_data_folder = _full_data_path(data_path, exp_name)

    if game_type == "bargaining":
        get_stats_func = get_stats_bargaining
    elif game_type == "persuasion":
        get_stats_func = get_stats_persuasion
    elif game_type == "negotiation":
        get_stats_func = get_stats_negotiation
    else:
        raise ValueError(f"Game type {game_type} not supported")

    def process_game_id(game_id, human_game):
        return get_stats_func(game_type, (game_id, human_game), main_data_folder=main_data_folder)

    num_cores = max(1, multiprocessing.cpu_count() - 8)
    pairs = list(zip(configs["game_id"], configs["human_game"]))
    if not pairs:
        raise ValueError(
            f"Config file '{config_path}' does not contain any games. Did data generation finish successfully?"
        )

    results = Parallel(n_jobs=num_cores)(
        delayed(process_game_id)(game_id, human_game)
        for game_id, human_game in tqdm(pairs, total=len(pairs))
    )
    results_df = pd.DataFrame(results)
    configs = pd.concat([configs, results_df], axis=1)

    if game_type != "bargaining":
        configs = configs[[col for col in configs.columns if "delta" not in col]]
    if game_type == "negotiation":
        tmp_path = Path(get_configs_with_stats_path(game_type, data_path=data_path, exp_name=exp_name)).with_name(
            f"tmp_{Path(config_path).stem}_with_stats.csv"
        )
        configs.to_csv(tmp_path)
        bad_games = (configs["n_messages"] > configs["n_decisions"]) & (configs["n_messages"] > 1)
        print(f"Removing {bad_games.sum()} games with more messages than decisions")
        bad_games = bad_games | ((configs["n_decisions"] > configs["rounds_played"]) & ~configs["human_game"])
        print(f"Removing {bad_games.sum()} games with inconsistent decisions vs. rounds")
        configs = configs[~bad_games]

    def change_otree_names(name: str):
        if name == "otree":
            return "human"
        if name == "otree_LLM":
            return "gemini-1.5-flash"
        return name

    for player_col in ["player_1_args_model_name", "player_2_args_model_name"]:
        if player_col in configs.columns:
            configs[player_col] = configs[player_col].apply(change_otree_names)

    stats_path = Path(get_configs_with_stats_path(game_type, data_path=data_path, exp_name=exp_name))
    if stats_path.exists():
        stats_path.unlink()
    configs.to_csv(stats_path, index=False)
    print(f"Saved {len(configs)} configs to {stats_path}")
    return str(stats_path)