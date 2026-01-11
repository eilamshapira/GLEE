import itertools
import json
import os
import random

import streamlit as st
from Levenshtein import distance as levenshtein_distance


def generate_model_pairs(models, runs_per_model):
    """Generate model pairs for the specified runs per model."""
    model_pairs = []

    def get_model_2():
        while True:
            for model in models:
                yield model

    model2_gen = get_model_2()

    for model1 in models:
        for _ in range(runs_per_model):
            model2 = next(model2_gen)
            model_pairs.append((model1, model2))

    return model_pairs


def generate_configurations_without_models(parameters):
    """Generate and return all combinations of game parameters without models."""
    param_keys = [key for key in parameters if isinstance(parameters[key], list)]
    param_combos = list(itertools.product(*(parameters[key] for key in param_keys)))
    configurations = []

    for param_combo in param_combos:
        config = {
            "game_args": {key: param_combo[param_keys.index(key)] for key in param_keys if "player_1_" not in key and "player_2_" not in key},
            "player_1_args": {key.replace("player_1_", ""): param_combo[param_keys.index(key)] for key in param_keys if "player_1_" in key and key != "player_1_delta"},
            "player_2_args": {key.replace("player_2_", ""): param_combo[param_keys.index(key)] for key in param_keys if "player_2_" in key and key != "player_2_delta"},
        }
        # Legacy support: handle player_1_delta and player_2_delta if they exist
        if "player_1_delta" in param_keys:
            config["player_1_args"]["delta"] = param_combo[param_keys.index("player_1_delta")]
        if "player_2_delta" in param_keys:
            config["player_2_args"]["delta"] = param_combo[param_keys.index("player_2_delta")]
        configurations.append(config)

    return configurations


def closest_match(value, options):
    return min(options, key=lambda x: levenshtein_distance(str(value), str(x))) if options else None


def create_config_files(experiment_name, runs_per_model):
    base_path = f"interface_app/generated_experiments/{experiment_name}"
    os.makedirs(base_path, exist_ok=True)

    model_pairs = generate_model_pairs(st.session_state.selected_models, runs_per_model)

    index = 0
    for game_name, parameters in st.session_state.parameters.items():
        configurations = generate_configurations_without_models(parameters)

        def get_configuration():
            while True:
                random.shuffle(configurations)
                for config in configurations:
                    yield config

        config_gen = get_configuration()

        for model_combo in model_pairs:
            config = next(config_gen)
            full_config = {
                "player_1_type": "litellm",
                "player_2_type": "litellm",
                "player_1_args": {
                    "public_name": st.session_state.player_1_name,
                    "model_name": model_combo[0],
                    **config["player_1_args"],
                },
                "player_2_args": {
                    "public_name": st.session_state.player_2_name,
                    "player_id": 3,
                    "model_name": model_combo[1],
                    **config["player_2_args"],
                },
                "game_type": game_name.lower(),
                "experiment_name": experiment_name,
                "game_args": config["game_args"],
            }

            file_path = os.path.join(base_path, f"{game_name}_{index}.json")
            with open(file_path, 'w') as file:
                json.dump(full_config, file, indent=4)

            index += 1
