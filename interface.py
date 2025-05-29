import streamlit as st
import itertools
import os
import json
import re
import sys
import threading
from subprocess import Popen
import random
import time
import analyze.main as analyze_main
import pandas as pd
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define model list
models = [
    "xai/grok-2-1212", "gpt-4o", "o3-mini", "gpt-4o-mini",
    "vertex_ai/mistral-large-2411", "vertex_ai/meta/llama-3.3-70b-instruct-maas", "vertex_ai/meta/llama-3.1-405b-instruct-maas",
    "vertex_ai/gemini-1.5-pro",
    "vertex_ai/gemini-1.5-flash", "vertex_ai/gemini-2.0-flash", "vertex_ai/gemini-2.0-flash-lite",
    "vertex_ai/claude-3-7-sonnet@20250219", "vertex_ai/claude-3-5-sonnet-v2@20241022"

]

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'model_selection'
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'parameters' not in st.session_state:
    st.session_state.parameters = {}

# Semaphore to limit concurrent threads
sem = threading.Semaphore(50)

def get_range_input(label, min_val=0.0, max_val=100.0, step=10.0, game_name=""):
    start, stop = st.slider(f"{label} Range", min_value=min_val, max_value=max_val, value=(min_val, max_val), step=step, key=f"{game_name}_{label}_range")
    interval = st.number_input(f"{label} Interval", min_value=0.01, max_value=max_val, value=step, step=0.01, key=f"{game_name}_{label}_interval")
    
    values = []
    current = start
    while current <= stop:
        values.append(round(current, 2))
        current += interval
    
    return values

def get_numeric_input(label, min_val=0.0, max_val=1000.0, step=10.0, game_name="", default_options=[], astype=int):
    # mode = st.radio(
    #     f"{label} Input Mode",
    #     options=["Manual Values", "Range"],
    #     key=f"{game_name}_{label}_mode"
    # )
    
    mode = "Manual Values"

    if mode == "Range":
        start, stop = st.slider(
            f"{label} Range", min_value=min_val, max_value=max_val,
            value=(min_val, max_val), step=step,
            key=f"{game_name}_{label}_range"
        )
        interval = st.number_input(
            f"{label} Interval", min_value=0.01, max_value=max_val,
            value=step, step=0.01, key=f"{game_name}_{label}_interval"
        )

        values = []
        current = start
        while current <= stop:
            values.append(round(current, 2))
            current += interval

        return values

    else:  
        values_key = f"{game_name}_{label}_values"
        edit_mode_key = f"{game_name}_{label}_edit_mode"
        new_input_key = f"{game_name}_{label}_new_input"
        checkbox_key = f"{game_name}_{label}_checkbox"

        if values_key not in st.session_state:
            st.session_state[values_key] = sorted(set(default_options))

        if edit_mode_key not in st.session_state:
            st.session_state[edit_mode_key] = False

        if not st.session_state[edit_mode_key]:
            options = st.session_state[values_key] + ["add..."]
            selected = st.multiselect(
                f"{label}:",
                options=options,
                default=st.session_state[values_key],
                key=checkbox_key
            )

            if "add..." in selected:
                st.session_state[edit_mode_key] = True
                st.rerun()

            return [astype(v) for v in selected if v != "add..."]

        else:
            new_values_str = st.text_input(
                f"Enter new {label} values (comma-separated):",
                key=new_input_key
            )

            if new_values_str.strip():
                try:
                    new_values = sorted(set(astype(x.strip()) for x in new_values_str.split(',')))
                    st.session_state[values_key] = sorted(set(st.session_state[values_key]).union(new_values))
                    st.session_state[edit_mode_key] = False
                    st.session_state.pop(new_input_key, None)
                    st.rerun()
                except ValueError:
                    st.error(f"Please enter valid numeric ({astype.__name__}) values separated by commas.")

            return []



def get_bool_input(label, game_name=""):
    options = ['True', 'False', 'Both']
    choice = st.selectbox(f"{label}", options, key=f"{game_name}_{label}")
    return [True, False] if choice == 'Both' else [choice == 'True']

def get_bool_multiselect(label, game_name="", options=[True, False], default_options=None):
    if default_options is None:
        default_options = options
    selected_options = st.multiselect(f"{label}", options, default=default_options, key=f"{game_name}_{label}")
    return selected_options

def get_num_input(label, game_name=""):
    choice =  st.number_input("Max rounds", min_value=1, max_value=100, value=10, step=1, key=f"max_rounds")
    return [choice]

def game_parameters(game_name):

    player_1_name = st.text_input("Player 1 Public Name", key=f"{game_name}_p1", value="Alice")
    player_2_name = st.text_input("Player 2 Public Name", key=f"{game_name}_p2", value="Bob")

    st.session_state.player_1_name = player_1_name
    st.session_state.player_2_name = player_2_name

    params = {}

    if game_name == "Persuasion":
        params.update({
            'p': get_numeric_input(f"{game_name} p", game_name=game_name, default_options=[1/3, 0.5, 0.8], astype=float),
            'v': get_numeric_input(f"{game_name} v", game_name=game_name, default_options=[1.2, 1.25, 2, 3, 4], astype=float),
            'c': get_numeric_input(f"{game_name} c", game_name=game_name, default_options=[0], astype=float),
            'product_price': get_numeric_input(f"{game_name} Product Price", min_val=0.001, max_val=1000000000, game_name=game_name, default_options=[100,1_00_00,1_00_00_00]),
            'total_rounds': get_numeric_input(f"{game_name} Total Rounds", game_name=game_name, default_options=[20]),
            'is_seller_know_cv': get_bool_multiselect(f"{game_name} Is Seller Know CV", game_name=game_name),
            'seller_message_type': get_bool_multiselect(f"{game_name} Seller Message Type", game_name=game_name, options=['text', 'binary']),
            'is_myopic': get_bool_multiselect(f"{game_name} Is Myopic", game_name=game_name),
        })

    elif game_name == "Bargaining":
        params.update({
            'player_1_delta': get_numeric_input(f"{game_name} Player 1 Delta", min_val=0.0, max_val=1.0, step=0.1, game_name=game_name, default_options=[0.8,0.9,0.95,1], astype=float),
            'player_2_delta': get_numeric_input(f"{game_name} Player 2 Delta", min_val=0.0, max_val=1.0, step=0.1, game_name=game_name, default_options=[0.8,0.9,0.95,1], astype=float),
            'money_to_divide': get_numeric_input(f"{game_name} Product Price", min_val=0.0, max_val=1000.0, game_name=game_name, default_options=[100,1_00_00,1_00_00_00]),
            'max_rounds': get_numeric_input(f"{game_name} Max Rounds", min_val=0, max_val=99, game_name=game_name, default_options=[12,25], astype=int),
            'complete_information': get_bool_multiselect(f"{game_name} Complete Information", game_name=game_name),
            'messages_allowed': get_bool_multiselect(f"{game_name} Messages Allowed", game_name=game_name),
        })

    elif game_name == "Negotiation":
        params.update({
            'seller_value': get_numeric_input(f"{game_name} Seller Value", min_val=0.0, max_val=1000.0, game_name=game_name, default_options=[0.8, 1, 1.2, 1.5], astype=float),
            'buyer_value': get_numeric_input(f"{game_name} Buyer Value", min_val=0.0, max_val=1000.0, game_name=game_name, default_options=[0.8, 1, 1.2, 1.5], astype=float),
            'product_price_order': get_numeric_input(f"{game_name} Product Price Order", min_val=0.0, max_val=10.0, step=1.0, game_name=game_name, default_options=[100,1_00_00,1_00_00_00]),
            'max_rounds': get_numeric_input(f"{game_name} Max Rounds", min_val=0, max_val=99, game_name=game_name, default_options=[1,10,25], astype=int),
            'complete_information': get_bool_multiselect(f"{game_name} Complete Information", game_name=game_name),
            'messages_allowed': get_bool_multiselect(f"{game_name} Messages Allowed", game_name=game_name),            
        })
    # print total number of configurations
    total_configurations = 1
    for param in params.values():
        if isinstance(param, list):
            total_configurations *= len(param)
        else:
            total_configurations *= 1
    if total_configurations>0:
        st.write(f"Total configurations: {total_configurations}.")
        each_model_runs_per_config = st.session_state.run_per_model / total_configurations
        st.write(f"Each model will play each configuration {each_model_runs_per_config:.2f} times ({each_model_runs_per_config/2:.2f} as {player_1_name}, {each_model_runs_per_config/2:.2f} as {player_2_name}).")
        st.write(f"Each configuration will be played {len(st.session_state.selected_models) * st.session_state.run_per_model / total_configurations:.2f} times.")
    
    st.session_state.parameters[game_name] = {**params}


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
    # print(param_combos)
    configurations = []

    for param_combo in param_combos:
        config = {
            "game_args": {key: param_combo[param_keys.index(key)] for key in param_keys if "player_1_" not in key and "player_2_" not in key},
            "player_1_args": {key.replace("player_1_", ""): param_combo[param_keys.index(key)] for key in param_keys if "player_1_" in key},
            "player_2_args": {key.replace("player_2_", ""): param_combo[param_keys.index(key)] for key in param_keys if "player_2_" in key},
        }
        configurations.append(config)

    return configurations

def closest_match(value, options):
    return min(options, key=lambda x: levenshtein_distance(str(value), str(x))) if options else None

def create_config_files(experiment_name, runs_per_model):
    base_path = f"large_scale_experiments/{experiment_name}"
    os.makedirs(base_path, exist_ok=True)
    
    model_pairs = generate_model_pairs(st.session_state.selected_models, runs_per_model)
    
    index = 0
    for game_name, parameters in st.session_state.parameters.items():
        # Generate configurations without models
        configurations = generate_configurations_without_models(parameters)
        
        def get_configuration():
            while True:
                random.shuffle(configurations)
                for config in configurations:
                    yield config
                
        config_gen = get_configuration()
            
        # Assign model pairs to configurations
        for model_combo in model_pairs:
            config = next(config_gen)
            full_config = {
                "player_1_type": "litellm",
                "player_2_type": "litellm",
                "player_1_args": {
                    "public_name": st.session_state.player_1_name,
                    "model_name": model_combo[0],
                    **config["player_1_args"]
                },
                "player_2_args": {
                    "public_name": st.session_state.player_2_name,
                    "player_id": 3,
                    "model_name": model_combo[1],
                    **config["player_2_args"]
                },
                "game_type": game_name.lower(),
                "experiment_name": experiment_name,
                "game_args": config["game_args"],
            }

            file_path = os.path.join(base_path, f"{game_name}_{index}.json")
            with open(file_path, 'w') as file:
                json.dump(full_config, file, indent=4)

            index += 1

def run_experiment(file_path, n_games=1):
    sem.acquire()

    try:
        if '_done' not in file_path:
            process = Popen(['python', 'main.py', '--config_path', file_path, '--n_games', str(n_games)])
            return_code = process.wait()
            if return_code == 0:
                new_file_path = file_path.replace('.json', '_done.json')
                os.rename(file_path, new_file_path)
                print(f"Experiment {file_path} completed successfully.")
    finally:
        sem.release()
        

def list_existing_experiments():
    """List existing experiments from the 'large_scale_experiments' directory."""
    base_path = "large_scale_experiments"
    if not os.path.exists(base_path):
        return []
    return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]


def list_existing_results():
    """List existing experiments from the 'large_scale_experiments' directory."""
    base_path = "Data"
    if not os.path.exists(base_path):
        return []
    return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]


def continue_experiment(experiment_name):
    """Handles continuation of an existing experiment."""
    base_path = os.path.join("large_scale_experiments", experiment_name)
    config_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.json')]

    st.success(f"Continuing experiment '{experiment_name}'... Check your logs for details.")

    new_count = 1
    for file_path in config_files:
        print(new_count)
        threading.Thread(target=run_experiment, args=(file_path, 1)).start()
        new_count += 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        done_files = [f for f in os.listdir(base_path) if f.endswith('_done.json')]
        done_count = len(done_files)
        progress = done_count / len(config_files)
        progress_bar.progress(progress)
        status_text.text(f"{done_count}/{len(config_files)} experiments completed")
        
        if done_count == len(config_files):
            break
        
        time.sleep(1)    
    status_text.text("All experiments completed!")

    st.stop()


def main():
    st.title("GLEE Experimentation Platform")

    if 'initial_choice_made' not in st.session_state:
        st.session_state.initial_choice_made = False

    if not st.session_state.initial_choice_made:
        choice = st.radio("Choose an Option", ("Run New Experiment", "Continue Existing Experiment", "Analyze Results"))

        if st.button("Proceed"):
            st.session_state.initial_choice_made = True
            st.session_state.experiment_mode = choice
            st.rerun()
    
    else:
        if st.session_state.experiment_mode == "Run New Experiment":
            if st.session_state.step == 'model_selection':
                st.subheader("Select Models to Run")
                st.session_state.selected_game = st.selectbox("Select Game to Configure", ["Bargaining", "Persuasion", "Negotiation"])
                models_choice = st.multiselect("Choose models", options=models)
                runs_per_model = st.number_input("Runs per models", min_value=1, max_value=100000, value=500, step=100, key=f"num_per_model")
                total_runs = len(models_choice) * runs_per_model
                if models_choice:
                    st.write(f"Each model will play {runs_per_model} times as Player 1 and {runs_per_model} times as Player 2.")
                    st.write(f"Total number of games: {total_runs:,}")
                    st.write(f"Each pair of models will play {total_runs / (len(models_choice) ** 2):.2f} games.")
                if st.button("Confirm Models"):
                    st.session_state.selected_models = models_choice
                    st.session_state.run_per_model = runs_per_model
                    st.session_state.step = 'configure_game'
                    st.query_params['step'] = 'configure_game'
                    st.rerun()

            elif st.session_state.step == 'configure_game':
                game_name = st.session_state.selected_game
                st.subheader(f"Configure {game_name} Parameters")
                game_parameters(game_name)

                if st.button("Save and Go Back"):
                    st.session_state.step = 'game_parameters_selection'
                    st.query_params['step'] = 'game_parameters_selection'
                    st.rerun()


                if st.button("Confirm Parameters"):
                    st.session_state.step = 'confirmation'
                    st.query_params['step'] = 'confirmation'
                    st.rerun()

            elif st.session_state.step == 'confirmation':
                st.subheader("Confirm Experiment Setup")
                st.write(f"Total number of runs: {len(st.session_state.parameters) * st.session_state.run_per_model * len(st.session_state.selected_models)}")
                st.write(st.session_state.parameters)

                experiment_name = st.text_input("Enter an experiment name:")
                sanitized_name = re.sub(r'\W+', '', experiment_name)

                if st.button("Start Experiment") and sanitized_name:
                    
                    create_config_files(sanitized_name, st.session_state.run_per_model)

                    base_path = f"large_scale_experiments/{sanitized_name}"
                    config_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.json')]

                    st.success("Running experiments... Check your logs for details.")
                    new_count = 1
                    for file_path in config_files:
                        print(new_count)
                        threading.Thread(target=run_experiment, args=(file_path, 1)).start()
                        new_count += 1
                        
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while True:
                        done_files = [f for f in os.listdir(base_path) if f.endswith('_done.json')]
                        done_count = len(done_files)
                        progress = done_count / len(config_files)
                        progress_bar.progress(progress)
                        status_text.text(f"{done_count}/{len(config_files)} experiments completed")
                        
                        if done_count == len(config_files):
                            break
                        
                        time.sleep(1)    
                    status_text.text("All experiments completed!")
                    st.stop()

        elif st.session_state.experiment_mode == "Continue Existing Experiment":
            existing_experiments = list_existing_experiments()
            if existing_experiments:
                experiment_name = st.selectbox("Select an Experiment to Continue", existing_experiments)
                sanitized_name = re.sub(r'\W+', '', experiment_name)
                base_path = f"large_scale_experiments/{sanitized_name}"
                done_count = len([f for f in os.listdir(base_path) if f.endswith('_done.json')])
                config_count = len([f for f in os.listdir(base_path) if f.endswith('.json')])
                
                st.write(f"Selected Experiment: {experiment_name} ({done_count}/{config_count} completed)")
                if st.button("Continue", disabled=(done_count==config_count)):
                    continue_experiment(experiment_name)
            else:
                st.info("No existing experiments found.")
                st.stop()
                
        elif st.session_state.experiment_mode == "Analyze Results":
            if st.session_state.step == 'model_selection':
                st.subheader("Analyze Results")
                existing_experiments = list_existing_results()
                if existing_experiments:
                    experiment_names = st.multiselect("Select Experiments", existing_experiments)
                    market_mode = st.checkbox("Market Mode", value=False)
                    models_interaction = st.checkbox("Models Interaction", value=False)
                    myopic_split = st.checkbox("Myopic Split", value=False)
                    # st.info(experiment_names)
                    show_continue = True
                    if st.button("Analyze Results", disabled=(len(experiment_names)==0)):
                        st.write("Analyzing results...")
                        all_exp_names = []
                        config_files = defaultdict(list)
                        for experiment_name in experiment_names:
                            sanitized_name = re.sub(r'\W+', '', experiment_name)
                            all_exp_names.append(sanitized_name)
                            base_path = f"Data/{sanitized_name}"
                            done_count = len([f for f in os.listdir(base_path) if f.endswith('_done.json')])
                            config_count = len([f for f in os.listdir(base_path) if f.endswith('.json')])
                            
                            game_types = [game_type for game_type in ["bargaining", "persuasion", "negotiation"] if os.path.exists(os.path.join(base_path, f"{game_type}"))]
                            for game_type in game_types:
                                config_path = f"output/configs/Data_{experiment_name}_{game_type}_with_stats.csv"
                                if not os.path.exists(config_path):
                                    analyze_main.create_configs_file(game_type, first_eligible_commit=None, data_path="Data", exp_name=experiment_name, include_human=False)
                                    analyze_main.create_config_with_stats(game_type, data_path="Data", exp_name=experiment_name)
                                    st.write(f"Created config file for **{game_type}** games in **{config_path}** in the experiment **{experiment_name}**")
                                else:
                                    st.write(f"Config file for **{game_type}** games in the experiment **{experiment_name}** already exists in **{config_path}**")
                                config_files[game_type].append(config_path)
                        configs_files_str = []
                        for game_family, file_paths in config_files.items():
                            if len(file_paths) > 0:
                                configs_files_str.append(f"--{game_family}")
                                configs_files_str.extend(file_paths)
                        all_exp_names = "_".join(sorted(all_exp_names))
                        if market_mode:
                            configs_files_str.append("--merge_features=market")
                            all_exp_names += "_markets"
                            st.query_params['merge_features'] = 'market'
                        if myopic_split:
                            configs_files_str.append("--myopic_split")
                            all_exp_names += "_myopic_split"
                            st.query_params['myopic_split'] = True
                        if models_interaction:
                            configs_files_str.append("--models_interaction")
                            all_exp_names += "_models_interaction"
                            st.query_params['models_interaction'] = True
                        return_code = 0
                        if not os.path.exists(f"output/analyze_coefs/{all_exp_names}.csv"):
                            process = Popen(['python', 'analyze/ML.py', '--exp_name', all_exp_names, *configs_files_str])
                            return_code = process.wait()
                        if return_code == 0:
                            st.session_state.step = "Results"
                            st.query_params['step'] = 'Results'
                            st.query_params['final_exp_name'] = all_exp_names
                            st.query_params['config_files'] = json.dumps(config_files)
                            st.rerun()
                        else:
                            st.error("Error running the analysis. Please check your logs.")
                    
                else:
                    st.info("No existing experiments found.")
                    if st.button("Back to Main Menu"):
                        st.session_state.initial_choice_made = False
                        st.session_state.step = 'model_selection'
                        st.session_state.selected_models = []
                        st.session_state.parameters = {}
                        st.rerun()
            
            elif st.session_state.step == "Statistics" or st.session_state.step == "ShowStatistics":
                st.subheader("Statistics")
                if st.session_state.step == "Statistics":
                    st.write("Statistics will be shown here.")
                    paths = json.loads(st.query_params['config_files'])
                    process = []
                    for family in paths.keys():
                        new_process = Popen(['python', 'analyze/basic_statistics.py', f'--{family}', *paths[family]])
                        process.append(new_process)
                    # wait for all processes to finish
                    return_codes = [p.wait() for p in process]
                    # check return codes
                    for return_code in return_codes:
                        if return_code != 0:
                            st.error("Error running the analysis. Please check your logs.")
                            break
                    else:
                        st.write("Statistics generated successfully.")
                        st.session_state.step = "ShowStatistics"
                        st.rerun()
                if st.session_state.step == "ShowStatistics":
                    configs_paths = json.loads(st.query_params['config_files'])
                    def get_statistic_path(family, configs_paths):
                        # Handle list of config paths
                        stat_path = ""
                        for path in sorted(configs_paths):
                            if stat_path:
                                stat_path += "_"
                            # Convert path to use output directory if it doesn't already
                            if not path.startswith("output/"):
                                path = f"output/configs/{os.path.basename(path)}"
                            tmp_path = path.replace("output/configs/", "")
                            tmp_path = tmp_path.split(".")[:-1]
                            tmp_path = ".".join(tmp_path)
                            stat_path += tmp_path
                        
                        stat_path = f"output/basic_statistics/{family}_" + stat_path
                        stat_path += "/statistics.csv"
                        return stat_path
                    statistics = []
                    for family, family_configs in configs_paths.items():
                        statistic_file = get_statistic_path(family, family_configs)
                        family_data = pd.read_csv(statistic_file, index_col=0)
                        # set index col to named "model" and reset index
                        family_data = family_data.reset_index()
                        family_data = family_data.rename(columns={"index": "model"})
                        # drop the row "total"                        
                        family_data["family"] = family
                        statistics.append(family_data)
                    statistics = pd.concat(statistics, ignore_index=True)
                    
                    choice = st.radio("Groupping Options", ("Group by Family", "Group by Game Type"))
                    groupping_row = st.checkbox("Show totals")
                    st.write(choice)
                    
                    if choice == "Group by Family":
                        grouped_statistics = statistics[statistics["model"] == "total"].set_index("family")
                        grouped_statistics = grouped_statistics.drop(columns=["model"])
                        
                    if choice == "Group by Game Type":
                        grouped_statistics = statistics[statistics["model"] != "total"].groupby("model").sum()
                        grouped_statistics = grouped_statistics.drop(columns=["family"])
                        
                    if groupping_row:
                        total_row = grouped_statistics.sum()
                        # total_row["model"] = "total"
                        # total_row["family"] = "total"
                        total_row = total_row.rename("total")
                        grouped_statistics = pd.concat([grouped_statistics, total_row.to_frame().T], ignore_index=False)
                    grouped_statistics = grouped_statistics.applymap(lambda x: f"{x:,}" if isinstance(x, (int, float)) else x)

                    st.write(grouped_statistics.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))
                    st.write("###### Statistics Table")                    
                    
                if st.sidebar.button("Back to Results"):
                    st.session_state.step = "Results"
                    st.query_params['step'] = 'Results'
                    st.rerun()

            elif st.session_state.step == "Results":
                @st.cache_data
                def load_data(exp_name):
                    df = pd.read_csv(f"output/analyze_coefs/{exp_name}.csv")
                    return df

                coef_df = load_data(st.query_params['final_exp_name'])

                # Sidebar selections
                # show the value of the selected parameter in the sidebar
                st.sidebar.header("Selected Parameters")
                st.sidebar.write("Market Mode:", st.query_params['merge_features'] if 'merge_features' in st.query_params else "False")
                st.sidebar.write("Myopic Split:", st.query_params['myopic_split'] if 'myopic_split' in st.query_params else "False")
                st.sidebar.write("Models Interaction:", st.query_params['models_interaction'] if 'models_interaction' in st.query_params else "False")
                
                # add "show statistics" button
                if st.sidebar.button("Show Statistics"):
                    st.session_state.step = "Statistics"
                    st.rerun()
                
                # st.sidebar.write("Game Type:", st.query_params['final_exp_name'])
                
                st.sidebar.header("Filter Options")
                selected_family = st.sidebar.selectbox("Select Family", sorted(coef_df["family"].unique()))
                filtered_data = coef_df[coef_df["family"] == selected_family]

                selected_metric = st.sidebar.selectbox("Select Metric", sorted(filtered_data["metric"].unique()))
                filtered_data = filtered_data[filtered_data["metric"] == selected_metric]

                param_options = sorted(filtered_data["paramter_coef"].unique())
                if 'selected_param' not in st.session_state:
                    st.session_state.selected_param = param_options[0]
                if st.session_state.selected_param not in param_options:
                    st.session_state.selected_param = closest_match(st.session_state.selected_param, param_options)
                selected_param = st.sidebar.selectbox(
                    "Select Parameter Coefficient",
                    param_options,
                    index=param_options.index(st.session_state.selected_param) if st.session_state.selected_param in param_options else 0,
                    key="selected_param"
                )
                filtered_data = filtered_data[filtered_data["paramter_coef"] == selected_param]

                # Sorting and filtering
                # filtered_data = filtered_data[(filtered_data["games_per_model"] == 5000) & (filtered_data["seed"] == 2)]
                filtered_data = filtered_data.sort_values("effect", ascending=False)[["value", "effect" ,"ci_low","ci_high"]]

                # Display table
                st.write(f"##### Effect of {selected_param} on {selected_metric} in {selected_family} Games")
                st.write("###### Ranking Table")
                
                print(selected_param)
                if selected_param == "models_names":
                    tmp_df = filtered_data.copy()
                    tmp_df["value"] = tmp_df["value"].str.replace("alice_", "")
                    tmp_df[["model_a", "model_b"]] = tmp_df["value"].str.split("_bob_", expand=True, n=1)
                    tmp_df = tmp_df.pivot(index="model_a", columns="model_b", values="effect")
                    tmp_df -= tmp_df.mean().mean()
                    styled_df = tmp_df.style.background_gradient(cmap='coolwarm', axis=None)

                    styles = []
                    for i in range(tmp_df.shape[1]):
                        styles.extend([
                            {"selector": f"th.col{i}", "props": [("min-width", "3ch"), ("max-width", "3ch")]},
                            {"selector": f"td.col{i}", "props": [("min-width", "3ch"), ("max-width", "3ch")]}
                        ])

                    styled_df = styled_df.set_table_styles(styles)
                    st.dataframe(styled_df) 
                    
                    # שונות לכל שורה (לפי model_a)
                    mean_value = tmp_df.mean().mean()
                    st.write("Mean value")
                    st.write(mean_value)
                    vlaue_ii = np.mean(np.array([tmp_df.iloc[i, i] for i in range(tmp_df.shape[0])]))
                    st.write("Value II")
                    st.write(vlaue_ii)

                       
                if selected_param == "market":
                    tmp_df = filtered_data.copy()
                    def split_kv_string(s):
                        return dict(part.split("=") for part in s.split("_"))
                    
                    expanded = tmp_df['value'].apply(split_kv_string).apply(pd.Series)

                    tmp_df = pd.concat([expanded, tmp_df.drop("value", axis=1)], axis=1)

                    for col in expanded.columns:
                        if set(tmp_df[col].dropna().unique()) <= {"True", "False"}:
                            tmp_df[col] = tmp_df[col].map({"True": True, "False": False})

                    st.dataframe(tmp_df.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))

                else:
                    st.dataframe(filtered_data.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))
                

                if st.button("Export to LaTeX"):
                    if selected_param == "models_names":
                        # Process data
                        export_df = tmp_df.copy()

                        # Column numbers
                        col_ids = list(range(1, len(export_df.columns) + 1))
                        export_df.columns = [f"\\makebox[8pt]{{{i:X}}}" for i in col_ids]

                        # Row numbers
                        row_ids = list(range(1, len(export_df.index) + 1))
                        export_df.index = [f"{i:X}. {name}" if selected_metric != "fairness" else f"{i:X}." for i, name in zip(row_ids, export_df.index)]

                        # Colormap normalization
                        cmap = cm.get_cmap('coolwarm')
                        norm = mcolors.Normalize(vmin=export_df.min().min(), vmax=export_df.max().max())

                        # LaTeX Table
                        latex_core = (
                            "\\noindent\n"
                            "\\begin{minipage}{\\linewidth}\n"
                            "\\centering\n"
                            "\\renewcommand{\\arraystretch}{1}\n"
                            "\\setlength{\\tabcolsep}{1pt}\n"
                            "\\begin{tabular}{l" + "c" * len(export_df.columns) + "}\n"
                            " & " + " & ".join(export_df.columns) + " \\\\\n\\hline\n"
                        )

                        for idx, row in export_df.iterrows():
                            line = [f"{idx}"]
                            for val in row:
                                if pd.isna(val):
                                    cell = ""
                                else:
                                    r, g, b = [int(255 * c) for c in cmap(norm(val))[:3]]
                                    cell = f"\\cellcolor[RGB]{{{r},{g},{b}}}"
                                line.append(cell)
                            latex_core += " & ".join(line) + " \\\\\n"

                        # Colorbar
                        vmin = export_df.values.min()
                        vmax = export_df.values.max()
                        latex_core += (
    "& \\multicolumn{" + str(len(export_df.columns)) + "}{c}{\n"
    "\\begin{tikzpicture}\n"
    "\\begin{axis}[\n"
    "hide axis,\n"
    "scale only axis,\n"
    "height=0.15cm,\n"
    f"width={8 * len(export_df.columns)}pt,\n"
    "colormap name=coolwarm,\n"
    "colorbar horizontal,\n"
    f"point meta min={vmin:.2f},\n"
    f"point meta max={vmax:.2f},\n"
    "colorbar style={\n"
    f"width={8 * len(export_df.columns)}pt,\n"
    "height=0.15cm,\n"
    "scaled ticks=false,\n"
    f"xtick={{ {vmin:.2f}, {(vmin + vmax)/2:.2f}, {vmax:.2f} }},\n"
    "xticklabel style={font=\\scriptsize},\n"
    "xticklabel={\\pgfmathprintnumber[fixed,precision=2]{\\tick}},\n"
    "},\n"
    "]\n"
    f"\\addplot [draw=none] coordinates {{ ({vmin:.2f},0) ({vmax:.2f},0) }};\n"
    "\\end{axis}\n"
    "\\end{tikzpicture}\n} \\\\\n\\end{tabular}\n\\end{minipage}"
)
                        
                    elif selected_param == "market":
                        export_df = tmp_df.copy()

                        # 1. סימונים אלגנטיים
                        true_sign = r"\checkmark"
                        false_sign = r"--"
                        
                        def make_MA_binary(x):
                            if x == "binary":
                                return False
                            elif x == "text":
                                return True
                            else:
                                return x

                        for col in export_df.columns:
                            export_df[col] = export_df[col].apply(lambda x: make_MA_binary(x))
                            if export_df[col].dtype == bool:
                                export_df[col] = export_df[col].map({True: true_sign, False: false_sign})
                            export_df[col] = export_df[col].apply(lambda x: r"$\infty$" if x == "inf" else x)
                            
                            # if there are the same values in the column, remove the column
                            if len(export_df[col].unique()) == 1:
                                export_df = export_df.drop(columns=[col])
                            

                        # 2. חישוב effect ± CI (הכפלה ב-100 והצגת ערכים כנקודות אחוז)
                        def format_effect(row):
                            delta = row["ci_high"] - row["effect"]
                            effect = row["effect"] * 100
                            delta = delta * 100

                            effect_str = f"{effect:.1f}".rstrip('0').rstrip('.')
                            delta_str = f"{delta:.1f}".rstrip('0').rstrip('.')

                            return effect_str if delta == 0 else f"{effect_str} ± {delta_str}"

                        export_df["effect ± CI"] = export_df.apply(format_effect, axis=1)
                        for col in export_df.columns:
                            export_df[col] = export_df[col].apply(lambda x: f"{x}".rstrip('0').rstrip('.'))


                        export_df = export_df.drop(columns=["effect", "ci_low", "ci_high"])
                        feature_cols = [col for col in export_df.columns if col != "effect ± CI"]
                        export_df = export_df[feature_cols + ["effect ± CI"]]

                        latex_core = export_df.to_latex(index=False, escape=False)

                    else:
                        export_df = filtered_data.copy()

                        def format_effect(row):
                            if np.isinf(row["effect"]):
                                return r"$\infty$" if row["effect"] > 0 else r"$-\infty$"
                            delta = row["ci_high"] - row["effect"]
                            effect = row["effect"] * 100
                            delta = delta * 100
                            return f"{effect:.1f}".rstrip('0').rstrip('.') + " ± " + f"{delta:.1f}".rstrip('0').rstrip('.')

                        export_df["effect ± CI"] = export_df.apply(format_effect, axis=1)
                        export_df = export_df[["value", "effect ± CI"]]

                        latex_core = export_df.to_latex(index=False, escape=False)

                    # עטיפה ב-footnotesize
                    # latex_table = "\\begin{footnotesize}\n" + latex_core + "\\end{footnotesize}"

                    st.code(latex_core, language="latex")

                    
            
                if st.button("Back to Main Menu"):
                    st.session_state.initial_choice_made = False
                    st.session_state.step = 'model_selection'
                    st.session_state.selected_models = []
                    st.session_state.parameters = {}
                    st.rerun()


if __name__ == "__main__":
    main()