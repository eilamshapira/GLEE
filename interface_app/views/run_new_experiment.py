import os
import re
import threading
import time
import json

import streamlit as st

from interface_app.config_generation import create_config_files
from interface_app.models_manager import get_models, save_models
from interface_app.experiments import run_experiment
from interface_app.game_parameters import game_parameters


def render_run_new_experiment():
    if st.session_state.step == 'model_selection':
        st.subheader("Select Models to Run")
        st.session_state.selected_game = st.selectbox(
            "Select Game to Configure",
            ["Bargaining", "Persuasion", "Negotiation"],
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            models_choice = st.multiselect("Choose models", options=get_models())
        with col2:
            if st.button("Edit Models"):
                st.session_state.show_model_editor = True

        if st.session_state.get("show_model_editor", False):
            with st.expander("Edit Models List", expanded=True):
                current_models_json = json.dumps(get_models(), indent=4)
                new_models_json = st.text_area("Models (JSON list)", value=current_models_json, height=300)
                if st.button("Save Models List"):
                    try:
                        new_models = json.loads(new_models_json)
                        if isinstance(new_models, list):
                            save_models(new_models)
                            st.success("Models list updated!")
                            st.session_state.show_model_editor = False
                            st.rerun()
                        else:
                            st.error("Input must be a JSON list of strings")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
                if st.button("Close Editor"):
                    st.session_state.show_model_editor = False
                    st.rerun()

        runs_per_model = st.number_input(
            "Runs per models",
            min_value=1,
            max_value=100000,
            value=500,
            step=100,
            key="num_per_model",
        )
        total_runs = len(models_choice) * runs_per_model
        if models_choice:
            st.write(f"Each model will play {runs_per_model} times as Player 1 and {runs_per_model} times as Player 2.")
            st.write(f"Total number of games: {total_runs:,}")
            if len(models_choice) > 0:
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
        st.write(
            f"Total number of runs: {len(st.session_state.parameters) * st.session_state.run_per_model * len(st.session_state.selected_models)}"
        )
        st.write(st.session_state.parameters)

        experiment_name = st.text_input("Enter an experiment name:")
        sanitized_name = re.sub(r'\W+', '', experiment_name)

        if st.button("Start Experiment") and sanitized_name:
            create_config_files(sanitized_name, st.session_state.run_per_model)

            base_path = f"interface_app/generated_experiments/{sanitized_name}"
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
