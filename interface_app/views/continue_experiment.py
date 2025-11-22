import os
import re

import streamlit as st

from interface_app.experiments import continue_experiment, list_existing_experiments


def render_continue_existing_experiment():
    existing_experiments = list_existing_experiments()
    if existing_experiments:
        experiment_name = st.selectbox("Select an Experiment to Continue", existing_experiments)
        sanitized_name = re.sub(r'\W+', '', experiment_name)
        base_path = f"interface_app/generated_experiments/{sanitized_name}"
        done_count = len([f for f in os.listdir(base_path) if f.endswith('_done.json')])
        config_count = len([f for f in os.listdir(base_path) if f.endswith('.json')])

        st.write(f"Selected Experiment: {experiment_name} ({done_count}/{config_count} completed)")
        if st.button("Continue", disabled=(done_count == config_count)):
            continue_experiment(experiment_name)
    else:
        st.info("No existing experiments found.")
        st.stop()
