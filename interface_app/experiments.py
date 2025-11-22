import os
import threading
import time
from subprocess import Popen

import streamlit as st

sem = threading.Semaphore(50)


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
    base_path = "interface_app/generated_experiments"
    if not os.path.exists(base_path):
        return []
    return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]


def list_existing_results():
    base_path = "Data"
    if not os.path.exists(base_path):
        return []
    return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]


def continue_experiment(experiment_name):
    base_path = os.path.join("interface_app/generated_experiments", experiment_name)
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
