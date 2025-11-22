import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import matplotlib

from analyze.configs import create_config_with_stats, create_configs_file
from analyze import generate_tables
from analyze.metrics import calc_metrics
from analyze.calc_adj_r2 import calc_adj_r2
from analyze.parameters import create_figures

# Create output directory and its subdirectories
os.makedirs("output", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)
os.makedirs("output/configs", exist_ok=True)
os.makedirs("output/model", exist_ok=True)
os.makedirs("output/model_summary", exist_ok=True)
os.makedirs("output/basic_statistics", exist_ok=True)
os.makedirs("output/analyze_coefs", exist_ok=True)

matplotlib.use('Agg')  # Use the 'Agg' backend


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
