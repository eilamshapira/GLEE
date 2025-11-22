# Analyze Directory Documentation

This directory contains scripts and modules for analyzing game data, calculating metrics, running regression models, and generating tables and figures.

## Files and Functions

### `basic_statistics.py`
Calculates basic statistics for games, such as counting messages, characters, and words.
- `get_paths(main_dir, game_id, game_type)`: Returns paths for game CSV, config, and other files.
- `count_games_as_factor_of_llms(configs)`: Counts games per LLM pair.
- `correlation_between_first_offer_and_final_gain(configs)`: Calculates correlation between first offer and final gain.
- `process_game(exp_name, game, family)`: Wrapper to process a single game.
- `count_messages_and_characters(exp_name, game_id, game_type)`: Counts messages, words, and characters for a game.
- `make_saving_path(family, config_files)`: Creates a directory for saving statistics.

### `calc_adj_r2.py`
Calculates the adjusted R-squared for regression models.
- `calc_adj_r2()`: Iterates over model files, calculates adjusted R2, and saves results to `tables/models_r2.csv`.

### `configs.py`
Utilities for discovering experiment runs and materializing config CSVs.
- `slugify_name(value)`: Sanitizes strings for filenames.
- `get_configs_csv_path(game_type, ...)`: Returns path for config CSV.
- `load_data(game_type, game_id, main_data_folder)`: Loads game data from CSV.
- `get_stats_bargaining(...)`, `get_stats_negotiation(...)`, `get_stats_persuasion(...)`: Extracts game-specific statistics.
- `create_configs_file(...)`: Scans directories and creates a master config CSV.
- `create_config_with_stats(...)`: Adds statistics to the config CSV.

### `consts.py`
Defines constants used across the analysis scripts.
- `DataStore`, `HUMAN_DIR`, `VECTORS_DIR`: Directory paths.
- Player names and other constants.

### `generate_tables.py`
Generates tables for basic statistics.
- `get_paths(game_id, game_type)`: Returns paths for game files (similar to `basic_statistics.py`).
- `count_games_as_factor_of_llms(configs)`: Generates table for game counts.
- `correlation_between_first_offer_and_final_gain(configs)`: Generates correlation table.
- `process_game(game, family)`: Wrapper for processing games.
- `count_messages_and_characters(game_id, game_type)`: Counts messages/chars (similar to `basic_statistics.py`).
- `create_basic_statistics_tables(family, config_files)`: Main function to generate statistics tables for a family of games.

### `main.py`
The entry point for the analysis pipeline.
- `main()`: Orchestrates the workflow:
    1. Creates config files (`create_configs_file`).
    2. Adds stats to configs (`create_config_with_stats`).
    3. Generates basic stats tables (`generate_tables.create_basic_statistics_tables`).
    4. Calculates metrics (`calc_metrics`).
    5. Calculates adjusted R2 (`calc_adj_r2`).
    6. Creates figures (`create_figures`).

### `metrics.py`
Defines classes for calculating game-specific metrics.
- `Metrics`: Base class for metrics calculation.
    - `get_statistics(mode, group_by)`: Returns statistics for a mode.
    - `plot_graphs(...)`: Plots metrics.
    - `complete_missing_groups(...)`: Handles missing data combinations and runs OLS regression.
- `NegotiationMetrics`: Subclass for Negotiation games.
- `BargainingMetrics`: Subclass for Bargaining games.
- `PersuasionMetrics`: Subclass for Persuasion games.
- `create_table(...)`: Generates LaTeX tables for metrics.
- `calc_metrics()`: Main function to calculate and print metrics tables.

### `ML.py`
Performs machine learning analysis, specifically regression to analyze feature importance.
- `load_data(...)`: Loads data for all game types.
- `ModelOfOneHots`: Wrapper for models using one-hot encoding.
- `StatsModelOfOneHots`: Wrapper for statsmodels with one-hot encoding and baseline handling.
- `merge_features(...)`: Merges features based on strategy (market, all, none).
- `prepare_dataset_for_task(...)`: Prepares data for a specific task/metric.
- `analyze_coefs(args)`: Analyzes coefficients from regression models.

## Usage Examples

### Running `ML.py`
To run the regression analysis, use the following command structure:

```bash
python analyze/ML.py --exp_name <experiment_name> \
    --bargaining <path_to_bargaining_config> \
    --negotiation <path_to_negotiation_config> \
    --persuasion <path_to_persuasion_config>
```

**Key Arguments:**
- `--exp_name`: Name for the output files (default: "exp").
- `--per_market`: Run analysis separately for each market (combination of game parameters).
- `--merge_features`: Strategy to merge features (`none`, `market`, `all`).
- `--myopic_split`: Split persuasion data by myopic/non-myopic buyer.
- `--models_interaction`: Include interaction terms between player models.
- `--seed`: Random seed (default: 0).

**Example - Per Market Analysis:**
```bash
python analyze/ML.py --exp_name market_analysis --per_market \
    --bargaining output/configs/bargaining_with_stats.csv \
    --negotiation output/configs/negotiation_with_stats.csv \
    --persuasion output/configs/persuasion_with_stats.csv
```
