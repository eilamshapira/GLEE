# GLEE (Games in Language-based Economic Environments) Copilot Instructions

This repository contains the GLEE framework for evaluating LLMs in economic games (Bargaining, Negotiation, Persuasion).

## Project Architecture

- **Core Logic**:
  - `games/`: Contains game logic. All games inherit from `games.base_game.Game`.
  - `players/`: Contains player implementations (LLM wrappers). All players inherit from `players.base_player.Player`.
  - `main.py`: Entry point for LLM vs LLM simulations. Orchestrates game setup via `game_factory` and `player_factory`.
- **Analysis**:
  - `analyze/`: Scripts for metrics, statistics, and ML analysis.
  - `analyze/main.py`: Main pipeline for generating tables and figures.
- **Human Experiments**:
  - `otree_game/`: oTree application for Human vs LLM experiments.
- **Interface**:
  - `interface_app/`: Streamlit app for data collection and analysis visualization.

## Development Workflows

### Environment Setup
Before running any code, ensure you are in the `.GLEE-env` environment:
```bash
source init.sh
```

### Running Simulations (LLM vs LLM)
Use `main.py` with a configuration file:
```bash
python main.py -c sample_configs/bargaining/vertexai_config.json
```
- **Configuration**: JSON files in `sample_configs/` or `output/configs/` control game parameters, player types, and prompts.

### Running Analysis
Run the full analysis pipeline:
```bash
python analyze/main.py
```
Or specific regression analysis:
```bash
python analyze/ML.py --exp_name <name> --bargaining <path_to_csv> ...
```

### Human Data Collection
Start the oTree server:
```bash
otree devserver
```

### Interface
Run the Streamlit app:
```bash
streamlit run interface.py
```

## Code Conventions & Patterns

- **Factories**: Use `games.game_factory` and `players.player_factory` to instantiate components. When adding new games or players, register them here.
- **Base Classes**:
  - **`Game`**: Implement the `game()` method for the main loop. Use `set_rules()` and `add_format()` for prompt management.
  - **`Player`**: Implement `add_message()` and `get_text_answer()`. The `act()` method handles the prompt-response loop and JSON parsing.
- **Data Logging**: Use `utils.utils.DataLogger` to save game results.
- **Prompting**: Prompts are typically stored in external files (`.txt` or `.json`) and formatted using `utils.utils.create_prompts_from_templates`.
- **Environment**:
  - API keys are managed via `litellm/init_litellm.sh` or environment variables.
  - Local models use `FastChat`.

## Key Files
- `games/base_game.py`: Abstract base class for games.
- `players/base_player.py`: Abstract base class for players.
- `utils/utils.py`: Shared utilities for logging and config reading.
- `analyze/ML.py`: Regression analysis logic.
