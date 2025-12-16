# GLEE: A Framework and Benchmark for LLM Evaluation in Language-based Economics


## Overview

GLEE (Games in Language-based Economic Environments) is a comprehensive framework designed to systematically evaluate the performance of Large Language Models (LLMs) and human participants in strategic, language-driven economic interactions. The framework standardizes research across three fundamental economic games—bargaining, negotiation, and persuasion—each modeled to reflect real-world scenarios involving strategic decision-making and natural language communication.

GLEE addresses critical questions regarding the rationality, fairness, and efficiency of LLMs in strategic contexts. By providing extensive parameterization and controlled environments, the framework facilitates:

*Simulation and experimentation* in automated interactions between LLM agents and between humans and LLMs.

*Human-data-collection* system leveraging the oTree platform to conduct structured human-vs-LLM experiments.

*Analysis suite* for evaluating new and existing interaction data. The system enables comparison of fresh results with the extensive GLEE dataset and supports in-depth examination of model behavior across varied configurations.

In addition, the GLEE package supports new data collection for follow-up research and future publications.

This repository includes all the necessary code and data to run experiments, collect new data, and replicate or extend the results from the corresponding research paper.

## Quick Start

1. Clone & Install

```
# Clone repository
git clone https://github.com/eilamshapira/GLEE.git
cd GLEE

# Install everything (creates a new uv Python environment, downloads models, prepares Data/)
source init.sh
```


2. Provide API Keys (optional)

If you plan to collect data using paid LLM endpoints (e.g., OpenAI, Anthropic) through litellm, create/edit the files inside the litellm/ directory and paste your keys there. The format is documented in that folder.

3. Run the Streamlit Interface

The same Streamlit app is used both for collecting fresh data and for analyzing existing GLEE logs:

```
streamlit run interface.py
```

You will be prompted to choose between data‑collection mode and analysis mode.

## Running the code

### Run a game of LLM vs. LLM using command line

In order to run games of LLMS vs. LLMS, you can use the *main.py* script in the code directory and the sample configurations
in the sample_configs directory. For example, to run a game between two LLAMA3 players
in the bargaining game, you can run the following command:

```
python main.py -c sample_configs/bargaining/vertexai_config.json
```

If you want to run a similar game with terminal players, you can run the following command:

```
python main.py -c sample_configs/bargaining/terminal_config.json
```

### Run a game between Human and LLM

In order to run the human data collection system via otree in your local machine, you can use the following commands:

```
otree devserver
```

Follow the instructions in the terminal to open the system in your browser. There you can create demo games,
run the games and collect the data. We created some sample configurations which will appear at the top of the configurations list.
If you want to run the human data collection system in a public server you can use the heroku platform or any other server platform.

In addition, the directory includes a script for creating the human data collection configurations and the script for processing the human data.
- **players/**: The directory includes the code for the players classes. The directory includes the classes for the LLM players,
the terminal player, the oTree player, the LLM player for the oTree system and the demo player, which is used for debugging.
- **sample_configs/**: The directory includes some sample configurations for the LLM vs. LLM games.
- **settings.py**: The file includes the otree settings for the human data collection system.
- **utils/**: The directory includes some utility functions for the whole project.
- **templates/**: The directory includes the templates for the human data collection system.
