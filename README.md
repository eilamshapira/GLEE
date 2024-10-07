# GLEE: A Framework and Benchmark for LLM Evaluation in Language-based Economics


## General information

This repository contains the code for the GLEE framework, which is a framework for evaluating LLMs in language-based economics.
The framework includes three games: bargaining, negotiation and persuasion. The framework includes a simulation environment
for running games between LLM players and human players, and a human data collection system for running human experiments
in which human players play against LLM players. The framework includes a set of metrics for evaluating the LLM players,
and a set of baselines for comparison. The framework is used to evaluate the performance of several LLM players
on the three games, and to compare the performance of the LLM players to the baselines.

The repository include 3 main parts:
1. The main part is the LLM vs. LLM games. It is run by the *main.py* script in the code directory.
The script runs a game between two players, and receives a configuration path as an argument. 
Several sample configurations are in the sample_configs directory, which includes
both terminal and LLM games for each of the three games families. If tou want to run
a large number of games, you can use the *create_YAMLs.py* script in the utils directory. This script uses the weights
and biases API to create and run a large number of games.
2. The human data collection system is in the otree_game directory. The system is based on the oTree package,
and is used to run human experiments, in which human players play against LLM players.
3. The analysis part is in the analyze directory. There are several scripts for analyzing the results of the games,
and extracting the metrics from the results.


## Getting started


### Prerequisites

Before you begin, ensure you have the following tools installed on your system:
- Git
- Anaconda or Miniconda

### Installation

Complete installation instructions are in the installation_instructions.sh file. Notice that there are 
different instructions for the code directory, the data directory, the human data collection system and the analysis subdirectory
in the code directory.

## Running the code

### Run a game of LLM vs. LLM

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

### Analyze the results

The folder "analyze" contains all the necessary files for data analysis as presented in the paper. 
The file "main.py" should be executed to perform the analysis on all the files located in the "data" folder.

```
python analyze/main.py
```


## Main Files and Directories

- **analyze/**: The directory includes the scripts for analyzing the results of the games. The main script is *main.py*.
- **consts.py**: The file includes the constants used in the code.
- **games/**: The directory includes the code for the three games: bargaining, negotiation and persuasion. 
In addition, the directory includes code for the bots classes which is used in the human data collection system to simulate the LLM player.
- **installation_instructions.sh**: The file includes the installation instructions for the repository.
- **main.py**: The main script for running the LLM vs. LLM games.
- **otree_game/**: The directory includes the otree code for the human data collection system. 
The directory includes the models classes, the pages classes, the pages templates and some utils functions.
In addition, the directory includes a script for creating the human data collection configurations and the script for processing the human data.
- **players/**: The directory includes the code for the players classes. The directory includes the classes for the LLM players,
the terminal player, the oTree player, the LLM player for the oTree system and the demo player, which is used for debugging.
- **sample_configs/**: The directory includes some sample configurations for the LLM vs. LLM games.
- **settings.py**: The file includes the otree settings for the human data collection system.
- **utils/**: The directory includes some utility functions for the whole project.
- **templates/**: The directory includes the templates for the human data collection system.

