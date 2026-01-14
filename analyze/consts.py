import os
from pathlib import Path

NATURE_NAME = "Nature"
HUMAN = "human"
BOT = "bot"
PROPOSER = "proposer"
RECEIVER = "receiver"
RESPONSE = "response"

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(REPO_ROOT / "Data")
DataStore = OUTPUT_DIR
HUMAN_DIR = os.path.join(DataStore, "human_vs_llm")
VECTORS_DIR = os.path.join(DataStore, "vectors")

os.makedirs(DataStore, exist_ok=True)

OTREE_CONFIGS_PATH = "otree_game/configs"
OTREE_PAGES = "otree_game/pages"
OTREE_MAX_ROUNDS = 99
