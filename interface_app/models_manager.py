import os
import json

MODELS_FILE = "models_list.json"
DEFAULT_MODELS = [
    "xai/grok-2-1212",
    "gpt-4o",
    "o3-mini",
    "gpt-4o-mini",
    "vertex_ai/mistral-large-2411",
    "vertex_ai/meta/llama-3.3-70b-instruct-maas",
    "vertex_ai/meta/llama-3.1-405b-instruct-maas",
    "vertex_ai/gemini-1.5-pro",
    "vertex_ai/gemini-1.5-flash",
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.0-flash-lite",
    "vertex_ai/claude-3-7-sonnet@20250219",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
]

def get_models():
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_MODELS
    return DEFAULT_MODELS

def save_models(models):
    with open(MODELS_FILE, "w") as f:
        json.dump(models, f, indent=4)
