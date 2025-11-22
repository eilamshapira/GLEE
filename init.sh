#!/bin/bash

ENV_DIR=".GLEE-env"

# Ensure uv is installed
if ! command -v uv &>/dev/null; then
    echo "📦 uv not found. Installing uv..."
    "$PIP" install uv || echo "⚠️ Failed to install uv; continuing..."
else
    echo "📦 uv is already installed. Proceeding..."
fi

# Create uv venv with Python 3.11 if it doesn't already exist
if [ ! -d "$ENV_DIR" ]; then
    echo "🌀 Creating uv venv ($ENV_DIR) with Python 3.11"
    uv venv -p python3.11 "$ENV_DIR" || echo "⚠️ Failed to create uv venv; continuing..."
else
    echo "🌀 uv venv ($ENV_DIR) already exists. Skipping creation."
fi

# Activate the environment
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "🚀 Activating uv venv ($ENV_DIR)"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate" || echo "⚠️ Failed to activate venv; continuing..."
else
    echo "⚠️ Activation script not found in $ENV_DIR; did venv creation fail?"
fi

# Install dependencies using uv
if [ -f "requirements.txt" ]; then
    echo "📄 Installing dependencies from requirements.txt using uv"
    uv pip install -r requirements.txt || echo "⚠️ Dependency install failed; continuing..."
else
    echo "⚠️ requirements.txt not found; skipping dependency install."
fi

# if the file litellm/init_litellm.sh does not exist, copy from litellm/init_litellm.sh.example
if [ ! -f "litellm/init_litellm.sh" ]; then
    echo "📋 Copying litellm/init_litellm.sh from example"
    cp litellm/init_litellm.sh.example litellm/init_litellm.sh
fi

echo "🎉 Setup completed! To activate in a new shell, run: source $ENV_DIR/bin/activate"