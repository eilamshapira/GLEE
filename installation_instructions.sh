# Create the conda environment
conda env create -f environment.yml
conda activate GLEE

# Clone FastChat at specific commit
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
git checkout a5c29e1b94f0537a73273b0c688a4454a8db362a
pip install -e ".[model_worker,webui]"
cd ..

# additional requirements for the analysis:
#pip install joblib matplotlib tqdm statsmodels patsy seaborn

# additional requirements for the human data collection app:
#pip install otree psycopg2 sentry-sdk

# additional requirement for the create_otree_configs.py script and the connection to the mturk requester:
#pip install boto3

# To use API models, add them to the appropriate files:
# - litellm/init_litellm.sh for most models
# - litellm/google_key.json for Vertex AI models
