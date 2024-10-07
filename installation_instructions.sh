# clone the repository
git clone https://github.com/gleeframework/GLEE.git

# create a conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda create --name GLEE -y python=3.11
conda activate GLEE

# install the requirements

# requirements for the LLM games:
pip install torch openai anthropic google.generativeai google-cloud-aiplatform vertexai
pip install psutil
pip install transformers
pip install accelerate
pip install pandas
pip install pytest
pip install flash_attn==2.5.8

pip install wandb

# get the code from external package FastChat
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
git checkout a5c29e1b94f0537a73273b0c688a4454a8db362a
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"

cd ..

# additional requirements for the analysis:
#pip install joblib matplotlib tqdm statsmodels patsy seaborn

# additional requirements for the human data collection app:
#pip install otree psycopg2 sentry-sdk

# additional requirement for the create_otree_configs.py script and the connection to the mturk requester:
#pip install boto3

# Don't forget to add google key to google_key.json if you want to use the google api

