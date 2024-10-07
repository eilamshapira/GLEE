import pandas as pd
import json
import os
import boto3
from utils.utils import get_commit_hash
from consts import OTREE_CONFIGS_PATH


def create_qualification(config_name, client):
    try:
        response = client.create_qualification_type(
            Name=f'{config_name}',
            Keywords='qualification, game, study',
            Description=f'Qualification for {config_name}. Preventing workers from participating in the same '
                        f'configuration group more than once.',
            QualificationTypeStatus='Active',
            AutoGranted=False
        )
        qualification_id = response['QualificationType']['QualificationTypeId']
        print(f"Created qualification for {config_name}: {qualification_id}")
        return qualification_id
    except Exception as e:
        print(f"Error creating qualification for {config_name} {str(e)}")
        raise e


def read_csv_and_generate_json(csv_file_path, config_base_path, commit_string=None, qualifications=True):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    cols = df.columns
    assert df.isna().sum().sum() == 0, "There are empty entries in the relevant columns"

    assert "id" in cols, "id column is required"
    assert "game_type" in cols, "game_type column is required"
    assert "participation_fee" in cols, "participation_fee column is required"
    assert "human_is_player" in cols, "human_is_player column is required"
    assert "qualification" in cols or not qualifications, "qualification_id column is required"

    player_1_columns = [col for col in cols if "player_1_args" in col]
    player_2_columns = [col for col in cols if "player_2_args" in col]
    game_columns = [col for col in cols if "game_args" in col]
    general_columns = ["participation_fee"]

    df.set_index("id", inplace=True)

    qualifications_names = {}
    if qualifications:
        mturk_client = boto3.client('mturk', region_name='us-east-1')
        try:
            response = mturk_client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)
        except Exception as e:
            print(f"Error listing qualifications: {str(e)}")
            raise e
        qualification_types = response['QualificationTypes']
        for qualification in qualification_types:
            qualification_id = qualification['QualificationTypeId']
            qualification_name = qualification['Name']
            qualifications_names[qualification_name] = qualification_id
    else:
        mturk_client = None

    commits_list = []

    # Convert each row to a JSON object and save it to a file
    for index, row in df.iterrows():
        # Convert row to a dictionary and then to a JSON string
        general_data = {col: row[col] for col in general_columns}
        player_1_data = {col.removeprefix("player_1_args_"): row[col] for col in player_1_columns}
        player_1_data["public_name"] = "Alice"
        player_2_data = {col.removeprefix("player_2_args_"): row[col] for col in player_2_columns}
        player_2_data["public_name"] = "Bob"
        game_data = {col.removeprefix("game_args_"): row[col] for col in game_columns}

        game_type = row['game_type']
        config_name = f'{game_type}_{index}'
        if qualifications:
            qualification_name = "many_game_" + row['qualification'] + "_qualification"
            if qualification_name not in qualifications_names:
                qualification_id = create_qualification(qualification_name, mturk_client)
                qualifications_names[qualification_name] = qualification_id
            else:
                qualification_id = qualifications_names[qualification_name]
        else:
            qualification_id = None

        if row['human_is_player'] == 1:
            player_1_type = "otree"
            player_2_type = "otree_LLM"
        else:
            player_1_type = "otree_LLM"
            player_2_type = "otree"

        json_data = {
            "name": config_name,
            "display_name": config_name,
            "title": row['title'],
            "description": row['description'],
            "game_type": game_type,
            "player_1_type": player_1_type,
            'player_1_args': player_1_data,
            "player_2_type": player_2_type,
            'player_2_args': player_2_data,
            'game_args': game_data,
            **general_data,
        }
        if qualifications:
            json_data['qualification_id'] = qualification_id
            json_data['qualification_base'] = row['qualification']

        dir_path = os.path.join(config_base_path, game_type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        json_file_name = f"{index}.json"
        json_file_path = rf'{dir_path}/{json_file_name}'

        if commit_string is None and os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                existing_json_data = json.load(json_file)
                json_data['commit'] = existing_json_data['commit']
                commits_list.append(existing_json_data['commit'])
        elif commit_hash is not None:
            json_data['commit'] = commit_string

        with open(json_file_path, 'w') as json_file:
            # add an empty row at the end of the file
            json.dump(json_data, json_file, indent=4)
            json_file.write('\n')
            print(f"Created JSON file: {json_file_name}")

    print("Commits used: ", set(commits_list))


config_path = OTREE_CONFIGS_PATH.split('/')[-1]
csv_paths = ['human_configs/bargaining.csv', 'human_configs/persuasion.csv', 'human_configs/negotiation.csv']
commit_hash = get_commit_hash()
for csv_path in csv_paths:
    read_csv_and_generate_json(csv_path, config_path, commit_string=None)
