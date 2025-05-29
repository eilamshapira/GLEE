from metrics import BargainingMetrics, PersuasionMetrics, NegotiationMetrics
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
# plot the regression line
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
import os
import sys
import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import BargainingMetrics, PersuasionMetrics, NegotiationMetrics
import statsmodels.api as sm



def load_data(bargaining_files, negotiation_files, persuasion_files, games_per_model=-1):
    if games_per_model > 0:
        print(f"loading {games_per_model} games per LLM")
    else:
        print("loading all games")
    
    # Convert file paths to use output directory if they don't already
    def convert_to_output_path(file_path):
        if not file_path.startswith("output/"):
            return f"output/configs/{os.path.basename(file_path)}"
        return file_path
    
    bargaining_files = [convert_to_output_path(f) for f in bargaining_files] if bargaining_files else []
    negotiation_files = [convert_to_output_path(f) for f in negotiation_files] if negotiation_files else []
    persuasion_files = [convert_to_output_path(f) for f in persuasion_files] if persuasion_files else []
    
    rubin = [BargainingMetrics(file, drop_unknown_llms=False) for file in bargaining_files] if bargaining_files else []
    nego = [NegotiationMetrics(file, drop_unknown_llms=False) for file in negotiation_files] if negotiation_files else []
    pers = [PersuasionMetrics(file, drop_unknown_llms=False) for file in persuasion_files] if persuasion_files else []

    # print data sizes
    
    print(f"Negotiation data size: {[file.data.shape for file in nego]}")
    print(f"Persuasion data size: {[file.data.shape for file in pers]}")
    print(f"Bargaining data size: {[file.data.shape for file in rubin]}")
    all_data = []

    # for each data file, calculate how the percentage of fairness and efficiency outside the range [0, 1]. in addition, print the number of times each value is outside the range
    for game_familiy in [nego, pers, rubin]:
        if len(game_familiy) == 0:
            continue
        data = pd.concat([file.data for file in game_familiy])
        print(f"Data size: {data.shape}")
        # print the class of game_familiy
        print("Game: ", game_familiy.__class__.__name__)
        print("Shape: ", data.shape)
        print(f"Fairness outside range [0, 1]: {len(data[(data['fairness'] < 0) | (data['fairness'] > 1)]) / len(data) * 100:.3f}% ({len(data[(data['fairness'] < 0) | (data['fairness'] > 1)])} times)")
        print(f"Efficiency outside range [0, 1]: {len(data[(data['efficiency'] < 0) | (data['efficiency'] > 1)]) / len(data) * 100:.3f}% ({len(data[(data['efficiency'] < 0) | (data['efficiency'] > 1)])} times)")
        # clipping
        data["fairness"] = data["fairness"].clip(0, 1)
        data["efficiency"] = data["efficiency"].clip(0, 1)
        
        print(data.head())
        print(data.columns)
        
        # if player_1_type is litellm player, set player_1_args_model_name to be player_1_args_model_name.split("/")[1:] and join them back
        def remove_litellm_provider_from_model_name(row, player_id):
            assert player_id == 1 or player_id == 2
            player_type_col = f"player_{player_id}_type"
            model_name_col = f"player_{player_id}_args_model_name"
            if row[player_type_col] == "litellm":
                if "/" in row[model_name_col]:
                    model_name = row[model_name_col].split("/")
                    model_name = "/".join(model_name[1:])
                    return model_name
                else:
                    return row[model_name_col]
            else:
                return row[model_name_col]
            
        for player_id in [1, 2]:        
            data[f"player_{player_id}_args_model_name"] = data.apply(lambda row: remove_litellm_provider_from_model_name(row, player_id), axis=1)
        
        # drop null if any values of fairness, efficiency, alice_gain, bob_gain are null. print how many rows were dropped
        before_shape = data.shape
        data = data.dropna(subset=['fairness', 'efficiency', 'alice_self_gain', 'bob_self_gain'])
        after_shape = data.shape
        if before_shape[0] != after_shape[0]:
            print("Dropping null values")
            print("Before: ", before_shape)
            print("After: ", after_shape)
        all_data.append(data)

    
    all_data = pd.concat(all_data).sample(frac=1)
    all_data.reset_index(drop=True, inplace=True)

    return all_data


def replace_non_deal_in_negotiation_to_nan(data):
    # data[""]
    print(data.columns)
    return data

METRICS = ["fairness", "efficiency", "alice_self_gain", "bob_self_gain"]

# get seed from command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--exp_name", type=str, default="exp")
parser.add_argument("--task", type=str, default="all")
parser.add_argument("--merge_features", type=str, default="none", choices=["none", "market", "all"])
# add store_true for myopic_split
parser.add_argument("--myopic_split", action="store_true", default=False)
parser.add_argument("--models_interaction", action="store_true", default=False)
parser.add_argument("--bargaining", type=str, nargs="+")
parser.add_argument("--persuasion", type=str, nargs="+")
parser.add_argument("--negotiation", type=str, nargs="+")


args = parser.parse_args()
seed = args.seed
task = args.task

FAMILIES = []
for family in ["persuasion", "negotiation", "bargaining"]:
    for family_name in ["persuasion", "negotiation", "bargaining"]:
        if getattr(args, family_name) is not None and len(getattr(args, family_name)) > 0:
            FAMILIES.append(family_name)

# set seed
np.random.seed(seed)


class ModelOfOneHots:
    def __init__(self, model, model_name=None):
        self.model = model
        self.x_dummies = None
        self.model_name = model_name if model_name else model.__class__.__name__

    def fit(self, X, y):
        # each value of each parameter will be one-hot encoded
        X = pd.get_dummies(X.astype(str), prefix_sep="==")
        self.model.fit(X, y)
        self.x_dummies = X.columns

    def predict(self, X):
        X = pd.get_dummies(X.astype(str), prefix_sep="==")
        # add missing columns
        for col in self.x_dummies:
            if col not in X.columns:
                X[col] = 0
        X = X[self.x_dummies]
        return self.model.predict(X)

    def score(self, X, y):
        X = pd.get_dummies(X.astype(str), prefix_sep="==")
        # add missing columns
        for col in self.x_dummies:
            if col not in X.columns:
                X[col] = 0
        X = X[self.x_dummies]
        return self.model.score(X, y)

    def get_r2(self, X, y):
        X = pd.get_dummies(X.astype(str), prefix_sep="==")
        # add missing columns
        for col in self.x_dummies:
            if col not in X.columns:
                X[col] = 0
        X = X[self.x_dummies]
        return self.model.score(X, y)

class StatsModelOfOneHots:
    def __init__(self):
        self.model = None
        self.x_dummies = None
        self.baselines = {}  # נשמור מה הושמט
        self.all_levels = {}  # כל הקטגוריות שהיו
        self.feature_names = []
        
    def _fit_with_baseline(self, X_raw, y, forced_baselines=None):
        dummies_input = X_raw.copy()
        forced_baselines = forced_baselines or {}

        for col in self.feature_names:
            all_vals = self.all_levels[col]

            if col in forced_baselines and forced_baselines[col] in all_vals:
                baseline_value = forced_baselines[col]
            elif col in forced_baselines:
                baseline_value = all_vals[0]
                print(f"[WARNING] The value '{forced_baselines[col]}' does not exist in column '{col}'. Using arbitrary value: {baseline_value}")
                print('Options:', all_vals)
            else:
                baseline_value = all_vals[0]

            # Convert the value to a categorical column with precise order
            dummies_input[col] = pd.Categorical(
                X_raw[col].astype(str),
                categories=[baseline_value] + [v for v in all_vals if v != baseline_value],
                ordered=True
            )

        # Create dummies with drop_first to remove the baseline
        dummies = pd.get_dummies(dummies_input, prefix_sep="==", drop_first=True)

        # Check what was actually removed
        baselines = {}
        for col in self.feature_names:
            all_vals = self.all_levels[col]
            present = [c.split("==")[1] for c in dummies.columns if c.startswith(f"{col}==")]
            missing = list(set(all_vals) - set(present))
            if len(missing) == 1:
                detected_baseline = missing[0]
            elif len(missing) == 0:
                detected_baseline = None  # Can happen if there is only one value
            else:
                print(f"[ERROR] Problem detecting baseline for column {col}, missing values: {missing}")
                detected_baseline = None

            expected_baseline = forced_baselines.get(col, all_vals[0])
            if detected_baseline != expected_baseline:
                print(f"[WARNING] In column '{col}', the baseline actually removed is '{detected_baseline}', not what you intended ('{expected_baseline}')")

            baselines[col] = detected_baseline

        # התאמה בעזרת statsmodels
        X = sm.add_constant(dummies).astype(float)
        y = y.astype(float)
        model = sm.OLS(y, X).fit()
        return model, dummies.columns, baselines
    
    def fit(self, X_raw, y, forced_baselines=None):
        self.feature_names = list(X_raw.columns)
        self.all_levels = {
            col: sorted(X_raw[col].astype(str).unique())
            for col in self.feature_names
        }

        if forced_baselines is None:
            model1, dummies1, _ = self._fit_with_baseline(X_raw, y)
            coefs = model1.params
            best_values = {}
            for col in self.feature_names:
                values = self.all_levels[col]
                value_coefs = {v: (coefs.get(f"{col}=={v}", 0.0)) for v in values}
                best_value = max(value_coefs, key=value_coefs.get)
                best_values[col] = best_value
            forced_baselines = best_values

        self.model, self.x_dummies, self.baselines = self._fit_with_baseline(
            X_raw, y, forced_baselines=forced_baselines
        )

    def summary(self):
        return self.model.summary()

    def get_params_and_ci_relative_to_baseline(self):
        results = []
        cov = self.model.cov_params()
        params = self.model.params

        for feature in self.feature_names:
            baseline = self.baselines[feature]
            for value in self.all_levels[feature]:
                if value == baseline:
                    results.append({
                        "param": feature,
                        "value": value,
                        "coef": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "baseline": True
                    })
                else:
                    key = f"{feature}=={value}"
                    base_key = f"{feature}=={baseline}" if baseline is not None else None
                    coef = params.get(key, 0.0)

                    if base_key and base_key in params:
                        diff = coef - params[base_key]
                        var_diff = (
                            cov.loc[key, key]
                            + cov.loc[base_key, base_key]
                            - 2 * cov.loc[key, base_key]
                        )
                    else:
                        diff = coef
                        var_diff = cov.loc[key, key] if key in cov else 0.0

                    se = np.sqrt(var_diff) if var_diff > 0 else 0.0
                    ci_low = diff - 1.96 * se
                    ci_high = diff + 1.96 * se

                    results.append({
                        "param": feature,
                        "value": value,
                        "coef": diff,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "baseline": False
                    })
        return pd.DataFrame(results)


params_per_familiy = {
    "persuasion": ["game_args_is_myopic",
                   "game_args_product_price",
                   "game_args_p",
                   "game_args_v",
                   #    "c", # this parameter is not used since it is the same for all games (0)
                   #    "total_rounds", # this parameter is not used since it is the same for all games (20)
                   "game_args_is_seller_know_cv",
                   "game_args_is_buyer_know_p",
                   "game_args_seller_message_type",
                   "game_args_allow_buyer_message"],
    "negotiation": ["game_args_complete_information",
                    "game_args_messages_allowed",
                    "game_args_max_rounds",
                    "game_args_seller_value",
                    "game_args_buyer_value",
                    "game_args_product_price_order"],
    "bargaining": ["game_args_complete_information",
                   "game_args_messages_allowed",
                   "game_args_max_rounds",
                   "player_1_args_delta",
                   "player_2_args_delta",
                   "game_args_money_to_divide",
                   #    "game_args_show_inflation_update"
                   ]
}

def merge_features(X, merging_method, game_type):
    if merging_method == "none":
        return X
    elif merging_method == "market":
        # for persuasion rows: merge game_args_is_seller_know_cv, game_args_seller_message_type, and
        # game_args_is_myopic to one feature, called "market"
        # for negotiation: merge game_args_complete_information, game_args_messages_allowed, and
        # game_args_max_rounds to one feature, called "market"
        # for bargaining: merge game_args_complete_information, game_args_messages_allowed, and
        # game_args_max_rounds to one feature, called "market"
        def merge_persuasion(row):
            return f"CI={row['game_args_is_seller_know_cv']}_MA={row['game_args_seller_message_type']}_MYOPIC={row['game_args_is_myopic']}"
        def merge_negotiation(row):
            return f"CI={row['game_args_complete_information']}_MA={row['game_args_messages_allowed']}_MR={row['game_args_max_rounds']}"
        def merge_bargaining(row):
            return f"CI={row['game_args_complete_information']}_MA={row['game_args_messages_allowed']}_MR={row['game_args_max_rounds']}"
        X["market"] = X.apply(lambda row: merge_persuasion(row) if game_type == "persuasion" else merge_negotiation(row) if game_type == "negotiation" else merge_bargaining(row), axis=1)
        cols_to_remove = [c for c in ["game_args_is_seller_know_cv", "game_args_seller_message_type", "game_args_is_myopic", "game_args_complete_information", "game_args_messages_allowed", "game_args_max_rounds"] if c in X.columns]
        return X.drop(columns=cols_to_remove)
    elif merging_method == "all":
        # merge all features in params_per_familiy[family] and to one feature, called "parameters"
        cols_to_join = [col for col in params_per_familiy[game_type]]
        cols_to_join += (["models_names"] if "models_names" in X.columns else ["player_1_args_model_name", "player_2_args_model_name"])
        def merge_all(row):
            return "_".join(str(row[col]) for col in cols_to_join)
        X["parameters"] = X.apply(merge_all, axis=1)
        return X.drop(columns=cols_to_join)
            
    

def prepare_dataset_for_task(full_data, family, metric, split):
    global params_per_familiy
    assert metric in METRICS
    task_data = full_data[full_data["game_type"] == family]
    task_data = task_data.copy()
    
    if split == "myopic":
        task_data = task_data[task_data["game_args_is_myopic"] == True]
    elif split == "non_myopic":
        task_data = task_data[task_data["game_args_is_myopic"] == False]
    task_data = fix_infinity(family, task_data)
    
    task_data["result"] = task_data[metric]
    relvent_columns = params_per_familiy[family].copy()
    if args.models_interaction:
        relvent_columns += ["models_names"]
    else:
        relvent_columns += ["player_1_args_model_name",
                           "player_2_args_model_name"]
    relvent_columns += ["result"]
    task_data = task_data[relvent_columns].copy()
    task_data = merge_features(task_data, args.merge_features, game_type=family)
    
    nunique = task_data.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        # print(f"[{family}] Dropping constant columns: {constant_cols}")
        task_data = task_data.drop(columns=constant_cols)
    return task_data

results_mse = {}
results_adj_r2 = {}

N_SEEDS = 1
all_coefs = []
data = load_data(bargaining_files=args.bargaining,
                  persuasion_files=args.persuasion,
                  negotiation_files=args.negotiation,
                  games_per_model=-1)
data["models_names"] = "alice_" + data["player_1_args_model_name"] + \
        "_bob_" + data["player_2_args_model_name"]
        
        
def fix_infinity(family, X):
    if family == "persuasion":
        return X
    else:
        col = "game_args_max_rounds"
        X[col] = X[col].apply(lambda x: x if x <=20 else "inf")
        return X


forced_baselines_per_family = {
    "persuasion": {
        "game_args_is_myopic": False,
        "game_args_product_price": "10000.0",
        "game_args_p": "0.5",
        "game_args_v": "1.25",
        "game_args_is_seller_know_cv": True,
        "game_args_seller_message_type": "binary",
        # "models_names": "alice_gpt-3.5-turbo_bob_gpt-3.5-turbo",
        "player_1_args_model_name": "gemini-1.5-flash",
        "player_2_args_model_name": "gemini-1.5-flash",
        "model_names": "alice_gemini-1.5-flash_bob_gemini-1.5-flash",
        "market": "CI=True_MA=binary_MYOPIC=False",
    },
    "negotiation": {
        "game_args_complete_information": True,
        "game_args_messages_allowed": False,
        "game_args_max_rounds": "1",
        "game_args_seller_value": "1.0",
        "game_args_buyer_value": "1.0",
        "game_args_product_price_order": "10000.0",
        # "models_names": "alice_gpt-3.5-turbo_bob_gpt-3.5-turbo",
        "player_1_args_model_name": "gemini-1.5-flash",
        "player_2_args_model_name": "gemini-1.5-flash",
        "model_names": "alice_gemini-1.5-flash_bob_gemini-1.5-flash",
        "market": "CI=True_MA=False_MR=1",
    },
    "bargaining": {
        "game_args_complete_information": True,
        "game_args_messages_allowed": False,
        "game_args_max_rounds": "12",
        "player_1_args_delta": "0.9",
        "player_2_args_delta": "0.9",
        "game_args_money_to_divide": "10000.0",
        # "models_names": "alice_gpt-3.5-turbo_bob_gpt-3.5-turbo",
        "player_1_args_model_name": "gemini-1.5-flash",
        "player_2_args_model_name": "gemini-1.5-flash",
        "model_names": "alice_gemini-1.5-flash_bob_gemini-1.5-flash",
        "market": "CI=True_MA=False_MR=inf",
    }
}

    
t_bar = tqdm(total=len(METRICS) * (len(FAMILIES) + int(args.myopic_split == True)))


for metric in METRICS:
    for family in FAMILIES:
        split_options = [""]
        if args.myopic_split and family == "persuasion":
            split_options = ["myopic", "non_myopic"]
        for split in split_options:
            games_data = prepare_dataset_for_task(data, family, metric, split)
            X = games_data.drop(columns=["result"])
            y = games_data["result"].values

            model = StatsModelOfOneHots()
            
            forced_baselines = forced_baselines_per_family.get(family, {}).copy()
            if split == "myopic":
                if "game_args_is_myopic" in forced_baselines:
                    forced_baselines["game_args_is_myopic"] = "True"
                if "market" in forced_baselines and family == "persuasion":
                    forced_baselines["market"] = forced_baselines["market"].replace("MYOPIC=False", "MYOPIC=True")
            elif split == "non_myopic":
                if "game_args_is_myopic" in forced_baselines:
                    forced_baselines["game_args_is_myopic"] = "False"
                if "market" in forced_baselines and family == "persuasion":
                    forced_baselines["market"] = forced_baselines["market"].replace("MYOPIC=True", "MYOPIC=False")

            model.fit(X, y, forced_baselines=forced_baselines)

            df_effects = model.get_params_and_ci_relative_to_baseline()
            family_split = family + "_" + split if split else family

            for _, row in df_effects.iterrows():
                coef_tuple = (
                    family_split,
                    metric,
                    row["param"],
                    row["value"],
                    row["coef"],
                    row["ci_low"],
                    row["ci_high"]
                )
                all_coefs.append(coef_tuple)
                
            t_bar.update(1)

# save to csv
df = pd.DataFrame(all_coefs, columns=["family", "metric", "paramter_coef", "value", "effect", "ci_low", "ci_high"])
os.makedirs("output/analyze_coefs", exist_ok=True)
df.to_csv(f"output/analyze_coefs/{args.exp_name}.csv", index=False)

def analyze_coefs(args):
    # Create output directory if it doesn't exist
    os.makedirs("output/analyze_coefs", exist_ok=True)
    
    # Load the data
    df = pd.read_csv(f"output/analyze_coefs/{args.exp_name}.csv")
    
    # Save the analyzed coefficients
    df.to_csv(f"output/analyze_coefs/{args.exp_name}_analyzed.csv", index=False)

