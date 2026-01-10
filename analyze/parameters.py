import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import itertools
import re
import numpy as np  # Import numpy for numerical operations
import json

import matplotlib

matplotlib.use('Agg')

base_configs = {
    "Negotiation": {'game_args_seller_value': 1,
                    'game_args_buyer_value': 1.2,
                    'game_args_product_price_order': 10000,
                    'game_args_max_rounds': 30,
                    'game_args_messages_allowed': True,
                    'game_args_complete_information': True},
    "Persuasion": {'game_args_p': 0.5,
                   'game_args_c': 2,
                   'game_args_product_price': 10000,
                   'game_args_is_seller_know_cv': True,
                   'game_args_seller_message_type': "text",
                   'game_args_is_myopic': False
                   },
    "Bargaining": {'game_args_delta_1': 0.95,
                   'game_args_delta_2': 0.95,
                   'game_args_money_to_divide': 10000,
                   'game_args_max_rounds': 99,
                   'game_args_complete_information': True,
                   'game_args_messages_allowed': True}
}

pretty_x_features = {
    "Negotiation": {
        "game_args_seller_value": "$F_A$",
        "game_args_buyer_value": "$F_B$",
        "game_args_product_price_order": "$M$",
        "game_args_max_rounds": "$T$",
        "game_args_messages_allowed": "Messages Allowed",
        "game_args_complete_information": "Complete Information"
    },
    "Persuasion": {
        "game_args_p": "$p$",
        "game_args_v": "$v$",
        "game_args_product_price": "$M$",
        "game_args_is_myopic": "Buyer is Myopic",
        "game_args_is_seller_know_cv": "Complete Information",
        "game_args_seller_message_type": "Message Type"
    },
    "Bargaining": {
        "player_1_args_delta": "$\delta_A$",
        "player_2_args_delta": "$\delta_B$",
        "game_args_delta_1": "$\delta_A$",
        "game_args_delta_2": "$\delta_B$",
        "game_args_money_to_divide": "$M$",
        "game_args_max_rounds": "$T$",
        "game_args_complete_information": "Complete Information",
        "game_args_messages_allowed": "Messages Allowed"
    }
}

model_short_names = {
    "human": "human",
    "Qwen/Qwen2-7B-Instruct": "Qwen-2",
    'gemini-1.5-flash': "Gemini Flash",
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama-3',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'Llama-3.1',
}
model_orders = {model: i for i, model in enumerate(model_short_names.values())}


# Function to plot the effect of each categorical feature on the target variable
def plot_feature_effects(feature_name, coefficients, family, metric):
    # Filter the coefficients that belong to the current feature
    feature_coeffs = {k: v for k, v in coefficients.items() if k.startswith(feature_name)}

    # Extract the feature values (categories) and their effects
    categories = list(feature_coeffs.keys())
    effects = list(feature_coeffs.values())

    # Create a bar plot for the feature
    plt.figure(figsize=(10, 6))
    plt.bar(categories, effects)
    plt.xticks(rotation=90)
    plt.title(f"Effect of {feature_name} on Target Variable")
    plt.xlabel("Feature Values")
    plt.ylabel("Effect (Coefficient)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    os.makedirs(f"plots/FI/{family}/{metric}", exist_ok=True)
    plt.savefig(f"plots/FI/{family}/{metric}/{feature_name}.png")


def plot_feature_importance(family, metric, feature_importance):
    # Create output directory if it doesn't exist
    os.makedirs(f"output/plots/FI/{family}/{metric}", exist_ok=True)
    
    # Plot and save the feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(range(len(feature_importance)), feature_importance.index, rotation=45)
    plt.title(f"Feature Importance for {family} - {metric}")
    plt.tight_layout()
    plt.savefig(f"output/plots/FI/{family}/{metric}/feature_importance.png")
    plt.close()


# Traverse through directories to find model files

def create_single_plots_and_make_general_df():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    data_list = []
    for root, dirs, files in os.walk("model"):
        for file in files:
            if file.endswith("model.joblib"):
                with open(os.path.join(root, file), "rb") as f:
                    _, family, metric = root.split("/")
                    model = joblib.load(f)

                    # Check if 'metric' includes 'player' info
                    if 'player_1' in metric or 'player_2' in metric:
                        metric_parts = metric.split('_')
                        metric_name = '_'.join(metric_parts[:-1])
                        player_num = metric_parts[-1]
                        if player_num == '1':
                            player = 'Alice'
                        elif player_num == '2':
                            player = 'Bob'
                        else:
                            continue  # Skip if player number not recognized
                        metric = metric_name
                    else:
                        player = 'Both'  # If no player info, maybe it's combined metric

                    # Extract the coefficients from the model
                    coefficients = model.params.to_dict()
                    # Extract the covariance matrix of the parameters
                    cov_params = model.cov_params()

                    # Identify categorical features based on the coefficients
                    features = set([coef.split('[')[0] for coef in coefficients.keys() if '[' in coef])

                    model_features = [("player_1_args_model_name", "Alice"), ("player_2_args_model_name", "Bob")]
                    for hue_feature, player_name in model_features:
                        if player_name != player and player != 'Both':
                            continue  # Skip if player doesn't match

                        for x_feature in features:
                            if x_feature in [x[0] for x in model_features]:
                                continue
                            data = []
                            # Check if both hue_feature and x_feature exist in the features
                            if hue_feature in features and x_feature in features:
                                # Define regex patterns to match levels specific to the features
                                hue_pattern = re.compile(rf"{hue_feature}\[(.*?)\]")
                                x_pattern = re.compile(rf"{x_feature}\[(.*?)\]")

                                # Extract levels specifically matching the hue_feature and x_feature
                                hue_levels = set(
                                    [hue_pattern.search(coef).group(1) for coef in coefficients.keys() if
                                     hue_pattern.search(coef)]
                                )
                                x_levels = set(
                                    [x_pattern.search(coef).group(1) for coef in coefficients.keys() if
                                     x_pattern.search(coef)]
                                )

                                original_data_columns = model.model.data.xnames

                                # Check if the original data columns are available
                                #            if os.path.exists(f'{model_dir}/all_combinations.joblib'):

                                all_combinations_df = joblib.load(f'{root}/all_combinations.joblib')

                                if hue_feature in all_combinations_df.columns and x_feature in all_combinations_df.columns:
                                    all_hue_levels = set(all_combinations_df[hue_feature].unique())
                                    all_x_levels = set(all_combinations_df[x_feature].unique())
                                else:
                                    raise ValueError(
                                        f"One or both of the features '{hue_feature}' or '{x_feature}' are missing from the data.")

                                all_hue_levels = set(f"T.{level}" for level in all_hue_levels)
                                all_x_levels = set(f"T.{level}" for level in all_x_levels)
                                # Identify missing (reference) levels
                                missing_hue_levels = all_hue_levels - hue_levels
                                missing_x_levels = all_x_levels - x_levels

                                assert len(missing_hue_levels) <= 1, f"Multiple missing levels found for {hue_feature}"
                                assert len(missing_x_levels) <= 1, f"Multiple missing levels found for {x_feature}"

                                # Iterate over combinations of hue_feature and x_feature levels
                                for hue_level, x_level in itertools.product(all_hue_levels, all_x_levels):
                                    # if (x_level if x_level[-2:] != ".0" else x_level[:-2]) == f"T.{base_configs[family][x_feature]}": continue
                                    hue_key = f"{hue_feature}[{hue_level}]"
                                    x_key = f"{x_feature}[{x_level}]"
                                    interaction_key = f"{hue_feature}[{hue_level}]:{x_feature}[{x_level}]"
                                    interaction_key2 = f"{x_feature}[{x_level}]:{hue_feature}[{hue_level}]"

                                    # Extract the specific level effects of hue_feature, x_feature, and their interaction
                                    hue_effect = coefficients.get(hue_key,
                                                                  0) if hue_level not in missing_hue_levels else 0
                                    x_effect = coefficients.get(x_key, 0) if x_level not in missing_x_levels else 0
                                    interaction_effect = coefficients.get(interaction_key,
                                                                          coefficients.get(interaction_key2, 0))

                                    # Extract variances and covariances
                                    # Variance of x_effect
                                    if x_key in cov_params.index:
                                        var_x_effect = cov_params.loc[x_key, x_key]
                                    else:
                                        var_x_effect = 0

                                    # Variance of interaction_effect
                                    if interaction_key in cov_params.index:
                                        var_interaction_effect = cov_params.loc[interaction_key, interaction_key]
                                    elif interaction_key2 in cov_params.index:
                                        var_interaction_effect = cov_params.loc[interaction_key2, interaction_key2]
                                    else:
                                        var_interaction_effect = 0

                                    # Covariance between x_effect and interaction_effect
                                    if x_key in cov_params.index and interaction_key in cov_params.columns:
                                        cov_x_interaction = cov_params.loc[x_key, interaction_key]
                                    elif x_key in cov_params.index and interaction_key2 in cov_params.columns:
                                        cov_x_interaction = cov_params.loc[x_key, interaction_key2]
                                    else:
                                        cov_x_interaction = 0

                                    # Compute total variance and standard error
                                    total_variance = var_x_effect + var_interaction_effect + 2 * cov_x_interaction
                                    if total_variance < 0:
                                        total_variance = 0  # Ensure non-negative variance
                                    standard_error = np.sqrt(total_variance)

                                    # Compute 95% confidence intervals
                                    ci_lower = (x_effect + interaction_effect) - 1.96 * standard_error
                                    ci_upper = (x_effect + interaction_effect) + 1.96 * standard_error

                                    # Add data to the list
                                    data.append({
                                        'Family': family,
                                        'x_feature': x_feature,
                                        'Metric': metric,
                                        'Player': player_name,
                                        'hue_level': hue_level,
                                        'x_level': x_level,
                                        'hue_effect': hue_effect,
                                        'x_effect': x_effect,
                                        'interaction_effect': interaction_effect,
                                        'standard_error': standard_error,
                                        'ci_lower': ci_lower,
                                        'ci_upper': ci_upper
                                    })

                                # Create a DataFrame from the collected data
                                df = pd.DataFrame(data)

                                df["total_effect"] = df["x_effect"] + df["interaction_effect"]
                                # df = df[["x_level", "hue_level", "total_effect", "standard_error", "ci_lower", "ci_upper"]]

                                # drop missing x levels
                                df = df[~df["x_level"].isin(missing_x_levels)]

                                df["x_level"] = df["x_level"].apply(lambda x: x[2:] if x[:2] == "T." else x)
                                df["hue_level"] = df["hue_level"].apply(lambda x: x[2:] if x[:2] == "T." else x)
                                df['hue_level'] = df['hue_level'].apply(lambda x: model_short_names[x])

                                # sort by x_level
                                # First, map the 'hue_level' column to the custom order
                                df['hue_order'] = df['hue_level'].map(model_orders)
                                df = df.sort_values(by=["hue_order", "x_level"])
                                df = df.drop(columns=["hue_order"])

                                data_list.append(df)

                                # Plot the effects using seaborn
                                plt.figure(figsize=(12, 8), dpi=300)
                                ax = sns.barplot(data=df, x="x_level", y="total_effect", hue="hue_level", errorbar=None)

                                # Add error bars manually
                                x_coords = []
                                for p in ax.patches:
                                    x_coords.append(p.get_x() + p.get_width() / 2.0)

                                # Since seaborn doesn't provide the x positions directly, we need to calculate them
                                # Extract unique positions for each group
                                x_levels = sorted(df['x_level'].unique().tolist())

                                hue_levels = df['hue_level'].unique()
                                # hue_levels = sorted(hue_levels, key=lambda x: model_orders[x])

                                n_hue = len(hue_levels)
                                width = 0.8 / n_hue  # Adjust width based on number of hue levels

                                for i, hue_level in enumerate(hue_levels):
                                    hue_data = df[df['hue_level'] == hue_level]
                                    x_pos = np.arange(len(x_levels)) - 0.4 + width / 2 + i * width
                                    ax.errorbar(x=x_pos, y=hue_data['total_effect'],
                                                yerr=1.96 * hue_data['standard_error'],
                                                fmt='none', c='black', capsize=5)

                                # Plot zero line
                                plt.axhline(0, color='black', linestyle='--', linewidth=1)

                                # get the only item in set
                                baseline_x = list(missing_x_levels)[0]
                                baseline_x = baseline_x[2:] if baseline_x[:2] == "T." else baseline_x

                                plt.title(
                                    f"The effect of changing {pretty_x_features[family][x_feature]} on {metric} for different models"
                                    f"\ncompare to the case where {pretty_x_features[family][x_feature]} = {baseline_x}")

                                plt.xlabel(pretty_x_features[family][x_feature])
                                plt.ylabel(f"$\Delta$ in {metric}")

                                plt.tight_layout()
                                plt.legend(title="Model")

                                os.makedirs(f"plots/IE_v2/{family}/{metric}", exist_ok=True)
                                plt.savefig(f"plots/IE_v2/{family}/{metric}/{hue_feature}_{x_feature}.png")
                                plt.savefig(f"sample.png")
                                plt.close()
                                print(f"the plot saved to plots/IE_v2/{family}/{metric}/{hue_feature}_{x_feature}.png")

    all_data = pd.concat(data_list, ignore_index=True)
    all_data.to_csv("output/IE_v2_all_data.csv", index=False)
    return all_data


def make_grid_plots(all_data=None):
    if all_data is None:
        all_data = pd.read_csv("output/IE_v2_all_data.csv")

    for (family, x_feature), group_df in all_data.groupby(['Family', 'x_feature']):

        # Create a figure with 3 rows and 2 columns (transposed layout)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12),
                                 dpi=300)  # Increased figure size for better visibility
        fig.subplots_adjust(right=0.85)  # Adjust to make space for the legend

        # Set up row and column labels
        players = ['Alice', 'Bob']
        metrics = ['self_gain', 'efficiency', 'fairness']

        # Increase font size globally
        plt.rcParams.update({'font.size': 16})  # Change font size for better readability

        base_x_level = ""
        for i, player in enumerate(players):
            player_metrics = [f"{player}_Self Gain", f"{player}_Efficiency", f"{player}_Fairness"]
            for j, metric in enumerate(player_metrics):
                ax = axes[j, i]  # Transpose the subplot indexing (rows now correspond to metrics, columns to players)
                df_plot = group_df[(group_df['Player'] == player) & (group_df['Metric'] == metric)]

                if df_plot.empty:
                    ax.axis('off')  # Hide subplot if no data
                    continue

                # Plot the effects using seaborn, set color palette
                sns.barplot(ax=ax, data=df_plot, x="x_level", y="total_effect", hue="hue_level", errorbar=None,
                            palette='deep')

                # Add error bars manually
                x_levels = df_plot['x_level'].unique()
                hue_levels = df_plot['hue_level'].unique()
                n_hue = len(hue_levels)
                width = 0.8 / n_hue  # Adjust width based on number of hue levels

                all_combinations_df = joblib.load(f'model/{family}/{metric}/all_combinations.joblib')
                # set as categorical

                all_x_levels = all_combinations_df[x_feature].unique()
                base_x_level = set([str(s) for s in all_x_levels]) - set([str(s) for s in x_levels])
                base_x_level = list(base_x_level)[0]

                for k, hue_level in enumerate(hue_levels):
                    hue_data = df_plot[df_plot['hue_level'] == hue_level]
                    x_pos = np.arange(len(x_levels)) - 0.4 + width / 2 + k * width
                    ax.errorbar(x=x_pos, y=hue_data['total_effect'], yerr=1.96 * hue_data['standard_error'],
                                fmt='none', c='black', capsize=5)

                # Plot zero line
                ax.axhline(0, color='black', linestyle='--', linewidth=1)

                pretty_metric = metric.split("_")[1]
                if pretty_metric == "Self Gain":
                    pretty_metric = f"Self-Gain of {player}"
                ax.set_title(f"$\Delta$ in {pretty_metric} as factor of {player}'s model",
                             fontsize=18)  # Adjusted font size

                x_label = f"New value of {pretty_x_features[family].get(x_feature, x_feature)}"
                ax.set_xlabel(x_label, fontsize=18)
                ax.set_ylabel(f"$\Delta$ in {pretty_metric}", fontsize=18)

                ax.legend().set_visible(False)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Move legend to the right of all graphs
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', title="Model", fontsize=16)  # Larger legend font

        # Save the figure
        os.makedirs(f"plots/IE_v2_grid_plots/{family}", exist_ok=True)
        plt.savefig(f"plots/IE_v2_grid_plots/{family}/{x_feature}_grid_plot.png")
        plt.close()

        info = {"param_name": pretty_x_features[family][x_feature], "base_param_value": base_x_level}

        # save info to "plots/IE_v2_grid_plots/{family}/{x_feature}_grid_metadata.json"
        with open(f"plots/IE_v2_grid_plots/{family}/{x_feature}_grid_metadata.json", "w") as f:
            json.dump(info, f)
        print(f"The grid plot saved to plots/IE_v2_grid_plots/{family}/{x_feature}_grid_plot.png")


def create_row_of_plots(config_list, all_data=None):
    if all_data is None:
        all_data = pd.read_csv("output/IE_v2_all_data.csv")

    n = len(config_list)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n * 6, 4), dpi=300)

    # Ensure axes is iterable even if n == 1
    if n == 1:
        axes = [axes]

    # Initialize a variable to collect legend handles and labels
    legend_handles_labels = None

    for i, config in enumerate(config_list):
        family = config['family']
        x_feature = config['x_feature']
        player = config['player']
        metric = config['player_metric']  # Single metric
        ax = axes[i]

        # Filter the data based on the config parameters
        group_df = all_data[(all_data['Family'] == family) & (all_data['x_feature'] == x_feature)]
        if group_df.empty:
            ax.axis('off')  # Hide subplot if no data
            continue

        df_plot = group_df[(group_df['Player'] == player) & (group_df['Metric'] == metric)]
        if df_plot.empty:
            ax.axis('off')
            continue

        # Plot the effects using seaborn
        sns.barplot(ax=ax, data=df_plot, x="x_level", y="total_effect", hue="hue_level", errorbar=None, palette='deep')

        # Add error bars manually
        x_levels = df_plot['x_level'].unique()
        hue_levels = df_plot['hue_level'].unique()
        n_hue = len(hue_levels)
        width = 0.8 / n_hue  # Adjust width based on number of hue levels

        for k, hue_level in enumerate(hue_levels):
            hue_data = df_plot[df_plot['hue_level'] == hue_level]
            x_positions = np.arange(len(x_levels)) - 0.4 + width / 2 + k * width
            y_values = hue_data['total_effect'].values
            y_err = 1.96 * hue_data['standard_error'].values
            ax.errorbar(x=x_positions, y=y_values, yerr=y_err,
                        fmt='none', c='black', capsize=5)

        # Plot zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # Set titles and labels
        pretty_metric = metric.split('_')[1]
        if pretty_metric == "Self Gain":
            pretty_metric = f"Self-Gain of {player}"
        # ax.set_title(f"$\Delta$ in {pretty_metric} for {player}", fontsize=16)
        x_label = f"New value of {pretty_x_features[family].get(x_feature, x_feature)}"
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(f"$\Delta$ in {pretty_metric}", fontsize=14)

        # Remove individual legends
        ax.legend_.remove()

        # Collect legend handles and labels from the last plot
        if legend_handles_labels is None:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles_labels = (handles, labels)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin to make space

    # Place the legend to the right of the rightmost plot
    if legend_handles_labels is not None:
        handles, labels = legend_handles_labels
        fig.legend(handles, labels, loc='center right', title="Model", fontsize=12)

    plt.savefig("row_of_plots.pdf")


def create_figures():
    data = None
    data = create_single_plots_and_make_general_df()
    make_grid_plots(data)

    row_of_plots = [
        {
            'family': 'Negotiation',
            'x_feature': 'game_args_complete_information',
            'player': 'Alice',
            'player_metric': 'Alice_Efficiency'
        },
        {
            'family': 'Bargaining',
            'x_feature': 'game_args_messages_allowed',
            'player': 'Bob',
            'player_metric': 'Bob_Efficiency'
        }]
    create_row_of_plots(row_of_plots)


if __name__ == "__main__":
    create_figures()
