import os
import re
from collections import defaultdict
from subprocess import Popen

import analyze.main as analyze_main
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st


def sanitize_experiment_name(experiment_name):
    return re.sub(r'\W+', '', experiment_name)


def prepare_analysis_inputs(experiment_names, market_mode, models_interaction, myopic_split):
    all_exp_names = []
    config_files = defaultdict(list)

    for experiment_name in experiment_names:
        sanitized_name = sanitize_experiment_name(experiment_name)
        all_exp_names.append(sanitized_name)
        base_path = f"Data/{sanitized_name}"

        game_types = [
            game_type
            for game_type in ["bargaining", "persuasion", "negotiation"]
            if os.path.exists(os.path.join(base_path, f"{game_type}"))
        ]

        for game_type in game_types:
            config_path = f"output/configs/{experiment_name}_{game_type}_with_stats.csv"
            if not os.path.exists(config_path):
                analyze_main.create_configs_file(
                    game_type,
                    first_eligible_commit=None,
                    data_path="Data",
                    exp_name=experiment_name,
                    include_human=False,
                )
                analyze_main.create_config_with_stats(game_type, data_path="Data", exp_name=experiment_name)
                st.write(
                    f"Created config file for **{game_type}** games in **{config_path}** in the experiment **{experiment_name}**"
                )
            else:
                st.write(
                    f"Config file for **{game_type}** games in the experiment **{experiment_name}** already exists in **{config_path}**"
                )
            config_files[game_type].append(config_path)

    configs_files_str = []
    for game_family, file_paths in config_files.items():
        if file_paths:
            configs_files_str.append(f"--{game_family}")
            configs_files_str.extend(file_paths)

    all_exp_names = "_".join(sorted(all_exp_names))
    query_params = {}

    if market_mode:
        configs_files_str.append("--merge_features=market")
        all_exp_names += "_markets"
        query_params['merge_features'] = 'market'
    if myopic_split:
        configs_files_str.append("--myopic_split")
        all_exp_names += "_myopic_split"
        query_params['myopic_split'] = True
    if models_interaction:
        configs_files_str.append("--models_interaction")
        all_exp_names += "_models_interaction"
        query_params['models_interaction'] = True

    return all_exp_names, config_files, configs_files_str, query_params


def run_analysis(exp_name, configs_files_str):
    if os.path.exists(f"output/analyze_coefs/{exp_name}.csv"):
        return 0
    process = Popen(['python', 'analyze/ML.py', '--exp_name', exp_name, *configs_files_str])
    return process.wait()


@st.cache_data
def load_data(exp_name):
    return pd.read_csv(f"output/analyze_coefs/{exp_name}.csv")


def get_statistic_path(family, configs_paths):
    stat_path = ""
    for path in sorted(configs_paths):
        if stat_path:
            stat_path += "_"
        if not path.startswith("output/"):
            path = f"output/configs/{os.path.basename(path)}"
        tmp_path = path.replace("output/configs/", "")
        tmp_path = tmp_path.split(".")[:-1]
        tmp_path = ".".join(tmp_path)
        stat_path += tmp_path

    stat_path = f"output/basic_statistics/{family}_" + stat_path
    stat_path += "/statistics.csv"
    return stat_path


def build_models_heatmap(filtered_data):
    tmp_df = filtered_data.copy()
    tmp_df["value"] = tmp_df["value"].str.replace("alice_", "")
    tmp_df[["model_a", "model_b"]] = tmp_df["value"].str.split("_bob_", expand=True, n=1)
    tmp_df = (
        tmp_df.groupby(["model_a", "model_b"], as_index=False)["effect"].mean().pivot(
            index="model_a", columns="model_b", values="effect"
        )
    )
    tmp_df -= tmp_df.mean().mean()
    return tmp_df


def expand_market_dataframe(filtered_data):
    def split_kv_string(s):
        return dict(part.split("=") for part in s.split("_"))

    expanded = filtered_data['value'].apply(split_kv_string).apply(pd.Series)
    tmp_df = pd.concat([expanded, filtered_data.drop("value", axis=1)], axis=1)

    for col in expanded.columns:
        if set(tmp_df[col].dropna().unique()) <= {"True", "False"}:
            tmp_df[col] = tmp_df[col].map({"True": True, "False": False})

    return tmp_df


def export_models_latex(df, selected_metric):
    export_df = df.copy()
    col_ids = list(range(1, len(export_df.columns) + 1))
    export_df.columns = [f"\\makebox[8pt]{{{i:X}}}" for i in col_ids]
    row_ids = list(range(1, len(export_df.index) + 1))
    export_df.index = [
        f"{i:X}. {name}" if selected_metric != "fairness" else f"{i:X}."
        for i, name in zip(row_ids, export_df.index)
    ]

    cmap = cm.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=export_df.min().min(), vmax=export_df.max().max())
    colorbar_width = 8 * len(export_df.columns)

    latex_core = (
        "\\noindent\n"
        "\\begin{minipage}{\\linewidth}\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1}\n"
        "\\setlength{\\tabcolsep}{1pt}\n"
        "\\begin{tabular}{l" + "c" * len(export_df.columns) + "}\n"
        " & " + " & ".join(export_df.columns) + " \\\\n\\hline\n"
    )

    for idx, row in export_df.iterrows():
        line = [f"{idx}"]
        for val in row:
            if pd.isna(val):
                cell = ""
            else:
                r, g, b = [int(255 * c) for c in cmap(norm(val))[:3]]
                cell = f"\\cellcolor[RGB]{{{r},{g},{b}}}"
            line.append(cell)
        latex_core += " & ".join(line) + " \\\n"

    vmin = export_df.values.min()
    vmax = export_df.values.max()
    latex_core += (
        "& \\multicolumn{" + str(len(export_df.columns)) + "}{c}{\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "hide axis,\n"
        "scale only axis,\n"
        "height=0.15cm,\n"
        f"width={colorbar_width}pt,\n"
        "colormap name=coolwarm,\n"
        "colorbar horizontal,\n"
        f"point meta min={vmin:.2f},\n"
        f"point meta max={vmax:.2f},\n"
        "colorbar style={\n"
        f"width={colorbar_width}pt,\n"
        "height=0.15cm,\n"
        "scaled ticks=false,\n"
        f"xtick={{ {vmin:.2f}, {(vmin + vmax)/2:.2f}, {vmax:.2f} }},\n"
        "xticklabel style={font=\\scriptsize},\n"
        "xticklabel={\\pgfmathprintnumber[fixed,precision=2]{\\tick}},\n"
        "},\n"
        "]\n"
        f"\\addplot [draw=none] coordinates {{ ({vmin:.2f},0) ({vmax:.2f},0) }};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n} \\\n\\end{tabular}\n\\end{minipage}"
    )

    return latex_core


def export_market_latex(df):
    export_df = df.copy()
    true_sign = r"\checkmark"
    false_sign = r"--"

    def make_ma_binary(value):
        if value == "binary":
            return False
        if value == "text":
            return True
        return value

    for col in export_df.columns:
        export_df[col] = export_df[col].apply(lambda x: make_ma_binary(x))
        if export_df[col].dtype == bool:
            export_df[col] = export_df[col].map({True: true_sign, False: false_sign})
        export_df[col] = export_df[col].apply(lambda x: r"$\\infty$" if x == "inf" else x)
        if len(export_df[col].unique()) == 1:
            export_df = export_df.drop(columns=[col])

    def format_effect(row):
        delta = row["ci_high"] - row["effect"]
        effect = row["effect"] * 100
        delta = delta * 100
        effect_str = f"{effect:.1f}".rstrip('0').rstrip('.')
        delta_str = f"{delta:.1f}".rstrip('0').rstrip('.')
        return effect_str if delta == 0 else f"{effect_str} ± {delta_str}"

    export_df["effect ± CI"] = export_df.apply(format_effect, axis=1)
    for col in export_df.columns:
        export_df[col] = export_df[col].apply(lambda x: f"{x}".rstrip('0').rstrip('.') if isinstance(x, float) else x)

    export_df = export_df.drop(columns=["effect", "ci_low", "ci_high"])
    feature_cols = [col for col in export_df.columns if col != "effect ± CI"]
    export_df = export_df[feature_cols + ["effect ± CI"]]

    return export_df.to_latex(index=False, escape=False)


def export_generic_latex(filtered_data):
    export_df = filtered_data.copy()

    def format_effect(row):
        if np.isinf(row["effect"]):
            return r"$\\infty$" if row["effect"] > 0 else r"$-\\infty$"
        delta = row["ci_high"] - row["effect"]
        effect = row["effect"] * 100
        delta = delta * 100
        return f"{effect:.1f}".rstrip('0').rstrip('.') + " ± " + f"{delta:.1f}".rstrip('0').rstrip('.')

    export_df["effect ± CI"] = export_df.apply(format_effect, axis=1)
    export_df = export_df[["value", "effect ± CI"]]

    return export_df.to_latex(index=False, escape=False)


def run_basic_statistics(paths):
    processes = []
    for family, family_paths in paths.items():
        new_process = Popen(['python', 'analyze/basic_statistics.py', f'--{family}', *family_paths])
        processes.append(new_process)
    return [proc.wait() for proc in processes]
