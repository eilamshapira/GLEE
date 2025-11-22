import json

import numpy as np
import pandas as pd
import streamlit as st

from interface_app.analysis import (
    build_models_heatmap,
    expand_market_dataframe,
    export_generic_latex,
    export_market_latex,
    export_models_latex,
    get_statistic_path,
    load_data,
    prepare_analysis_inputs,
    run_analysis,
    run_basic_statistics,
)
from interface_app.config_generation import closest_match
from interface_app.experiments import list_existing_results
from interface_app.state import reset_to_main_menu


def render_analyze_results():
    if st.session_state.step == 'model_selection':
        st.subheader("Analyze Results")
        existing_experiments = list_existing_results()
        if existing_experiments:
            experiment_names = st.multiselect("Select Experiments", existing_experiments)
            market_mode = st.checkbox("Market Mode", value=False)
            models_interaction = st.checkbox("Models Interaction", value=False)
            myopic_split = st.checkbox("Myopic Split", value=False)

            if st.button("Analyze Results", disabled=(len(experiment_names) == 0)):
                st.write("Analyzing results...")
                (
                    all_exp_names,
                    config_files,
                    configs_files_str,
                    query_params,
                ) = prepare_analysis_inputs(experiment_names, market_mode, models_interaction, myopic_split)
                return_code = run_analysis(all_exp_names, configs_files_str)
                if return_code == 0:
                    st.session_state.step = "Results"
                    st.query_params['step'] = 'Results'
                    st.query_params['final_exp_name'] = all_exp_names
                    st.query_params['config_files'] = json.dumps(dict(config_files))
                    for key, value in query_params.items():
                        st.query_params[key] = value
                    st.rerun()
                else:
                    st.error("Error running the analysis. Please check your logs.")
        else:
            st.info("No existing experiments found.")
            if st.button("Back to Main Menu"):
                reset_to_main_menu()
                st.rerun()

    elif st.session_state.step in ("Statistics", "ShowStatistics"):
        st.subheader("Statistics")
        configs_paths = json.loads(st.query_params['config_files'])

        if st.session_state.step == "Statistics":
            st.write("Statistics will be shown here.")
            return_codes = run_basic_statistics(configs_paths)
            for return_code in return_codes:
                if return_code != 0:
                    st.error("Error running the analysis. Please check your logs.")
                    break
            else:
                st.write("Statistics generated successfully.")
                st.session_state.step = "ShowStatistics"
                st.rerun()

        if st.session_state.step == "ShowStatistics":
            statistics = []
            for family, family_configs in configs_paths.items():
                statistic_file = get_statistic_path(family, family_configs)
                family_data = pd.read_csv(statistic_file, index_col=0)
                family_data = family_data.reset_index().rename(columns={"index": "model"})
                family_data["family"] = family
                statistics.append(family_data)
            statistics = pd.concat(statistics, ignore_index=True)

            choice = st.radio("Groupping Options", ("Group by Family", "Group by Game Type"))
            groupping_row = st.checkbox("Show totals")
            st.write(choice)

            if choice == "Group by Family":
                grouped_statistics = statistics[statistics["model"] == "total"].set_index("family")
                grouped_statistics = grouped_statistics.drop(columns=["model"])
            else:
                grouped_statistics = statistics[statistics["model"] != "total"].groupby("model").sum()
                grouped_statistics = grouped_statistics.drop(columns=["family"])

            if groupping_row:
                total_row = grouped_statistics.sum().rename("total")
                grouped_statistics = pd.concat([grouped_statistics, total_row.to_frame().T], ignore_index=False)

            grouped_statistics = grouped_statistics.applymap(lambda x: f"{x:,}" if isinstance(x, (int, float)) else x)
            st.write(grouped_statistics.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))
            st.write("###### Statistics Table")

        if st.sidebar.button("Back to Results"):
            st.session_state.step = "Results"
            st.query_params['step'] = 'Results'
            st.rerun()

    elif st.session_state.step == "Results":
        coef_df = load_data(st.query_params['final_exp_name'])

        st.sidebar.header("Selected Parameters")
        st.sidebar.write("Market Mode:", st.query_params.get('merge_features', "False"))
        st.sidebar.write("Myopic Split:", st.query_params.get('myopic_split', "False"))
        st.sidebar.write("Models Interaction:", st.query_params.get('models_interaction', "False"))

        if st.sidebar.button("Show Statistics"):
            st.session_state.step = "Statistics"
            st.rerun()

        st.sidebar.header("Filter Options")
        selected_family = st.sidebar.selectbox("Select Family", sorted(coef_df["family"].unique()))
        filtered_data = coef_df[coef_df["family"] == selected_family]

        selected_metric = st.sidebar.selectbox("Select Metric", sorted(filtered_data["metric"].unique()))
        filtered_data = filtered_data[filtered_data["metric"] == selected_metric]

        param_options = sorted(filtered_data["paramter_coef"].unique())
        if 'selected_param' not in st.session_state:
            st.session_state.selected_param = param_options[0]
        if st.session_state.selected_param not in param_options:
            st.session_state.selected_param = closest_match(st.session_state.selected_param, param_options)
        selected_param = st.sidebar.selectbox(
            "Select Parameter Coefficient",
            param_options,
            index=param_options.index(st.session_state.selected_param)
            if st.session_state.selected_param in param_options
            else 0,
            key="selected_param",
        )
        filtered_data = filtered_data[filtered_data["paramter_coef"] == selected_param]
        filtered_data = filtered_data.sort_values("effect", ascending=False)[["value", "effect", "ci_low", "ci_high"]]

        st.write(f"##### Effect of {selected_param} on {selected_metric} in {selected_family} Games")
        st.write("###### Ranking Table")

        if selected_param == "models_names":
            tmp_df = build_models_heatmap(filtered_data)
            styled_df = tmp_df.style.background_gradient(cmap='coolwarm', axis=None)
            styles = []
            for i in range(tmp_df.shape[1]):
                styles.extend(
                    [
                        {"selector": f"th.col{i}", "props": [("min-width", "3ch"), ("max-width", "3ch")]},
                        {"selector": f"td.col{i}", "props": [("min-width", "3ch"), ("max-width", "3ch")]},
                    ]
                )
            styled_df = styled_df.set_table_styles(styles)
            st.dataframe(styled_df)

            mean_value = tmp_df.mean().mean()
            st.write("Mean value")
            st.write(mean_value)
            value_ii = np.mean(np.array([tmp_df.iloc[i, i] for i in range(tmp_df.shape[0])]))
            st.write("Value II")
            st.write(value_ii)
            export_payload = tmp_df
        elif selected_param == "market":
            tmp_df = expand_market_dataframe(filtered_data)
            st.dataframe(tmp_df.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))
            export_payload = tmp_df
        else:
            st.dataframe(filtered_data.style.set_properties(**{'background-color': 'lightgrey'}, subset=pd.IndexSlice[::2, :]))
            export_payload = filtered_data

        if st.button("Export to LaTeX"):
            if selected_param == "models_names":
                latex_core = export_models_latex(export_payload, selected_metric)
            elif selected_param == "market":
                latex_core = export_market_latex(export_payload)
            else:
                latex_core = export_generic_latex(export_payload)
            st.code(latex_core, language="latex")

        if st.button("Back to Main Menu"):
            reset_to_main_menu()
            st.rerun()
