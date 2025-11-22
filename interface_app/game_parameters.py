import streamlit as st

from interface_app.inputs import (
    get_bool_multiselect,
    get_numeric_input,
)


def game_parameters(game_name):
    player_1_name = st.text_input("Player 1 Public Name", key=f"{game_name}_p1", value="Alice")
    player_2_name = st.text_input("Player 2 Public Name", key=f"{game_name}_p2", value="Bob")

    st.session_state.player_1_name = player_1_name
    st.session_state.player_2_name = player_2_name

    params = {}

    if game_name == "Persuasion":
        params.update(
            {
                'p': get_numeric_input(f"{game_name} p", game_name=game_name, default_options=[1 / 3, 0.5, 0.8], astype=float),
                'v': get_numeric_input(f"{game_name} v", game_name=game_name, default_options=[1.2, 1.25, 2, 3, 4], astype=float),
                'c': get_numeric_input(f"{game_name} c", game_name=game_name, default_options=[0], astype=float),
                'product_price': get_numeric_input(
                    f"{game_name} Product Price",
                    min_val=0.001,
                    max_val=1_000_000_000,
                    game_name=game_name,
                    default_options=[100, 1_00_00, 1_00_00_00],
                ),
                'total_rounds': get_numeric_input(f"{game_name} Total Rounds", game_name=game_name, default_options=[20]),
                'is_seller_know_cv': get_bool_multiselect(f"{game_name} Is Seller Know CV", game_name=game_name),
                'seller_message_type': get_bool_multiselect(
                    f"{game_name} Seller Message Type", game_name=game_name, options=['text', 'binary']
                ),
                'is_myopic': get_bool_multiselect(f"{game_name} Is Myopic", game_name=game_name),
            }
        )

    elif game_name == "Bargaining":
        params.update(
            {
                'player_1_delta': get_numeric_input(
                    f"{game_name} Player 1 Delta",
                    min_val=0.0,
                    max_val=1.0,
                    step=0.1,
                    game_name=game_name,
                    default_options=[0.8, 0.9, 0.95, 1],
                    astype=float,
                ),
                'player_2_delta': get_numeric_input(
                    f"{game_name} Player 2 Delta",
                    min_val=0.0,
                    max_val=1.0,
                    step=0.1,
                    game_name=game_name,
                    default_options=[0.8, 0.9, 0.95, 1],
                    astype=float,
                ),
                'money_to_divide': get_numeric_input(
                    f"{game_name} Product Price",
                    min_val=0.0,
                    max_val=1000.0,
                    game_name=game_name,
                    default_options=[100, 1_00_00, 1_00_00_00],
                ),
                'max_rounds': get_numeric_input(
                    f"{game_name} Max Rounds",
                    min_val=0,
                    max_val=99,
                    game_name=game_name,
                    default_options=[12, 25],
                    astype=int,
                ),
                'complete_information': get_bool_multiselect(f"{game_name} Complete Information", game_name=game_name),
                'messages_allowed': get_bool_multiselect(f"{game_name} Messages Allowed", game_name=game_name),
            }
        )

    elif game_name == "Negotiation":
        params.update(
            {
                'seller_value': get_numeric_input(
                    f"{game_name} Seller Value",
                    min_val=0.0,
                    max_val=1000.0,
                    game_name=game_name,
                    default_options=[0.8, 1, 1.2, 1.5],
                    astype=float,
                ),
                'buyer_value': get_numeric_input(
                    f"{game_name} Buyer Value",
                    min_val=0.0,
                    max_val=1000.0,
                    game_name=game_name,
                    default_options=[0.8, 1, 1.2, 1.5],
                    astype=float,
                ),
                'product_price_order': get_numeric_input(
                    f"{game_name} Product Price Order",
                    min_val=0.0,
                    max_val=10.0,
                    step=1.0,
                    game_name=game_name,
                    default_options=[100, 1_00_00, 1_00_00_00],
                ),
                'max_rounds': get_numeric_input(
                    f"{game_name} Max Rounds",
                    min_val=0,
                    max_val=99,
                    game_name=game_name,
                    default_options=[1, 10, 25],
                    astype=int,
                ),
                'complete_information': get_bool_multiselect(f"{game_name} Complete Information", game_name=game_name),
                'messages_allowed': get_bool_multiselect(f"{game_name} Messages Allowed", game_name=game_name),
            }
        )

    total_configurations = 1
    for param in params.values():
        total_configurations *= len(param) if isinstance(param, list) else 1

    if total_configurations > 0:
        st.write(f"Total configurations: {total_configurations}.")
        each_model_runs_per_config = st.session_state.run_per_model / total_configurations
        st.write(
            f"Each model will play each configuration {each_model_runs_per_config:.2f} times"
            f" ({each_model_runs_per_config / 2:.2f} as {player_1_name},"
            f" {each_model_runs_per_config / 2:.2f} as {player_2_name})."
        )
        st.write(
            f"Each configuration will be played"
            f" {len(st.session_state.selected_models) * st.session_state.run_per_model / total_configurations:.2f} times."
        )

    st.session_state.parameters[game_name] = {**params}
