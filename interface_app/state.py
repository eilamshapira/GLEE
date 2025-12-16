import streamlit as st


def initialize_session_state():
    defaults = {
        'step': 'model_selection',
        'selected_models': [],
        'parameters': {},
        'initial_choice_made': False,
        'competition_step': 'setup',
        'competition_config': {},
        'competition_players': [],
        'competition_results': {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_to_main_menu():
    st.session_state.initial_choice_made = False
    st.session_state.step = 'model_selection'
    st.session_state.selected_models = []
    st.session_state.parameters = {}
