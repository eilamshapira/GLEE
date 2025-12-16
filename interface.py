import streamlit as st

from interface_app.state import initialize_session_state, reset_to_main_menu
from interface_app.views import (
    render_analyze_results,
    render_continue_existing_experiment,
    render_initial_choice,
    render_run_new_experiment,
    render_competition,
)


def main():
    st.set_page_config(layout="wide")
    st.title("GLEE Experimentation Platform")
    initialize_session_state()

    if st.session_state.initial_choice_made:
        if st.sidebar.button("Back to Main Page"):
            reset_to_main_menu()
            st.rerun()

    if not st.session_state.initial_choice_made:
        render_initial_choice()
        return

    mode = st.session_state.experiment_mode
    if mode == "Run New Experiment":
        render_run_new_experiment()
    elif mode == "Continue Existing Experiment":
        render_continue_existing_experiment()
    elif mode == "Analyze Results":
        render_analyze_results()
    elif mode == "Competition":
        render_competition()
    else:
        st.error(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
