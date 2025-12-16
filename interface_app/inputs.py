import streamlit as st


def get_range_input(label, min_val=0.0, max_val=100.0, step=10.0, game_name=""):
    start, stop = st.slider(
        f"{label} Range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step,
        key=f"{game_name}_{label}_range",
    )
    interval = st.number_input(
        f"{label} Interval",
        min_value=0.01,
        max_value=max_val,
        value=step,
        step=0.01,
        key=f"{game_name}_{label}_interval",
    )

    values = []
    current = start
    while current <= stop:
        values.append(round(current, 2))
        current += interval

    return values


def get_numeric_input(label, min_val=0.0, max_val=1000.0, step=10.0, game_name="", default_options=None, astype=int):
    default_options = default_options or []
    mode = "Manual Values"

    if mode == "Range":
        start, stop = st.slider(
            f"{label} Range",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step,
            key=f"{game_name}_{label}_range",
        )
        interval = st.number_input(
            f"{label} Interval",
            min_value=0.01,
            max_value=max_val,
            value=step,
            step=0.01,
            key=f"{game_name}_{label}_interval",
        )

        values = []
        current = start
        while current <= stop:
            values.append(round(current, 2))
            current += interval

        return values

    values_key = f"{game_name}_{label}_values"
    edit_mode_key = f"{game_name}_{label}_edit_mode"
    new_input_key = f"{game_name}_{label}_new_input"
    checkbox_key = f"{game_name}_{label}_checkbox"

    if values_key not in st.session_state:
        st.session_state[values_key] = sorted(set(default_options))

    if edit_mode_key not in st.session_state:
        st.session_state[edit_mode_key] = False

    if not st.session_state[edit_mode_key]:
        options = st.session_state[values_key] + ["add..."]
        selected = st.multiselect(
            f"{label}:",
            options=options,
            default=st.session_state[values_key],
            key=checkbox_key,
        )

        if "add..." in selected:
            st.session_state[edit_mode_key] = True
            st.rerun()

        return [astype(v) for v in selected if v != "add..."]

    new_values_str = st.text_input(
        f"Enter new {label} values (comma-separated):",
        key=new_input_key,
    )

    if new_values_str.strip():
        try:
            new_values = sorted(set(astype(x.strip()) for x in new_values_str.split(',')))
            st.session_state[values_key] = sorted(set(st.session_state[values_key]).union(new_values))
            st.session_state[edit_mode_key] = False
            st.session_state.pop(new_input_key, None)
            st.rerun()
        except ValueError:
            st.error(f"Please enter valid numeric ({astype.__name__}) values separated by commas.")

    return []


def get_bool_input(label, game_name=""):
    options = ['True', 'False', 'Both']
    choice = st.selectbox(f"{label}", options, key=f"{game_name}_{label}")
    return [True, False] if choice == 'Both' else [choice == 'True']


def get_bool_multiselect(label, game_name="", options=None, default_options=None):
    options = options or [True, False]
    default_options = default_options if default_options is not None else options
    selected_options = st.multiselect(
        f"{label}",
        options,
        default=default_options,
        key=f"{game_name}_{label}",
    )
    return selected_options


def get_num_input(label, game_name=""):
    choice = st.number_input(
        label,
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key=f"{game_name}_max_rounds",
    )
    return [choice]


def get_single_numeric_input(label, min_val=0.0, max_val=1000.0, step=10.0, game_name="", default_options=None, astype=int):
    default_options = default_options or []
    key = f"{game_name}_{label}_single"
    
    if default_options:
        options = list(default_options) + ["Custom"]
        # Try to find a reasonable default index
        idx = 0
        if len(options) > 1:
             idx = 0
        
        selection = st.selectbox(label, options, index=idx, key=f"{key}_select")
        
        if selection == "Custom":
             val = st.number_input(f"{label} (Custom)", min_value=min_val, max_value=max_val, step=step, key=f"{key}_custom")
             return [astype(val)]
        else:
             return [astype(selection)]
    else:
        val = st.number_input(label, min_value=min_val, max_value=max_val, step=step, key=key)
        return [astype(val)]


def get_single_bool_input(label, game_name="", options=None, default_options=None):
    options = options or [True, False]
    key = f"{game_name}_{label}_single"
    # default_options is ignored for single selection, we just pick the first one or let user choose
    val = st.radio(label, options, key=key)
    return [val]

