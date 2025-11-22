import os
import streamlit as st


def render_initial_choice():
    choice = st.radio(
        "Choose an Option",
        ("Run New Experiment", "Continue Existing Experiment", "Analyze Results"),
    )

    if st.button("Proceed"):
        st.session_state.initial_choice_made = True
        st.session_state.experiment_mode = choice
        st.rerun()

    st.markdown("---")
    st.subheader("Configuration")
    
    with st.expander("Edit API Keys and Environment Variables", expanded=False):
        file_path = "litellm/init_litellm.sh"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            exports = []
            others = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("export "):
                    rest = stripped[7:]
                    if "=" in rest:
                        key, value = rest.split("=", 1)
                        exports.append({"Key": key.strip(), "Value": value.strip()})
                    else:
                        others.append(line)
                elif stripped: # Keep non-empty other lines
                    others.append(line)
            
            st.write("Edit Environment Variables (lines starting with 'export'). This will overwrite the existing litellm/init_litellm.sh file.")
            edited_exports = st.data_editor(exports, num_rows="dynamic", key="exports_editor", width='stretch')
            
            if st.button("Save init_litellm.sh"):
                new_content = ""
                # Write exports
                for item in edited_exports:
                    k = item.get("Key", "").strip()
                    v = item.get("Value", "").strip()
                    if k:
                        new_content += f"export {k}={v}\n"
                
                # Write other lines (comments, etc)
                if others:
                    new_content += "\n"
                    for line in others:
                        new_content += line
                
                with open(file_path, "w") as f:
                    f.write(new_content)
                st.success("File saved successfully!")
        else:
            st.error(f"File not found: {file_path}")
