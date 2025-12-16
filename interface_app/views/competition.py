import streamlit as st
import json
import time
import os
import concurrent.futures
from http_player.check_players import check_player
from games.game_factory import game_factory
from players.player_factory import player_factory
from utils.utils import DataLogger
import pandas as pd
from datetime import datetime
from consts import OUTPUT_DIR, NATURE_NAME
from interface_app.game_parameters import game_parameters
from interface_app.config_generation import generate_configurations_without_models
from interface_app.models_manager import get_models

def run_game_wrapper(args):
    config, p1_def, p2_def, experiment_name = args
    try:
        # Merge player args
        p1_args = p1_def['args'].copy()
        if 'player_1_args' in config:
            p1_args.update(config['player_1_args'])
        
        # Force internal names for game logic
        p1_args['public_name'] = "Alice"

        p2_args = p2_def['args'].copy()
        if 'player_2_args' in config:
            p2_args.update(config['player_2_args'])
            
        # Force internal names for game logic
        p2_args['public_name'] = "Bob"

        p1 = player_factory(p1_def['type'], p1_args)
        p2 = player_factory(p2_def['type'], p2_args)
        
        timeout = config.get("game_args", {}).get("timeout", 30)
        p1.timeout = timeout
        p2.timeout = timeout
        
        # Add experiment_name to config for DataLogger
        config_with_exp = config.copy()
        config_with_exp['experiment_name'] = experiment_name
        
        dl = DataLogger(p1, p2, **config_with_exp)
        game_factory(config['game_type'], p1, p2, dl, config.get('game_args', {}))
        
        # Calculate Gain
        p1_gain = 0
        p2_gain = 0
        
        game_type = config['game_type'].lower()
        game_args = config.get('game_args', {})
        
        if game_type == "bargaining":
            if dl.actions:
                last_action = dl.actions[-1]
                if "decision" in last_action and last_action["decision"] == "accept":
                    if len(dl.actions) >= 2:
                        offer = dl.actions[-2]
                        # Extract raw gains
                        raw_p1 = 0
                        raw_p2 = 0
                        for k, v in offer.items():
                            if k.endswith("_gain"):
                                if p1.public_name.lower() in k.lower():
                                    raw_p1 = v
                                elif p2.public_name.lower() in k.lower():
                                    raw_p2 = v
                        
                        # Apply discount
                        rounds_played = last_action["round"]
                        if p1.delta is not None:
                            raw_p1 = raw_p1 * (p1.delta ** (rounds_played - 1))
                        if p2.delta is not None:
                            raw_p2 = raw_p2 * (p2.delta ** (rounds_played - 1))
                            
                        # Normalize
                        money = game_args.get('money_to_divide', 100)
                        if money > 0:
                            p1_gain = (raw_p1 / money) * 100
                            p2_gain = (raw_p2 / money) * 100
                            
        elif game_type == "negotiation":
            if dl.actions:
                last_action = dl.actions[-1]
                decision = last_action.get("decision")
                if decision == "AcceptOffer":
                    if len(dl.actions) >= 2:
                        offer = dl.actions[-2]
                        price = offer.get("product_price")
                        if isinstance(price, str):
                             try:
                                price = float(price.replace("$", ""))
                             except:
                                price = 0
                        
                        if price is not None:
                            seller_val = game_args.get("seller_value", 0)
                            buyer_val = game_args.get("buyer_value", 0)
                            price_order = game_args.get("product_price_order", 1)
                            
                            seller_cost = seller_val * price_order
                            buyer_valuation = buyer_val * price_order
                            
                            raw_p1 = price - seller_cost
                            raw_p2 = buyer_valuation - price
                            
                            # Normalize by monetary scale (product_price_order)
                            if price_order > 0:
                                p1_gain = (raw_p1 / price_order) * 100
                                p2_gain = (raw_p2 / price_order) * 100

        elif game_type == "persuasion":
            price = game_args.get("product_price", 10)
            total_rounds = game_args.get("total_rounds", 20)
            
            current_worth = 0
            seller_raw = 0
            buyer_raw = 0
            
            for action in dl.actions:
                player_name = action.get("player")
                if player_name == NATURE_NAME:
                    current_worth = action.get("product_worth", 0)
                elif player_name == p2.public_name: # Buyer
                    decision = action.get("decision")
                    if decision == "yes":
                        seller_raw += price
                        buyer_raw += current_worth - price
            
            # Normalize by monetary scale (total_rounds * product_price)
            monetary_scale = total_rounds * price
            if monetary_scale > 0:
                p1_gain = (seller_raw / monetary_scale) * 100
                p2_gain = (buyer_raw / monetary_scale) * 100
        
        return {
            "config_id": str(config.get("game_args")), # Simple ID
            "p1": p1_def['args']['public_name'], # Use original display name
            "p2": p2_def['args']['public_name'], # Use original display name
            "p1_gain": p1_gain,
            "p2_gain": p2_gain,
            "status": "success"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

def render_competition():
    st.header("Competition Mode")
    
    if "competition_name" not in st.session_state:
        st.session_state.competition_name = "default_competition"
        
    if "competition_results" not in st.session_state:
        st.session_state.competition_results = None
        
    st.session_state.competition_name = st.text_input("Competition Name", value=st.session_state.competition_name)

    # Use radio button for navigation to control execution flow
    page = st.radio("Navigation", ["Configuration", "Players", "Lobby", "Run", "Results", "History"], horizontal=True)
    
    # --- Configuration ---
    if page == "Configuration":
        st.subheader("Game Configuration")
        
        if "competition_configs" not in st.session_state:
            st.session_state.competition_configs = []

        uploaded_config = st.file_uploader("Load Configurations (JSON)", type="json", key="comp_config_load")
        if uploaded_config:
            loaded = json.load(uploaded_config)
            if isinstance(loaded, list):
                st.session_state.competition_configs = loaded
                st.success(f"Loaded {len(loaded)} configurations!")
            else:
                st.error("Invalid format. Expected a list of configurations.")

        st.markdown("---")
        st.write("### Add New Configurations")
        
        game_type = st.selectbox("Select Game Type", ["Bargaining", "Negotiation", "Persuasion"], key="comp_game_type")
        
        # Use existing game_parameters function to populate session_state.parameters
        game_parameters(game_type, single_value=True)
        
        if st.button("Generate Configurations"):
            params = st.session_state.parameters.get(game_type, {})
            new_configs = generate_configurations_without_models(params)
            
            # Add game_type to each config
            for config in new_configs:
                config["game_type"] = game_type.lower()
                # Ensure game_args has timeout
                if "game_args" not in config:
                    config["game_args"] = {}
                config["game_args"]["timeout"] = 30 # Default timeout
            
            st.session_state.competition_configs.extend(new_configs)
            st.success(f"Generated {len(new_configs)} configurations!")

        st.markdown("---")
        st.write("### Current Configurations")
        if st.session_state.competition_configs:
            # Display as list of JSONs instead of dataframe
            for i, config in enumerate(st.session_state.competition_configs):
                with st.expander(f"Config {i+1}: {config.get('game_type')}"):
                    st.json(config)
            
            if st.button("Clear All Configurations"):
                st.session_state.competition_configs = []
                st.rerun()
                
            st.download_button(
                "Save Configurations", 
                data=json.dumps(st.session_state.competition_configs, indent=4), 
                file_name="competition_configs.json", 
                mime="application/json"
            )
        else:
            st.info("No configurations added yet.")

    # --- Players ---
    elif page == "Players":
        st.subheader("Players Configuration")
        
        uploaded_players = st.file_uploader("Load Players (JSON)", type="json", key="comp_players_load")
        if uploaded_players:
            st.session_state.competition_players = json.load(uploaded_players)
            st.success("Players loaded!")
            
        players = st.session_state.competition_players
        
        with st.expander("Add New Player"):
            p_name = st.text_input("Name")
            p_type = st.selectbox("Type", ["http", "litellm", "terminal"])
            
            p_args_str = "{}"
            if p_type == "http":
                p_args_str = '{"url": "http://localhost:5000"}'
                st.text_area("Arguments (JSON)", value=p_args_str, key="p_args_http")
            elif p_type == "litellm":
                model_name = st.selectbox("Model Name", get_models())
                p_args_str = json.dumps({"model_name": model_name})
                st.text_area("Arguments (JSON)", value=p_args_str, key="p_args_litellm", disabled=True)
            else:
                p_args_str = '{"model_name": "gpt-4"}'
                st.text_area("Arguments (JSON)", value=p_args_str, key="p_args_terminal")

            if st.button("Add Player"):
                try:
                    # For litellm, we construct the args from the selectbox if not manually edited (which is disabled above)
                    if p_type == "litellm":
                         p_args = {"model_name": model_name}
                    else:
                        # We need to grab the value from the correct text area based on type
                        if p_type == "http":
                            val = st.session_state.p_args_http
                        else:
                            val = st.session_state.p_args_terminal
                        p_args = json.loads(val)
                        
                    p_args["public_name"] = p_name
                    players.append({"type": p_type, "args": p_args})
                    st.success(f"Added {p_name}")
                except Exception as e:
                    st.error(f"Invalid JSON or Error: {e}")
        
        st.write("Current Players:")
        st.json(players)
        
        if st.button("Clear Players"):
            st.session_state.competition_players = []
            st.rerun()
            
        st.session_state.competition_players = players
        st.download_button("Save Players", data=json.dumps(players, indent=4), file_name="competition_players.json", mime="application/json")

    # --- Lobby ---
    elif page == "Lobby":
        st.subheader("Lobby")
        players = st.session_state.competition_players
        
        # Auto-refresh logic
        if "last_check_time" not in st.session_state:
            st.session_state.last_check_time = time.time()
            
        col_btn, col_timer = st.columns([1, 4])
        with col_btn:
            if st.button("Check Connectivity Now"):
                st.session_state.last_check_time = time.time() # Reset timer
                st.rerun()
        
        with col_timer:
            # Placeholder for timer
            timer_placeholder = st.empty()
            
        # Simple approach: Just show time since last check.
        time_since = int(time.time() - st.session_state.last_check_time)
        next_check = 30 - time_since
        
        if next_check <= 0:
             st.session_state.last_check_time = time.time()
             st.rerun()
        
        timer_placeholder.metric("Next check in", f"{next_check}s")
        
        if not players:
            st.warning("No players loaded.")
        else:
            active_players = []
            cols = st.columns(6)
            for i, p in enumerate(players):
                with cols[i % 6]:
                    status_color = "gray"
                    status_text = "Unknown"
                    is_alive = False
                    
                    if p["type"] == "http":
                        url = p["args"].get("url", "")
                        
                        # Let's store status in session_state
                        status_key = f"status_{p['args'].get('public_name')}"
                        if status_key not in st.session_state or next_check >= 29: # Just checked or button pressed
                             is_alive_check, msg = check_player(url)
                             st.session_state[status_key] = (is_alive_check, msg)
                        
                        is_alive, msg = st.session_state[status_key]
                        
                        if is_alive:
                            status_color = "green"
                            status_text = "Online"
                        else:
                            status_color = "red"
                            status_text = "Offline"
                    else:
                        status_color = "blue"
                        status_text = "Ready"
                        is_alive = True
                    
                    if is_alive:
                        active_players.append(p)
                        
                    st.markdown(f"""
                    <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; color: white; text-align: center; margin-bottom: 10px;">
                        <strong>{p['args'].get('public_name', 'Unknown')}</strong><br>
                        {p['args'].get('url', p['type'])}<br>
                        {status_text}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Competition Statistics")
            
            num_configs = len(st.session_state.competition_configs)
            num_active = len(active_players)
            
            if num_active > 1:
                total_games = num_active * (num_active - 1) * num_configs
                games_per_player = 2 * (num_active - 1) * num_configs
            else:
                total_games = 0
                games_per_player = 0
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Connected Players", num_active)
            c2.metric("Configurations", num_configs)
            c3.metric("Games / Player", games_per_player)
            c4.metric("Total Games", total_games)
            
            if st.button("Start Competition"):
                if num_active < 2:
                    st.error("Need at least 2 connected players to start.")
                else:
                    st.session_state.active_competition_players = active_players
                    st.session_state.competition_step = "running"
                    # Switch to Run tab automatically
                    # We can't easily switch radio button programmatically without session state hack
                    # But user can click "Run" manually or we can use session state for page
                    st.info("Competition Started! Go to 'Run' tab.")

        # Only auto-refresh if we are not running a competition
        if st.session_state.get("competition_step") != "running":
             time.sleep(1)
             st.rerun()

    # --- Run ---
    elif page == "Run":
        st.subheader("Running Competition")
        
        num_parallel = st.number_input("Number of parallel games", min_value=1, value=1)
        
        if st.session_state.competition_step == "running":
            configs = st.session_state.competition_configs
            players_def = st.session_state.get("active_competition_players", st.session_state.competition_players)
            
            if not configs:
                st.error("No configurations loaded!")
                st.stop()
                
            pairs = []
            for i in range(len(players_def)):
                for j in range(len(players_def)):
                    if i != j:
                        pairs.append((players_def[i], players_def[j]))
            
            if st.button("Start Execution"):
                tasks = []
                for config in configs:
                    for (p1_def, p2_def) in pairs:
                        tasks.append((config, p1_def, p2_def, st.session_state.competition_name))
                
                total_games = len(tasks)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
                    futures = [executor.submit(run_game_wrapper, task) for task in tasks]
                    
                    completed_count = 0
                    for future in concurrent.futures.as_completed(futures):
                        completed_count += 1
                        res = future.result()
                        progress_bar.progress(completed_count / total_games)
                        status_text.text(f"Completed {completed_count}/{total_games} games")
                        
                        if res["status"] == "success":
                            results.append(res)
                        else:
                            st.error(f"Game failed: {res['error']}")
                
                st.session_state.competition_results = results
                st.session_state.competition_step = "finished"
                
                # Save competition data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_name = st.session_state.competition_name
                save_dir = os.path.join(OUTPUT_DIR, "competitions", f"{comp_name}_{timestamp}")
                os.makedirs(save_dir, exist_ok=True)
                
                with open(os.path.join(save_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=4)
                    
                with open(os.path.join(save_dir, "config.json"), "w") as f:
                    json.dump({
                        "configs": st.session_state.competition_configs,
                        "players": st.session_state.get("active_competition_players", st.session_state.competition_players),
                        "name": comp_name,
                        "timestamp": timestamp
                    }, f, indent=4)
                
                st.success(f"Competition Finished! Saved to {save_dir}")

    # --- Results ---
    elif page == "Results":
        st.subheader("Results")
        if st.session_state.competition_results:
            results = st.session_state.competition_results
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            player_gains = {}
            for r in results:
                p1 = r["p1"]
                p2 = r["p2"]
                player_gains[p1] = player_gains.get(p1, []) + [r["p1_gain"]]
                player_gains[p2] = player_gains.get(p2, []) + [r["p2_gain"]]
            
            leaderboard = []
            for p, gains in player_gains.items():
                leaderboard.append({"Player": p, "Average Gain": sum(gains)/len(gains) if gains else 0})
            
            lb_df = pd.DataFrame(leaderboard).sort_values("Average Gain", ascending=False)
            st.write("Leaderboard")
            st.dataframe(lb_df)
            
            st.download_button("Download Leaderboard", lb_df.to_csv(index=False), "leaderboard.csv", "text/csv")

    # --- History ---
    elif page == "History":
        st.subheader("History")
        
        comp_dir = os.path.join(OUTPUT_DIR, "competitions")
        if not os.path.exists(comp_dir):
            st.info("No competition history found.")
        else:
            competitions = sorted([d for d in os.listdir(comp_dir) if os.path.isdir(os.path.join(comp_dir, d))], reverse=True)
            if not competitions:
                st.info("No competition history found.")
            else:
                selected_comp = st.selectbox("Select Competition", competitions)
                
                if st.button("Load Competition"):
                    load_dir = os.path.join(comp_dir, selected_comp)
                    try:
                        with open(os.path.join(load_dir, "results.json"), "r") as f:
                            st.session_state.competition_results = json.load(f)
                            
                        with open(os.path.join(load_dir, "config.json"), "r") as f:
                            data = json.load(f)
                            st.session_state.competition_configs = data.get("configs", [])
                            st.session_state.competition_players = data.get("players", [])
                            st.session_state.competition_name = data.get("name", "loaded_competition")
                            
                        st.success(f"Loaded competition: {selected_comp}")
                        st.info("Go to 'Results' tab to view details.")
                    except Exception as e:
                        st.error(f"Error loading competition: {e}")
