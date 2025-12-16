import requests
import json
import os
from players.base_player import Player

class HTTPPlayer(Player):
    def __init__(self, url, public_name="HTTP Player", delta=None, player_id=0):
        super().__init__(public_name, delta, player_id)
        self.url = url
        self.messages = []
        self.system_message = ""
        self.api_key = self._load_api_key()

    def _load_api_key(self):
        keys_file = "http_keys.json"
        if os.path.exists(keys_file):
            try:
                with open(keys_file, "r") as f:
                    keys = json.load(f)
                # Try exact match or match without trailing slash
                return keys.get(self.url) or keys.get(self.url.rstrip('/'))
            except Exception as e:
                print(f"Error loading keys file: {e}")
        return None

    def add_message(self, message, role):
        self.messages.append({"role": role, "content": message})

    def get_text_answer(self, format_checker, decision=False):
        try:
            # Collect all serializable attributes of the player to represent game state/config
            # And merge with game_params if available
            
            payload_game_params = {
                "public_name": self.public_name,
                "delta": self.delta,
                "player_id": self.player_id,
                "rules": self.rules,
                "req_offer_text": self.req_offer_text,
                "req_decision_text": self.req_decision_text,
            }
            
            if hasattr(self, 'game_params') and self.game_params:
                payload_game_params.update(self.game_params)
            
            payload = {
                "messages": self.messages,
                "decision": decision,
                "game_params": payload_game_params
            }
            
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
                
            response = requests.post(f"{self.url}/chat", json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "")
            
            # Add the assistant's response to the history
            self.add_message(answer, "assistant")
            return answer
        except Exception as e:
            if isinstance(e, requests.exceptions.Timeout):
                raise e
            print(f"Error communicating with HTTP player: {e}")
            # Return a fallback or raise, depending on desired behavior. 
            # For now, returning empty string which might fail format check.
            return ""

    def clean_conv(self):
        self.messages = []
        if self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

    def set_system_message(self, message):
        self.system_message = message
        # If messages are empty or first is system, replace/set it
        if not self.messages:
            self.messages.append({"role": "system", "content": message})
        elif self.messages[0]["role"] == "system":
            self.messages[0]["content"] = message
        else:
            self.messages.insert(0, {"role": "system", "content": message})
