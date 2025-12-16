import requests
import argparse
import sys
import json
import os

def load_keys():
    keys_file = "http_keys.json"
    if os.path.exists(keys_file):
        try:
            with open(keys_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def check_player(url, key=None):
    try:
        # Ensure url has scheme
        if not url.startswith("http"):
            url = f"http://{url}"
        
        # Append /health if not present (assuming base url provided)
        base_url = url.rstrip('/')
        health_url = f"{base_url}/health"
        
        headers = {}
        if key:
            headers['X-API-Key'] = key
            
        response = requests.get(health_url, headers=headers, timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            # If a key was provided for this target, require an explicit auth OK
            if key:
                if data.get("auth") == "ok":
                    return True, "Alive & Authenticated"
                # key provided but server didn't confirm auth -> treat as auth failure
                return True, "Alive but Auth FAILED"
            # No key required/available for this target
            return True, "Alive (No Auth Check)"
        elif response.status_code == 401:
            return True, "Alive but Auth FAILED"
        else:
            return False, f"Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Check health of HTTP players")
    parser.add_argument('targets', nargs='+', help="List of URLs or ports (localhost assumed for ports) to check")
    args = parser.parse_args()
    
    keys = load_keys()

    print(f"{'Target':<30} | {'Status':<20} | {'Details'}")
    print("-" * 70)

    for target in args.targets:
        # If target is just a number, assume it's a port on localhost
        if target.isdigit():
            url = f"http://localhost:{target}"
        else:
            url = target
            
        # Find key for this url
        key = keys.get(url) or keys.get(url.rstrip('/'))
            
        is_alive, details = check_player(url, key)
        
        if not is_alive:
            status = "DEAD"
            color = "\033[91m" # Red
        elif "Auth FAILED" in details:
            status = "AUTH ERROR"
            color = "\033[93m" # Yellow/Orange
        else:
            status = "ALIVE"
            color = "\033[92m" # Green
            
        reset = "\033[0m"
        
        print(f"{target:<30} | {color}{status:<20}{reset} | {details}")

if __name__ == "__main__":
    main()
