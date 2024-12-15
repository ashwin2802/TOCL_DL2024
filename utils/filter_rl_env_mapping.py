import json

with open('data/full_rl_env_mapping.json', 'r') as f: 
    full_rl_env_mapping = json.load(f)

with open('data/selected_environments.json', 'r') as f: 
    selected_environments = json.load(f)

new_rl_env_mapping = {}
for env in selected_environments: 
    new_rl_env_mapping[env] = full_rl_env_mapping[env]

with open('data/rl_env_mapping.json', 'w') as f: 
    json.dump(new_rl_env_mapping, f, indent=4)