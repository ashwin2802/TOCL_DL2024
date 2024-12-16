import json

with open('data/valid_environments.json', 'r') as f: 
    valid_environments = json.load(f)

with open('data/full_rl_env_mapping.json', 'r') as f: 
    full_rl_env_mapping = json.load(f)

counts = {}

for env in valid_environments: 
    num_actions = full_rl_env_mapping[env][1]

    if num_actions in counts.keys(): 
        counts[num_actions].append(env)
    else: 
        counts[num_actions] = [env]

with open('data/counts.json', 'w') as f: 
    json.dump(counts, f, indent=4)