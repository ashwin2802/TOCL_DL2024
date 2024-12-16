import gym
import pandas as pd
import json

# Function to filter environments based on specific criteria
def extract_filtered_envs():
    # Dictionary to store the latest version of each environment
    latest_envs = {}
    
    for env_spec in gym.envs.registry.all():
        try:
            env_id = env_spec.id
            
            # Extract base name and version number
            base_name, version = env_id.rsplit('-v', 1)
            version = int(version)  # Convert version to integer
            
            # Update the latest version for each base name
            if base_name not in latest_envs or version > latest_envs[base_name]["version"]:
                latest_envs[base_name] = {"id": env_id, "version": version}
        except Exception as e:
            print(f"Could not process {env_spec.id}: {e}")
    
    # Extract detailed information for the latest environments
    env_data = []
    for env_info in latest_envs.values():
        try:
            env_id = env_info["id"]
            env = gym.make(env_id)
            
            # Observation space dimensions
            obs_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else None
            
            # Action space dimensions
            if isinstance(env.action_space, gym.spaces.Discrete):
                num_actions = env.action_space.n  # Discrete action space
            elif isinstance(env.action_space, gym.spaces.Box):
                num_actions = env.action_space.shape[0]  # Continuous action space
            else:
                num_actions = None  # Unusual action space
            
            # Append data
            env_data.append({
                "Environment": env_id,
                "Observation Dim": obs_dim,
                "Number of Actions": num_actions,
                "Action Space Type": type(env.action_space).__name__,
                "Observation Space Type": type(env.observation_space).__name__,
            })
            
            env.close()
        except Exception as e:
            print(f"Could not load {env_id}: {e}")
    
    return pd.DataFrame(env_data)

# Extract and filter based on the criteria
env_df = extract_filtered_envs()


# Filter environments based on 'Discrete' and 'Box'
filtered_df = env_df[
    (env_df["Action Space Type"] == "Discrete") &
    (env_df["Observation Space Type"] == "Box")
]

# Create a dictionary mapping env_id to (input_size, num_classes)
env_mapping = {
    row["Environment"]: (int(row["Observation Dim"]), int(row["Number of Actions"]))
    for _, row in filtered_df.iterrows()
}

# Save the mapping to a JSON file
json_path = "data/full_rl_env_mapping.json"
with open(json_path, "w") as json_file:
    json.dump(env_mapping, json_file, indent=4)

