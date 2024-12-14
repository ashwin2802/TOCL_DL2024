import gym
import pandas as pd

def extract_latest_env_info():
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
            # Handle parsing issues
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

# Extract and save information for the latest environment versions
latest_env_info_df = extract_latest_env_info()
latest_env_info_df.to_csv("gym_latest_envs_action_observation_dimensions.csv", index=False)
print(latest_env_info_df)
