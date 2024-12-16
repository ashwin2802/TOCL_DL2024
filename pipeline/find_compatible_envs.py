import torch
from torch.optim import Adam
from avalanche_rl.benchmarks.rl_benchmark_generators import gym_benchmark_generator

from avalanche_rl.models.actor_critic import ActorCriticMLP, MultiEnvActorCriticMLP
from avalanche_rl.training.strategies.actor_critic import A2CStrategy

import json
import csv

# Specify the file path
file_path = 'gym_latest_envs_action_observation_dimensions.csv'






selected_environments = [
    "ALE/Tetris-ram-v5",          # Puzzle/strategy
    "ALE/AirRaid-ram-v5",          # Classic control
    "ALE/Asterix-ram-v5",         # Action
    "ALE/Freeway-ram-v5",         # Reflex/racing
    "ALE/MsPacman-ram-v5",        # Maze-navigation
    "ALE/Kaboom-ram-v5",          # Reflex-based
    "ALE/BeamRider-ram-v5",       # Shooting/action
    "ALE/Centipede-ram-v5",       # Shooting/classic arcade
    "ALE/Frostbite-ram-v5",       # Platformer/survival
    "ALE/Riverraid-ram-v5",       # Flying/action
    "ALE/Seaquest-ram-v5",        # Exploration/action
    "ALE/Tutankham-ram-v5",       # Puzzle/exploration
    "ALE/Pitfall-ram-v5",         # Adventure/platformer
    "ALE/Boxing-ram-v5",          # Sports
    "ALE/IceHockey-ram-v5",       # Sports
    "ALE/Tennis-ram-v5",          # Sports
    "ALE/Skiing-ram-v5",          # Sports
    "ALE/WizardOfWor-ram-v5",     # Maze/combat
    "ALE/VideoPinball-ram-v5",    # Reflex/arcade
    "ALE/MontezumaRevenge-ram-v5",# Puzzle/exploration
]


def test_env(env_id, task_id_to_model_shape):
# Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # Model
    # model = MultiEnvActorCriticMLP(task_id_to_model_shape, actor_hidden_sizes=[1024, 1024], critic_hidden_sizes=[1024, 1024])
    num_inputs, num_actions = map(int, task_id_to_model_shape[env_id])
    model = ActorCriticMLP(num_inputs=num_inputs, num_actions=num_actions, actor_hidden_sizes=[64, 64], critic_hidden_sizes=[64, 64])

    # env_ids = list(task_id_to_model_shape.keys())
    # random_env_ids = random.sample(env_ids, 20)

    # environments_in_exps = []
    # for i in range(len(random_env_ids) // 2): 
    #     environments_in_exps.append([random_env_ids[2 * i], random_env_ids[2 * i + 1]])
    env_ids = [env_id]
    environments_in_exps = [[env_id]]

    n_experiences = len(environments_in_exps)
    scenario = gym_benchmark_generator(env_ids, n_experiences=n_experiences, n_parallel_envs=1, environments_in_exps=environments_in_exps)

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Reinforcement Learning strategy
    strategy = A2CStrategy(model, optimizer, per_experience_steps=1, max_steps_per_rollout=10, 
        device=device, eval_every=-1, eval_episodes=1)

    # train and test loop
    results = []
    for experience in scenario.train_stream:
        strategy.train(experience)
        results.append(strategy.eval(scenario.eval_stream))

    # print(f"results: {results}")

# Open and read the CSV file
with open(file_path, mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    
    # Skip the header row if the CSV has headers
    headers = next(csv_reader, None)
    
    valid_envs = []
    with open('data/rl_env_mapping.json', 'r') as f: 
        task_id_to_model_shape = json.load(f)
    
    # Iterate over rows
    for row in csv_reader:
        # Skip rows with missing entries
        print(f"row: {row}")
        if any(value == '' or value is None for value in row):
            continue
        
        env_id = row[0]
        if env_id not in selected_environments: 
            continue

        try: 
            test_env(env_id, task_id_to_model_shape)
            valid_envs.append(env_id)
        except Exception as e:
            print(f"env_id: {env_id}, exception: {e}")

    print(f"len(selected_environments): {len(selected_environments)}, len(valid_envs): {len(valid_envs)}\nvalid_envs: {valid_envs}")