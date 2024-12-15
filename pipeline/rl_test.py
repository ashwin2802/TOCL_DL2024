import torch
from torch.optim import Adam
from avalanche_rl.benchmarks.rl_benchmark_generators import gym_benchmark_generator

from avalanche_rl.models.actor_critic import ActorCriticMLP, MultiEnvActorCriticMLP
from avalanche_rl.training.strategies.actor_critic import A2CStrategy

import json

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

with open('data/rl_env_mapping.json', 'r') as f: 
    task_id_to_model_shape = json.load(f)

with open('data/selected_environments.json', 'r') as f: 
    selected_envs = json.load(f)

# Model
model = MultiEnvActorCriticMLP(task_id_to_model_shape, actor_hidden_sizes=[1024, 1024], critic_hidden_sizes=[1024, 1024])
# num_inputs, num_actions = map(int, task_id_to_model_shape["ALE/Tetris-ram-v5"])
# model = ActorCriticMLP(num_inputs=num_inputs, num_actions=num_actions, actor_hidden_sizes=[1024, 1024], critic_hidden_sizes=[1024, 1024])

environments_in_exps = []
for i in range(len(selected_envs) // 2): 
    environments_in_exps.append([selected_envs[2 * i], selected_envs[2 * i + 1]])

# env_ids = ["ALE/Tetris-ram-v5", "CartPole-v1"]
# environments_in_exps = [["ALE/Tetris-ram-v5"], ["CartPole-v1"]]

n_experiences = len(environments_in_exps)
scenario = gym_benchmark_generator(selected_envs, n_experiences=n_experiences, n_parallel_envs=1, environments_in_exps=environments_in_exps)

# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=10, max_steps_per_rollout=5, 
    device=device, eval_every=-1, eval_episodes=1)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.eval_stream))

print(f"results: {results}")