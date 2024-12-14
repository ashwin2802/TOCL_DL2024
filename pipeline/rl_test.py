import torch
from torch.optim import Adam
from avalanche_rl.benchmarks.rl_benchmark_generators import gym_benchmark_generator

from avalanche_rl.models.actor_critic import ActorCriticMLP, MultiEnvActorCriticMLP
from avalanche_rl.training.strategies.actor_critic import A2CStrategy

import json
import random

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

with open('data/rl_env_mapping.json', 'r') as f: 
    task_id_to_model_shape = json.load(f)

# Model
model = MultiEnvActorCriticMLP(task_id_to_model_shape, actor_hidden_sizes=[1024, 1024], critic_hidden_sizes=[1024, 1024])

env_ids = list(task_id_to_model_shape.keys())
random_env_ids = random.sample(env_ids, 20)

environments_in_exps = []
for i in range(len(random_env_ids) // 2): 
    environments_in_exps.append([random_env_ids[2 * i], random_env_ids[2 * i + 1]])

n_experiences = len(environments_in_exps)
scenario = gym_benchmark_generator(random_env_ids, n_experiences=n_experiences, n_parallel_envs=1, environments_in_exps=environments_in_exps)

# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=1, max_steps_per_rollout=5, 
    device=device, eval_every=1, eval_episodes=10)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.eval_stream))

print(f"results: {results}")