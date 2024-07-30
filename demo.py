from train import train_loop
from model import ActorCriticModel
import torch
import torch.nn as nn
import torch.optim as optim
from constants import learning_rate

num_workers = 25
num_tasks = 25
task_feature_dim = 1
worker_feature_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = ActorCriticModel (num_tasks, num_workers, task_feature_dim, worker_feature_dim)
optimizer = optim. Adam (policy_net.parameters (), lr=learning_rate)
policy_net.to(device)

train_loop(policy_net, optimizer)