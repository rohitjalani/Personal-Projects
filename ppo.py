from torch.distributions import Categorical
import torch
import torch.nn as nn
from constants import ppo_epochs, mini_batch_size, value_loss_coef, entropy_coef, clip_epsilon, max_grad_norm

def ppo_update(policy_net, optimizer, states1, states2, actions, times, last_actions, log_probs, returns, advantages, device):
    dataset = torch.utils.data.TensorDataset(states1, states2, actions,times, last_actions, log_probs, returns, advantages)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True)

    for _ in range(ppo_epochs):
        for batch in dataloader:
            states1_batch, states2_batch, actions_batch, times_batch, last_actions_batch, old_log_probs_batch, returns_batch, advantages_batch = batch

            # Normalize advantages
            # advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            logits, state_values = policy_net(states1_batch.to(device), states2_batch.to(device), times_batch.to(device), last_actions_batch.to(device))

            # print("states1_batch", states1_batch)
            # print(states2_batch)
            # print(times_batch)
            # print(last_actions_batch)
            # print(logits)
            # print()

            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions_batch.to(device))

            ratio = torch.exp(new_log_probs - old_log_probs_batch.to(device))
            surr1 = ratio * advantages_batch.to(device)
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_batch.to(device)

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns_batch.to(device) - state_values.squeeze()) ** 2).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
            optimizer.step()