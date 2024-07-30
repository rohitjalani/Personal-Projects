from constants import total_timesteps, print_interval
import torch
from torch.distributions import Categorical
from utils import compute_gae
from env import CustomEnv
import numpy as np
from ppo import ppo_update
from constants import num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types


def train_loop(policy_net, optimizer, device):
  env = CustomEnv(num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types)
  state1, state2, time, last_action, done = env.state()
  episode_reward = 0
  episode_rewards = []
  max_possible_reward = []
  for timestep in range(int(total_timesteps)):
      states1 = []
      states2 = []
      actions = []
      times = []
      last_actions = []

      log_probs = []
      rewards = []
      values = []
      masks = []
      with torch.no_grad():
        for _ in range(25):
            state_tensor1 = torch.FloatTensor(state1).unsqueeze(0).to(device)
            state_tensor2 = torch.FloatTensor(state2).unsqueeze(0).to(device)
            time = torch.tensor(time, dtype = torch.float32).reshape(1,-1).to(device)
            last_action = torch.tensor(last_action, dtype = torch.float32).reshape(1,-1).to(device)

            logits, state_value = policy_net(state_tensor1, state_tensor2, time, last_action)

            # actor_output = F.softmax(actor_output, dim=1)

            dist = Categorical(logits=logits.cpu())
            action = dist.sample().cpu()

            next_state1, next_state2, next_time, next_last_action, reward,action_, done = env.step(action.item())
            log_prob = dist.log_prob(action)

            states1.append(state_tensor1.cpu())
            states2.append(state_tensor2.cpu())
            times.append(time.cpu())
            last_actions.append(last_action.cpu())

            actions.append(action.cpu())
            log_probs.append(log_prob.cpu())
            rewards.append(reward)
            values.append(state_value.cpu())
            masks.append(1 - done)

            state1, state2, time, last_action = next_state1, next_state2, next_time, next_last_action
            episode_reward += reward

            if done:
                del env
                env = CustomEnv(num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types)
                max_r = np.sum(state1*max(state2).item())
                max_possible_reward.append(max_r)
                state1, state2, time, last_action, done = env.state()
                episode_rewards.append(episode_reward)
                episode_reward = 0

        next_state_tensor1 = torch.FloatTensor(next_state1).unsqueeze(0).to(device)
        next_state_tensor2 = torch.FloatTensor(next_state2).unsqueeze(0).to(device)
        next_time_tensor = torch.tensor(next_time, dtype = torch.float32).reshape(1,-1).to(device)
        next_last_action_tensor = torch.tensor(next_last_action, dtype = torch.float32).reshape(1,-1).to(device)

        _, next_value = policy_net(next_state_tensor1, next_state_tensor2, next_time_tensor, next_last_action_tensor)

        returns = compute_gae(rewards, masks, values, next_value.cpu())

        states1 = torch.cat(states1)
        states2 = torch.cat(states2)
        times = torch.cat(times)
        last_actions = torch.cat(last_actions)

        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values).detach()
        advantages = returns - values

      ppo_update(policy_net, optimizer, states1, states2, actions, times, last_actions, log_probs, returns, advantages, device)

      if timestep % print_interval == 0:
          print(f"Timestep: {timestep}, Reward: {episode_rewards[-1]}")