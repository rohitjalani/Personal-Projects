from constants import gamma, lambda_gae

def compute_gae (rewards, masks, values, next_value): 
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed (range (len (rewards))):
        delta = rewards[step] + gamma*values[step + 1]*masks[step] - values [step]
        gae = delta + gamma*lambda_gae*masks[step]*gae
        returns.insert(0, gae + values[step])
    return returns

def reward_func(state, action): 
    reward = state[0][state[2],-1]*state[1][action, 0]        
    return reward