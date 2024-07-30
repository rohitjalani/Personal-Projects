learning_rate = 1e-4
gamma = 1 #discount factor for future rewards
lambda_gae = 1 #Smoothing parameter that balances bias and variance in the advantage estimates
clip_epsilon = 0.2 #ppo clip factor
entropy_coef = 0.01 #proportion of enttropy loss in ppo loss
value_loss_coef = 0.1 #proportion of value loss in ppo loss
max_grad_norm = 0.1
ppo_epochs = 4 #epochs for a batch of data
mini_batch_size = 4 #batch size
total_timesteps = 500 #number of episodes to consider
print_interval = 10
num_customers = 25
num_sales_associates = 25
num_products = 1
num_recommendations = 2
max_leads = 2
num_types = 1