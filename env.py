import numpy as np
import pandas as pd 
from data import get_episode
from utils import reward_func

class CustomEnv():
  def __init__(self, num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types):
    episode = get_episode(num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types)
    associates = episode[1]
    leads = episode[2]
    associates = (associates.drop(columns = ['id','tenure','prod_1','type_1', 'max_leads']).to_numpy()/100).astype(np.float32)
    leads = (leads.drop(columns = ['id','type','product']).to_numpy()/100).astype(np.float32)

    self.num_customers = num_customers
    self.num_sales_associates = num_sales_associates
    self.num_products = num_products
    self.num_recommendations = num_recommendations
    self.max_leads = max_leads
    self.num_types = num_types

    self.leads = leads
    self.associate = associates
    self.associate_cap = associates[:,-1]
    self.time_step = 0
    self.last_action = -1
    self.steps = leads.shape[0]
    self.done = False

  def step(self, action):

    reward = reward_func((self.leads, self.associate, self.time_step), action)
    # reward = 0
    # reward+= self.leads[self.time_step,-1] * self.associate[action,0]

    # self.associate_cap[action]-=1
    # reward+= min(0, self.associate_cap[action])*10

    self.time_step+=1
    self.last_action = action

    if self.time_step==self.steps:
      self.done = True

    return self.leads, self.associate, self.time_step, self.last_action, reward, action, self.done

  def state(self):
    return self.leads, self.associate, self.time_step, self.last_action, self.done

  def reset(self):
    episode = get_episode(self.num_customers, self.num_sales_associates, self.num_products, self.num_recommendations, self.max_leads, self.num_types)
    associates = episode[1]
    leads = episode[2]
    associates = (associates.drop(columns = ['id','tenure','prod_1','type_1', 'max_leads']).to_numpy()/100).astype(np.float32)
    leads = (leads.drop(columns = ['id','type','product']).to_numpy()/100).astype(np.float32)

    self.leads = leads
    self.associate = associates
    self.associate_cap = associates[:,-1]
    self.time_step = 0
    self.last_action = -1
    self.steps = leads.shape[0]
    self.done = False

    return self.leads, self.associate, self.time_step, self.last_action, self.done