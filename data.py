import pandas as pd
import numpy as np
import random

def get_customer(id, num_types):
    types = np.random.randint(1, num_types+1)
    return id, types

def get_sales_associate(id, num_products, num_types, num_leads = 5):
    performance = np.random.uniform (0,1, num_products)*100
    tenure = np.random.randint(15)
    mean_performance = np.mean (performance)
    types = np.random.randint(2,size = num_types) if num_types> 1 else np.ones(num_types)
    if num_leads>1:
        max_leads = np.random.randint(2, num_leads+1)
    else:
        max_leads = num_leads
    return id, tenure, performance, mean_performance, types, max_leads


def get_recommendation(num_recommendations, total_products):
    products = np.random.choice(list(range (1, total_products+1)), size=min(num_recommendations, total_products), replace=False)
    propensity = np.random.uniform (0.4,1, size = num_recommendations)*100
    return products, propensity

def get_customers (num_customers, num_types):
    customers_id = []
    customers_type = []
    for i in range(1,num_customers+1):
        customer = get_customer(i,num_types) 
        customers_id.append(customer[0])
        customers_type.append(customer[1]) 
    customers = pd.DataFrame({'id':customers_id, 'type': customers_type})
    return customers

def get_sales_associates(num_sales_associates, max_leads, num_type, num_products):
    sales_associates_id = []
    sales_associates_tenure = []
    sales_associates_performance = []
    sales_associates_mean_performance = []
    sales_associates_types = []
    sales_associates_max_leads = []
    for i in range(1, num_sales_associates+1):
        sales_associate = get_sales_associate(i, num_products, num_type, num_leads = max_leads)
        sales_associates_id.append(sales_associate[0])
        sales_associates_tenure.append(sales_associate[1])
        sales_associates_performance.append(sales_associate[2])
        sales_associates_mean_performance.append(sales_associate[3])
        sales_associates_types.append(sales_associate [4])
        sales_associates_max_leads.append(sales_associate[5])
    sales_associates_performance = np.array(sales_associates_performance)
    sales_associates_types = np.array(sales_associates_types)
    performances = {f'prod_{i+1}': sales_associates_performance[:,i] for i in range(sales_associates_performance.shape[1])}
    types = {f'type_{i+1}': sales_associates_types[:,i] for i in range(sales_associates_types.shape[1])}
    sales_associates = pd.DataFrame({'id':sales_associates_id, 'tenure': sales_associates_tenure, **performances, **types,'mean_performance':sales_associates_mean_performance,'max_leads':sales_associates_max_leads})
    return sales_associates

def get_rows(customers, num_products, num_recommendations):
    recommendation_dict = {'id': [], 'type': [], 'product': [], 'propensity':[]}
    for i in range(customers.shape[0]):
        n_recommendations = np.random.random.randint(1, num_recommendations+1)
        products, propensities = get_recommendation(n_recommendations, num_products)
    for j in range(min(n_recommendations, num_products)):
        recommendation_dict['id'].append(customers.iloc[i]['id'])
        recommendation_dict['type'].append(customers.iloc[i]['type'])
        recommendation_dict['product'].append(products[j])
        recommendation_dict['propensity'].append(propensities[j])
    recommendations = pd.DataFrame (recommendation_dict)
    return recommendations

def get_episode(num_customers, num_sales_associates, num_products, num_recommendations, max_leads, num_types): 
    customers = get_customers(num_customers, num_types)
    sales_associates = get_sales_associates (num_sales_associates, max_leads, num_types, num_products)
    recommendations = get_rows(customers, num_products, num_recommendations)
    return customers, sales_associates, recommendations