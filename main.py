from components import Enviroment, Agent, Genie
from gradient import LinearReGradient as lr
from functions import ClientHelp as ch
from results import Results as res
import itertools as itr#to use the combination() to create subsets
import random as rnd
import numpy as np

#GLOBAL_PARAMS
num_of_clients= 20
avlb_channels= 5
t_min= 3 #The time to send receive the model as if client doesnt require time to train
desired_ratio= avlb_channels / num_of_clients #used in generalization function
beta= 4 #tuning parameter for generaliztaion
alpha= 0.001#hyper-parameter for balance mean-time and contribution

#Synthetic data generation
distMean=0
distStd=10
np.random.seed(1234)
x_train = np.random.randint(0, 51, size=400)
y_train= lr.createData(x_train,distMean,distStd)
#slope and intercept are arbitraty
slope = 0.1
intercept = 0.1
L = 0.0001 #learning rate

#========================================================================-----FL-ITERATION-----===========================================================================================
env = Enviroment(num_of_clients, x_train, y_train)
server = Agent(num_of_clients, ch.init_clients)
genie = Genie(num_of_clients, ch.init_clients)
genie.get_means(env.client_lst, t_min) #Provide Genie with prior knowledge

reward_agent = [] #where agent's reward is stored and sum at each iteration
reward_g = [] #where genie's reward is stored and sum at each iteration
regret_to_plt = [] #regret at each iteration

server_subsets= list(itr.combinations(server.clients , avlb_channels))#Create subsets.mask to list so i can iterate through it
genie_subsets= list(itr.combinations(genie.clients , avlb_channels))

for iter in range(1,501):
  genie_rew= []#Genie's subsets rewards
  to_select= []#Agent's selection based on ucb
  for subset , subsetg in zip(server_subsets, genie_subsets):
    server_contribution = 0
    genie_contribution = 0
    ucb_val = [] #list to append ucb's of subsets to find min
    genie_mean = []
# Iterating through subsets, observing speeds and contributions
    for server_clt_dict, genie_clt_dict in zip(subset, subsetg):
      ucb_val.append(server_clt_dict["ucb"])
      genie_mean.append(genie_clt_dict["mean_time"])
      server_contribution += server_clt_dict["g_k"]
      genie_contribution += genie_clt_dict["g_k"]
    to_select.append(min(ucb_val) + alpha/avlb_channels*server_contribution) #Selection Policy Algorithm
    genie_rew.append(min(genie_mean) + alpha/avlb_channels*genie_contribution)#Genie's Subsets Rewards

  subset_to_train = max(to_select)#Get the pointer from the list , hence the subset for train
  indices_of_max_values = [index for index, value in enumerate(to_select) if value == subset_to_train]
  arbirtrary_choice = rnd.choice(indices_of_max_values)
  to_regret_g = max(genie_rew)
  genie_max_values = [index for index , value in enumerate(genie_rew) if value == to_regret_g]#Get the pointer from the list , hence the subset.
  arbitrary_choice_g = rnd.choice(genie_max_values)
  reward_g.append(genie_rew[arbirtrary_choice])

#Chosen subset training
  train_time = []#For each client of the subset
  clt_slope = []#For the slope of each client's training update
  clt_intercept = []#For the intercept of each client's training update
  contribution_r = 0
  for i in server_subsets[arbirtrary_choice]:
    x_client = env.client_lst[i["client_id"]]
    contribution_r += i["g_k"]
    m , b , client_train = lr.gradient_descent(slope, intercept, x_client["x_points"], x_client["y_points"], L, env.client_lst, i["client_id"])
    clt_slope.append(m)
    clt_intercept.append(b)
    train_time.append(t_min/client_train)
    ch.counter_update(i["client_id"], server.clients)
    server.mean_update(i["client_id"], client_train, t_min)

#Aggregating and updating global model values
  slope = sum(clt_slope)/avlb_channels
  intercept = sum(clt_intercept)/avlb_channels

#Calculating the reward for the agent, the genie and also the regret.
  reward_agent.append((min(train_time) + (alpha/avlb_channels)*contribution_r))
  agent_sum = sum(reward_agent)
  genie_sum = sum(reward_g)
  regret = genie_sum - agent_sum
  regret_to_plt.append(regret)

#Updating the counter of chosen clients, hence their contribution and the confidence of the agent
  for g in genie_subsets[arbitrary_choice_g]:
   ch.counter_update(g["client_id"] , genie.clients) #Genie
  for server_clt_dict , genie_clt_dict in  zip(server.clients , genie.clients):
    server_clt_dict["g_k"] = ch.client_gen(server_clt_dict["client_id"], iter + 1, server.clients, desired_ratio, beta)
    server.upconb(iter, server_clt_dict["client_id"], avlb_channels)
    genie_clt_dict["g_k"] = ch.client_gen(genie_clt_dict["client_id"], iter + 1, genie.clients, desired_ratio, beta)#Genie

#Generating test data for the regression model
np.random.seed(1234)
x_test = np.random.randint(0, 51, size=50)
y_test= lr.createData(x_test, distMean, distStd)

res.clienstCt(server.clients)
res.plotReg(regret_to_plt)
res.plotLine(0.1, 0.1 , x_test , y_test)
res.plotLine(slope, intercept, x_test, y_test)
