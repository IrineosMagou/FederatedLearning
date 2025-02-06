from gradient import LinearReGradient as lr
from typing import Callable, List, Dict
import numpy as np

class Enviroment:
  def __init__(self , num_of_clients: int , r0_training: list, r1_training: list):
   self.client_lst= self.clients_init(num_of_clients ,r0_training , r1_training)
#Data for distribution
  def clients_init(self, num_of_clients: int , r0: list , r1: list)-> List[Dict[str, float | int]]:
    client_lst = []
    for i in range(num_of_clients):
      j = i*20
      env_clients_dict = {
          "id" : i ,
          "comp_resources" : np.random.randint(2 , 5), #to display processing power the of each node(client)
          "connection_status": np.random.randint(1, 6),
          "x_points" : r0[j:j+20] ,
          "y_points" : r1[j:j+20]
      }
      client_lst.append(env_clients_dict)
    return client_lst

#==========================================================================------GENIE-----==========================================================================================
class Genie:
  def __init__(self, num_of_clients: int, client_initializer: Callable[[int], List[Dict[str , float | int]]] ):
    self.clients_init= client_initializer
    self.clients = self.clients_init(num_of_clients)
#learnMeanTimes
  def get_means(self, clients_lst: list, t_min: int ):
    for i, client in enumerate(clients_lst):
      mean= []
      for _ in range(5):
        res = lr.gradient_descent(0.1, 0.1, client["x_points"], client["y_points"], 0.0001, clients_lst, i)
        mean.append(res.time)
      self.clients[i]["mean_time"] = sum(t_min / mean[i] for i in range(5)) / 5

#==========================================================================------SERVER------==========================================================================================
class Agent:
  def __init__(self, num_of_clients: int, client_initializer: Callable[[int], List[Dict[str , float | int]]] ):
    self.clients_init = client_initializer
    self.clients = self.clients_init(num_of_clients)

#MeanTime
  def mean_update(self, client, iter_time, t_min: int):
    c = self.clients[client]["counter"]
    self.clients[client]["mean_time"] = (self.clients[client]["mean_time"]*(c-1) + (t_min / iter_time )) / c

#UpperConf
  def upconb(self, iteration: int, client:int, avlb_channels: int):
    c = self.clients[client]["counter"]
    var = (avlb_channels + 1)*np.log(iteration)
    self.clients[client]["ucb"] = self.clients[client]["mean_time"] + ((var / c)**0.5)