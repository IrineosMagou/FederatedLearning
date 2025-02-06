#=======================================================================-------------FUNCTIONS--------------========================================================================
from typing import List , Dict
import numpy as np

class ClientHelp:
#Initialize-Clients-Knowledge
    @classmethod
    def init_clients(cls, num_of_clients: int)-> List[Dict[str, float | int]]:#Both Agent and Genie will use this function
      clients = []
      for i in range(num_of_clients):
        client_dict = {
            "client_id" : i ,
            "mean_time" : 0 ,
            "counter"   : 1 ,
            "g_k"       : (-0.31640625),
            "ucb"       : float('inf'),
            "loss"      : 0
        }
        clients.append(client_dict)
      return clients

#CounterUpdate
    @classmethod
    def counter_update(cls, client, lst_clients): #list argument cause both agent and genie will use this function
      lst_clients[client]["counter"] += 1

#Generalization Function for the contribution of the clients in the learning progress
    @classmethod
    def client_gen(cls, client: int, iteration: int, lst_clients: list, desired_ratio: float, beta: float)-> float:#list argument cause both agent and genie will use this function
      clt_ratio =  lst_clients[client]["counter"] / iteration
      n = desired_ratio - clt_ratio
      g_k = ((abs(n))**beta)*np.sign(n)
      return g_k