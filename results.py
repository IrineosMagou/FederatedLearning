#===================================================================------RESULTS------===================================================================================
import matplotlib.pyplot as plt
from typing import Callable
import numpy as np

class Results:
    @classmethod
    def calcRes(cls, env: object , loss: float, m, b, num_of_clients: int, x_test: list , y_test: list, loss_func: Callable[[float, float, list, list], float]):
        for i in (env.client_lst):
             loss.append(loss_func(m , b , i["x_points"] , i["y_points"] ))
        loss_cum = sum(loss)/num_of_clients
        loss_test = loss_func(m , b , x_test , y_test)
        print(f"loss : {loss_cum}")
        print(f"Test Loss : {loss_test}")

    @classmethod
    def plotReg(cls, regret_to_plt: list):
        x = np.arange(len(regret_to_plt))
# Apply a moving average with a window size (adjust as needed)
        window_size = 15
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(regret_to_plt, weights, mode='valid')
        plt.plot(x[window_size - 1:], smoothed, "b-")
        plt.xlabel("Iterations")
        plt.ylabel("Regret")
        plt.ylim(0, 1.2)
        plt.grid(True)
        plt.legend()
        plt.show()

    @classmethod
    def plotLine(cls, slope: float, intercept: float, x_points: list, y_points: list):
      print(f"Slope {slope} and Intercept {intercept}")
      X= np.array(x_points)
      # Y= np.array(y_points)
      plt.scatter(x_points , y_points)
      plt.plot(X, slope*X + intercept, "r")
      plt.plot(X, 3+4*X, color = "b") #original
      plt.show()

    @classmethod
    def clienstCt(cls, clients: list):
      for client in(clients):
        print(f"Client_id: {client['client_id']} | Counter: {client['counter']} | Gen: {client['g_k']}")
