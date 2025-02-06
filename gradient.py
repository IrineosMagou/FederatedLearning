import time
import numpy as np

class LinearReGradient:
    
    @classmethod
    def gradient_descent(cls, cur_m: float, cur_b: float, x_points, y_points, L, lst_clients, id: int):
      n = len(x_points)
      start = time.time()
      X = np.array(x_points)
      Y = np.array(y_points)
      for _ in range(n):
        y_predicted= (cur_m*X) + cur_b
        m_derivative = -(2/n) * sum(X*(Y-y_predicted))
        b_derivative = -(2/n) * sum(Y-y_predicted)
      m = cur_m - (L*m_derivative)
      b = cur_b - (L*b_derivative)
      end = time.time()
      noise = np.random.normal(3 , 0.6) # to represent the connection at each iteration.
      comp_resources = np.random.normal(2,0.6) #to display the available computational resources of each node(client)
      iteration_time =(end - start + noise + comp_resources)
      return m , b , (iteration_time + lst_clients[id]["comp_resources"])

    @classmethod
#Creating the Data with the linear relation y= 2a+ 1 so the model can generalize
    def createData(cls, x: list, distMean: float, distStd: float):
      y = 3 + 4*x + np.random.normal(distMean , distStd , x.shape[0])
      return y