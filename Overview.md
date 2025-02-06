## BANDIT SCHEDULING for FL(BSFL)

This novel approach for the client selection problem is based on the training time of each client, but also on the possibility of generalising the selection policy. To achieve this, the Agent in each iteration selects the appropriate subset based on an Upper Confidence Bounce(UCB) plus a specific generalization function. The UCB is based on background knowledge for the mean-time of each client that comes from iterations (explore) of the algorithm. The generalization function is based on how many times a client has been selected at a specific iteration t. For performance monitoring, the Agent's selection at each iteration is compared to a Genie that can make the decision with the maximum reward at each iteration, because is given prior knowledge of the mean-time of the clients. This algorithm shows way better results(almost logarithmic) in the regret compared to previous approaches(e.g Random Selection).

### Code Structure

The program is divided in five source files.
- **gradient**: 
    - It has all the necessary code to train a simple linear regression model. There is a function that creates synthetic data with a linear relation and the Gradient Descent function with the derivatives implementation. 
    - With the help of numpy.random.normal the connection status of the user, but also the available computational resources are stochastic to make the scenario more realistic.
- **functions**:
    - It contains the functions that both th Genie and the Agent will use to initialize their clients' list, the counter update and the contribution update.
    - The list of dictionaries is the knowledge that Genie and the Agent has for each client. The counter is set to 1 because if set to zero then an error for dividing number with zero will occur in Agent's upconb method. The contribution(g_k) is initialized based on the formula of the client_gen method when the appropriate initial values for the argument variables are used. The UCB is set to infinitive to represent the uncertainty of the Agent before the observation of the clients time.
    - The client_gen method is the one that is responsible for the contribution of each client in the training process. The counter is always updated before the client contribution.
- **results**:
    - It contains the the plotting functions to visualize the results of the program. 
    - The regret between Genie and Agent is plotted.
    - The regression line is first plotted with the initial random slope and intercept and then with the updated(trained) values. Both lines are compared to the one that the data where generated based on.
- **components**: 
    - Here are the basic components that the program needs.
    - The enviroment with the assignment of creating the clients and distribute to them their data for the training of the linear regression model. np.random.noraml is used  to represent the different machines/nodes(hence processing power) and connection each client has.
    -  The Genie with the function that calculates the mean-time of each client by invoking the gradient descent method, so it has the prior knowledge compared to the Agent.
    - The Agent with the mean-time and ucb update methods that invokes during the training process to update his knowledge.
- **main**: 
    - This is where all the above source files are used for the FL training process.
    - *(10-26)* Necessary initializations for the enviroment, the synthetic data for the LR model and the global model values.
    - *(29-32)* The enviroment(clients) are initialized and both the Agent and the Genie initialize their clients' knowledge lists. Genie observes clients mean-times and has its prior knowledge.
    - *(34-39)* The lists that will contain the reward and the regret for are empty-initialized and the subsets of clients are created. 
    - *(41-64)* Lists to hold the rewards of Genie and Agent. Iterating through each subset observing speeds(ucb for the agent) and contributions of each client, choosing the best subset to train based on the selection policy algorithm. Many times there are subsets with the same reward so the tie is broken arbitrary using random.choice().
    - *(67-90)* The actual training of the chosen subset of clients. Lists to hold the needed data of each client(time,updates and contribution). The counter and the mean time of each client is updated after the training process. The global parameters are updated based on the aggregated values of each client. Agent's reward and the regret is calculated.
    - *(92-98)* The counter of the chosen Genie clients is updated, as well as the UCB of ther Agent clients and the contribution of both Genie and Agent clients.
    - The last lines of code is using the result methods to visualize the model with the initial parameters/weights, the train weights and the regret of the Agent's Selection Policy.


