# Client Selection for Generalization in Federated Learning

### What is Federated/Collaborative Learning ?
The "traditional" way of training ML models is centralized, meaning gathering data from users to a central server. This approach raised concerns about the confidentiality of the users' data. In 2016, Google presents the concept of Federated Learning for training ML models. The idea is to train a model across multiple decentralized edge devices or servers holding local data samples, ensuring that the data never leave the device. 
Federated Learning operates by conducting model training iterations locally on the user's device, while only the updates to the model —rather than the data itself— are transmitted and aggregated on a central server. This process allows for collaborative learning without centralized data collection, thereby enhancing privacy and maintaining data locality. In FL, selected clients train their local models and send a function of the models to the server, which consumes a random processing and transmission time. The server updates the global model and broadcasts it back to the clients.

### The Client Selection Problem
Most of the times the amount of participants is huge and realistically there are not enough available transmission channels for every client, so subsets(using combinations) are created. At each iteration in the training processs a subset is chosen. One iteration ends when the server receives the updates from all the participants in the specific iteration. Each node (client) participating in the learning process has different data quantity and quality, available computational resources, transmission rate, and other factors that affect the training process and thus the efficiency of the system. The client selection problem in FL is to schedule a subset of the clients for training and transmission at each given time so as to optimize the learning performance(both time and generalization).

### The Proposed Solution
- The program aims to solve the aforementioned Client Selection Problem in Federated Learning. The approach involves modeling and analyzing the problem as a MAB(Multi-Armed-Bandit) problem by treating the iteration latency or the local loss function of each client as a reward taken from an unknown distribution and given to the server.
- The approach of treating clients’ latencies as unknown random variables drawn from unknown distributions, leads to a stochastic optimization problem that raises the well-known exploration versus exploitation dilemma.
- The server choses the subset of clients at each iteration based on the training mean-time plus a generalization function.

### Multi-Armed-Bandit Problems
These problems are about an agent repeatedly facing the same fundamental problem, but with evolving knowledge or understanding of the situation. At its heart, a MAB problem presents the agent with a set of options (the "arms" of the bandit, like slot machines). Each option has an unknown reward distribution. The agent's goal is to maximize its cumulative reward over time. The key is that the agent doesn't know the reward distributions upfront. It has to learn them through trial and error. Each time the agent chooses an arm, it gets a reward, which provides information about that arm's reward distribution. This updates the agent's knowledge. This leads to the classic exploration-exploitation dilemma:
- Exploration: The agent needs to try different arms to gather information and improve its understanding of the reward distributions.
- Exploitation: The agent wants to choose the arm that it currently believes will give the  highest reward, based on its current knowledge.

So, the agent is always trying to solve the same problem: "Which arm should I choose to maximize my reward?" However, its knowledge about the arms changes with each pull. This means the optimal action might change over time as the agent learns more.
 
### Credits
This code is based on the article: [Client Selection for Generalization in
Accelerated Federated Learning:
A Multi-Armed Bandit Approach](https://arxiv.org/pdf/2303.10373v1) by Dan Ben Ami, Kobi Cohen, Qing Zhao.

 
 




