# ***Collaboration and Competition*** 
***Udacity Deep Reinforcement Learning***

***Capstone 3: Collaboration and Competition***

***Swastik Nath***

---

### Words About the Environment:

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Algorithmic Approach:

To solve the environment we use an Actor-Critic architecture with DDPG Algorithm. We use a total of 4 Deep Neural Networks:

***Actor Networks***: We use a total of 2 Actor Networks which works by estimating Policy Gradients.   

 - Actor Network (Online) : Interacts with the Environment Real-time. It uses two Linear Layers with Batch Normalization in between them. It uses 128 neurons in the hidden layers. Between the linear layers we use the ***Relu*** activation. We use ***Tanh*** as the final activation layer. We feed the state from the environment to the Actor online model. 
 ```
 Actor(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
 ```

 - Actor Network (Target): The online model weights are copied to this model at a few timesteps. It uses two Linear Layers with Batch Normalization in between them. It uses 128 neurons in the hidden layers. Between the linear layers we use the ***Relu*** activation. The structure of the target network is totally similar to that of the online network.


***Critic Network***: We use 2 another Deep Neural Networks to estimate and evaluate the Action values for state, action pair. 

 - Critic Network (Online): The weights from the Actor Online Model is copied here after a few specified timesteps. We use a 128 neurons between the first layer and second layer, after which we add the size of actions and the number of hidden layers to the second linear layer. We use the output of the last linear layer as the output of the network.  
 ```
 Critic(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (fc2): Linear(in_features=130, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
 ```
 - Critic Network (Target):  The weights from the Actor Target Model is copied here after a few specified timesteps. We use a 128 neurons between the first layer and second layer, after which we add the size of actions and the number of hidden layers to the second linear layer. We use the output of the last linear layer as the output of the network. The structure of the target network is totally similar to that of the online network.

 
It is not possible to straightforwardly apply Q-learning to continuous action spaces, because in continuous spaces finding the greedy policy requires an optimization of the action at every timestep; this optimization is too slow to be practical with large, unconstrained function approximators and nontrivial
action spaces. Instead, here we used an actor-critic approach based on the DPG algorithm. 

The DPG algorithm maintains a parameterized actor function µ(s|θ
µ) which specifies the current
policy by deterministically mapping states to a specific action. The critic Q(s, a) is learned using
the Bellman equation as in Q-learning.

A major challenge of learning in continuous action spaces is exploration. An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently
from the learning algorithm. We constructed an exploration policy µ
0 by adding noise sampled from a noise process N to our actor policy
µ0(st) = µ(st|θ;µ;t) + N (7)

N can be chosen to suit the environment. We use the ***Ornstein-Uhlenbeck process*** (Uhlenbeck & Ornstein, 1930) to generate the noises and induce them to the states for further exploratory analysis and randomness. 


### Results:

 - ### Multi Agent Competitive Environment:  
   
   The DDPG Implementation trains multiple agents to interact with the environment. It was able to solve the environemt in **1439** episodes. It gave average of the scores as below:
   
   
  
   To train 3000 episodes it took around 3 hour and 47 minutes on CPU.

   ![multiple_agent](https://github.com/swastiknath/rl_ud_3/raw/master/episode_graph.png)


### Hyperparameters:
We use the following set of hyperparameters with the multi agent environment. 

| Hyperparameter | Multi Agent Environment |
|----------------|-------------------------|
| Buffer Memory Size |      1e-5       |
| Batch Size for Experience Replay    |    128      |    
| Discount Factor (Gamma)             |        0.99       |
| Interpolation Factor (TAU)          |        1e-3       |
| Learning Rate for Actor (LR) Adam Optimizer      |   2e-4     |
| Learning Rate for Critic (LR) Adam Optimizer    |    2e-4    |
| Weight Decay                |         0             |
| MU (Mean for Orstein Uhlenbeck Noise)   |    0       |
|Sigma (Standard Deviation for Orstein Uhlenbeck Noise)|    0.1            |
|Theta for Orstein Uhlenbeck Noise) |     0.15      |
| 1st Actor Linear Layer Hidden Size | 128 |
| 1st Critic Linear Layer Hidden Size | 128 |
| 2nd Actor Linear Layer Hidden Size |  128 |
| 2nd Critic Linear Layer Hidden Size | 128+2(Actions) = 130 |

### Future Ideas :


The performance of the agents might be improved by considering the following:

- Consider procedures to reduce the episode-to-episode variation
  
  I tried decaying the Ornstein-Uhlenbeck noise exponentially over the episodes (with rate 0.998), but this did not seem to help with reducing the variation (the maximum score and time to solve the environment also remained roughly the same). I could try decaying the noise more rapidly to see if this reduces the variation (at the expense of exploration).

- Hyperparameter optimisation 

  Many of the hyperparameters, such as the network architectures (number of layers and neurons), learning rates, batch and buffer sizes, and the level of noise, could be further tuned to improve performance.

- Alternative learning algorithms

  Alternative methods include PPO with team spirit and the multi-agent DDPG algorithm described [here](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) that uses centralised training and decentralised execution. 


