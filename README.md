###Poker Reinforcement Learning Project
##Overview
This project implements various reinforcement learning (RL) algorithms to train agents to play a simplified version of poker. The goal is to explore and compare the performance of different RL techniques in optimizing poker strategies. The project includes implementations of Monte Carlo Tree Search (MCTS), SARSA, Expected SARSA, Q-Learning, and Neural Networks. Each algorithm is evaluated in a simulated poker environment, and their performance is visualized through graphs.

##Key Features
#Multiple RL Algorithms:

#Monte Carlo Tree Search (MCTS)
#SARSA
#Expected SARSA
#Q-Learning
#Neural Networks
#Custom Poker Environment:

A simplified poker environment with discrete state and action spaces.
Simulates poker rounds with rewards and termination conditions.
Performance Comparison:

Tools to compare the performance of algorithms using metrics like average rewards over episodes.
Graphs to visualize the learning progress and compare two algorithms at a time.
Training and Visualization:

Each algorithm can be trained individually.
Training progress and results are visualized using Matplotlib.

##Project Structure
Poker Reinforcement Learning/
├── compare_algorithms.py   # Script to evaluate and compare all algorithms
├── monte_carlo.py          # Monte Carlo Tree Search implementation
├── sarsa.py                # SARSA algorithm implementation
├── Qlearning.py            # Q-Learning algorithm implementation
├── expected_sarsa.py       # Expected SARSA algorithm implementation
├── neural_networks.py      # Neural Network-based RL agent
├── .gitignore              # Git ignore file

##Algorithms Implemented
#Monte Carlo Tree Search (MCTS):

Uses simulations to evaluate actions and select the best one based on visit counts and rewards.
Implements the Upper Confidence Bound for Trees (UCT) formula for action selection.
#SARSA:

On-policy temporal difference learning algorithm.
Updates Q-values based on the current policy and the next state-action pair.
#Expected SARSA:

Similar to SARSA but uses the expected value of the next state-action pair for updates.
Balances exploration and exploitation using an epsilon-greedy policy.
#Q-Learning:

Off-policy temporal difference learning algorithm.
Updates Q-values using the maximum future reward.
#Neural Networks:

Uses a neural network to approximate Q-values.
Trains the agent using discounted rewards and categorical cross-entropy loss.
##Poker Environment
The poker environment is a simplified simulation with the following features:

#State Space: Represents the current state of the game (e.g., hand strength, pot odds, position, opponent aggression).
#Action Space: Includes actions like fold, check, call, raise_small, and raise_large.
#Rewards: Rewards are assigned based on the action taken and the current state.
#Termination: Each round ends with a 10% probability or after a fixed number of steps.
##How to Use
#Clone the Repository:

git clone https://github.com/your-username/poker-reinforcement-learning.git
cd poker-reinforcement-learning

Install Dependencies: Install the required Python packages:

pip install numpy matplotlib tqdm tensorflow

Train and Compare Algorithms: Run the compare_algorithms.py script to train all algorithms and compare their performance:

python compare_algorithms.py

This will display graphs comparing two algorithms at a time.

Train Individual Algorithms: You can train individual algorithms by running their respective scripts

Monte Carlo Tree Search
SARSA
Q-Learning
Expected SARSA
Neural Networks
##Results
The project generates graphs comparing the performance of the algorithms. Each graph shows:

Total Rewards per Episode: The rewards obtained by the agent in each episode.
Smoothed Rewards: A moving average of rewards over the last 100 episodes.
Example comparisons:

Monte Carlo vs Expected SARSA
SARSA vs Q-Learning


##Code Highlights
Monte Carlo Tree Search:

Implements selection, expansion, simulation, and backpropagation phases.
Uses UCT to balance exploration and exploitation.
SARSA and Expected SARSA:

Updates Q-values using temporal difference learning.
Balances exploration and exploitation with an epsilon-greedy policy.
Q-Learning:

Updates Q-values using the maximum future reward.
Off-policy learning for better generalization.
Neural Networks:

Uses a feedforward neural network to approximate Q-values.
Trains using discounted rewards and categorical cross-entropy loss.